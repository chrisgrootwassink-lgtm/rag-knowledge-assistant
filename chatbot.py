import streamlit as st
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# 1. Setup & Configuration
load_dotenv()
st.set_page_config(page_title="Research Assistant", page_icon="🔬", layout="wide")
st.title("Research Assistant")
st.caption("Ask research questions grounded in the interview data. Responses are based exclusively on the document collection.")

CHROMA_PATH = "./vectorstore"

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("Error: ANTHROPIC_API_KEY missing in environment variables.")
    st.stop()

# 2. Initialize Resources (Cached)
@st.cache_resource
def get_vector_store():
    """Connects to the existing ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    try:
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="interview_documents"
        )
        return vector_store
    except Exception as e:
        st.error(f"Error loading Chroma: {e}")
        return None

@st.cache_resource
def get_chains():
    """Initializes the LLM chains."""
    llm = ChatAnthropic(model_name="claude-opus-4-6", temperature=0.4, timeout=60, stop=[])
    fast_llm = ChatAnthropic(model_name="claude-haiku-4-5-20251001", temperature=0, timeout=30, stop=[])
    
    # Chain B: Synthesizing
    synthesis_template = """
You are a rigorous research analyst. Your role is to answer research questions using ONLY the retrieved document excerpts below. Do not draw on outside knowledge — if the documents do not address something, say so clearly and suggest a more targeted query.

RETRIEVED DOCUMENT EXCERPTS:
{vector_data}

INSTRUCTIONS:

**SCOPE:** Base every claim strictly on the retrieved excerpts. If a question falls outside what the documents address, state this explicitly and propose a refined question.

**LANGUAGE:** Write entirely in English.

**LENGTH & DEPTH:** Minimum 6–8 substantial paragraphs. Analyse rather than summarise. For every claim:
- State what the evidence says and what it implies.
- Identify tensions, contradictions, or patterns across excerpts.
- Where multiple perspectives appear on the same topic, compare them explicitly.

**STRUCTURE:** Use markdown headers (##, ###) to organise the response into clearly labelled analytical sections. Use bullet points for sub-points, but anchor them in prose analysis.

**BROAD QUESTIONS:** If the question is wide in scope, respond with:
1. A thematic overview of what the documents collectively reveal.
2. A breakdown of key sub-themes or threads.
3. Three specific follow-up questions the user could explore.

**CLOSING:** End with a section titled "### Further Research Directions" suggesting 2–3 concrete follow-up angles grounded in what the documents reveal.
"""
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", synthesis_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()

    # Chain C: Translation (Query to Swedish)
    translation_template = """
    Translate the following text to Swedish. Output ONLY the translation.
    If the text is already in Swedish, just output it as is.
    Text: {question}
    """
    translation_prompt = ChatPromptTemplate.from_template(translation_template)
    translation_chain = translation_prompt | fast_llm | StrOutputParser()

    return synthesis_chain, translation_chain

vector_store = get_vector_store()
synthesis_chain, translation_chain = get_chains()

# 3. Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if prompt := st.chat_input("Ask a research question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching and Synthesizing..."):
            # 1. Translate
            swedish_query = translation_chain.invoke({"question": prompt})
            
            # 2. Search Vectors — use Maximal Marginal Relevance (MMR) if available,
            # otherwise fall back to the original similarity search.
            if vector_store:
                if hasattr(vector_store, "max_marginal_relevance_search"):
                    docs = vector_store.max_marginal_relevance_search(
                        swedish_query,
                        k=15,
                        fetch_k=40,
                    )
                else:
                    docs = vector_store.similarity_search(swedish_query, k=15)
            else:
                docs = []
            def format_doc(i, d):
                meta = d.metadata
                source = meta.get("source", "Unknown source")
                page = meta.get("page", "")
                page_str = f", page {page}" if page != "" else ""
                return f"[{i+1}] Source: {source}{page_str}\n{d.page_content}"

            vector_context = "\n\n".join([format_doc(i, d) for i, d in enumerate(docs)]) if docs else "No text context found."

            # 3. Synthesize (stream response, cap history to last 10 messages)
            recent_history = st.session_state.chat_history[-10:]
            response = st.write_stream(synthesis_chain.stream({
                "question": prompt,
                "vector_data": vector_context,
                "chat_history": recent_history,
            }))
    
    # Update History
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=response))
