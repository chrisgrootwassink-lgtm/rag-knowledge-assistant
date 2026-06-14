"""Streamlit RAG chatbot: translate query, retrieve chunks, synthesise answer."""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings

from rag_assistant.config import CHROMA_PATH, VECTORSTORE_COLLECTION

load_dotenv()

_SYNTHESIS_SYSTEM_PROMPT: str = """
You are a rigorous research analyst. Answer questions using ONLY the \
retrieved document excerpts below. Do not draw on outside knowledge — \
if the documents do not address something, say so clearly and suggest \
a more targeted query.

RETRIEVED DOCUMENT EXCERPTS:
{vector_data}

INSTRUCTIONS:

**SCOPE:** Base every claim strictly on the retrieved excerpts.

**LANGUAGE:** Write entirely in English.

**LENGTH & DEPTH:** Minimum 6–8 substantial paragraphs. Analyse rather \
than summarise. For every claim:
- State what the evidence says and what it implies.
- Identify tensions, contradictions, or patterns across excerpts.
- Where multiple perspectives appear, compare them explicitly.

**STRUCTURE:** Use markdown headers (##, ###) to organise sections. \
Use bullet points for sub-points anchored in prose analysis.

**BROAD QUESTIONS:** Respond with:
1. A thematic overview of what the documents collectively reveal.
2. A breakdown of key sub-themes or threads.
3. Three specific follow-up questions the user could explore.

**CLOSING:** End with "### Further Research Directions" suggesting \
2–3 follow-up angles grounded in the documents.
"""

_TRANSLATION_PROMPT: str = """
Translate the following text to Swedish. Output ONLY the translation.
If the text is already in Swedish, output it unchanged.
Text: {question}
"""

_CHAT_HISTORY_WINDOW: int = 10
_RETRIEVAL_K: int = 15
_RETRIEVAL_FETCH_K: int = 40


# ---------------------------------------------------------------------------
# Cached resource initialisation
# ---------------------------------------------------------------------------


@st.cache_resource
def get_vector_store() -> Chroma | None:
    """Connect to the persisted ChromaDB collection (cached across reruns)."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    try:
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=VECTORSTORE_COLLECTION,
        )
    except Exception as exc:
        st.error(f"Error loading vector store: {exc}")
        return None


@st.cache_resource
def get_chains() -> tuple:
    """Initialise and return (synthesis_chain, translation_chain) (cached)."""
    opus = ChatAnthropic(
        model_name="claude-opus-4-6",
        temperature=0.4,
        timeout=60,
        stop=[],
    )
    haiku = ChatAnthropic(
        model_name="claude-haiku-4-5-20251001",
        temperature=0,
        timeout=30,
        stop=[],
    )

    synthesis_chain = (
        ChatPromptTemplate.from_messages([
            ("system", _SYNTHESIS_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        | opus
        | StrOutputParser()
    )

    translation_chain = (
        ChatPromptTemplate.from_template(_TRANSLATION_PROMPT)
        | haiku
        | StrOutputParser()
    )

    return synthesis_chain, translation_chain


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _retrieve_documents(
    vector_store: Chroma | None, query: str
) -> list:
    """Return the most relevant document chunks for *query* using MMR."""
    if vector_store is None:
        return []
    if hasattr(vector_store, "max_marginal_relevance_search"):
        return vector_store.max_marginal_relevance_search(
            query, k=_RETRIEVAL_K, fetch_k=_RETRIEVAL_FETCH_K
        )
    return vector_store.similarity_search(query, k=_RETRIEVAL_K)


def _format_retrieved_docs(documents: list) -> str:
    """Format a list of documents into a numbered context block."""
    if not documents:
        return "No relevant context found."

    def _fmt(index: int, doc) -> str:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        page_suffix = f", page {page}" if page != "" else ""
        return f"[{index + 1}] Source: {source}{page_suffix}\n{doc.page_content}"

    return "\n\n".join(_fmt(i, doc) for i, doc in enumerate(documents))


def _render_chat_history(chat_history: list) -> None:
    """Render all messages in *chat_history* into the Streamlit UI."""
    for message in chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the Streamlit research assistant application."""
    st.set_page_config(
        page_title="Research Assistant",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Research Assistant")
    st.caption(
        "Ask questions grounded in the document collection. "
        "Responses are based exclusively on the ingested documents."
    )

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
        st.stop()

    vector_store = get_vector_store()
    synthesis_chain, translation_chain = get_chains()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    _render_chat_history(st.session_state.chat_history)

    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and synthesizing..."):
                translated_query = translation_chain.invoke(
                    {"question": prompt}
                )
                documents = _retrieve_documents(vector_store, translated_query)
                context = _format_retrieved_docs(documents)
                recent_history = st.session_state.chat_history[
                    -_CHAT_HISTORY_WINDOW:
                ]

                response = st.write_stream(
                    synthesis_chain.stream({
                        "question": prompt,
                        "vector_data": context,
                        "chat_history": recent_history,
                    })
                )

        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=response))
