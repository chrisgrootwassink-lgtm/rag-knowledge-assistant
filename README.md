# RAG Knowledge Assistant

A dual-tool system for analysing qualitative document collections using **Retrieval-Augmented Generation (RAG)** and **LLM-powered knowledge graphs**.

## Tech Stack

| Layer | Technology |
|---|---|
| LLM (synthesis) | Claude Opus (Anthropic) |
| LLM (extraction) | GPT-4o (OpenAI) |
| LLM (translation) | Claude Haiku / GPT-4o-mini |
| Vector store | ChromaDB |
| Embeddings | OpenAI `text-embedding-3-large` |
| Graph database | Neo4j |
| Frontend | Streamlit |
| Orchestration | LangChain |

## Architecture

```
Documents (.docx)
      │
      ├─► vectorstore_creator.py ──► ChromaDB (vector store)
      │                                      │
      │                               chatbot.py (Streamlit)
      │                               ┌──────────────────┐
      │                               │ 1. Translate query│
      │                               │ 2. MMR retrieval  │
      │                               │ 3. Claude Opus    │
      │                               └──────────────────┘
      │
      └─► graphbuilder.py ──► Neo4j knowledge graph
                GPT-4o extracts nodes + relationships per chunk
                Checkpointing allows interrupted runs to resume
```

## Project Structure

```
chatbot.py               — Streamlit RAG chatbot
graphbuilder.py          — Neo4j knowledge-graph builder
vectorstore_creator.py   — Builds the ChromaDB vector store from documents
dependencies.txt         — Python package requirements (pip install -r)
neo4j_config.json        — Neo4j connection settings
query_nodes.txt          — LLM prompt for entity extraction
query_relations.txt      — LLM prompt for relationship extraction
data/                    — Place your .docx documents here (not committed)
vectorstore/             — Auto-generated ChromaDB storage (not committed)
```

## Setup

**1. Install dependencies**
```bash
pip install -r dependencies.txt
```

**2. Configure API keys** — copy `.env.example` to `.env` and fill in your keys:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**3. Configure Neo4j** (only needed for the graph builder) — edit `neo4j_config.json`:
```json
{
  "scheme": "bolt",
  "host_name": "localhost",
  "port": 7687,
  "user": "neo4j",
  "password": "your-password"
}
```

**4. Add your documents** — place `.docx` files in the `data/` folder (subfolders supported).

## Usage

**Step 1 — Build the vector store** (once, or when documents change):
```bash
python "vectorstore_creator.py"
```

**Step 2 — Launch the chatbot:**
```bash
streamlit run chatbot.py
```

**Step 3 (optional) — Build the Neo4j knowledge graph:**
```bash
python graphbuilder.py
```
Progress is saved to `checkpoint.json` after each chunk — interrupted runs resume automatically.

## How the Chatbot Works

1. The user's question is translated to the document language by Claude Haiku.
2. The translated query retrieves the 15 most relevant chunks from ChromaDB using **Maximal Marginal Relevance (MMR)** search.
3. The retrieved chunks and the original question are sent to **Claude Opus**, which produces a structured analytical response.
4. The last 10 messages are retained as chat history for follow-up questions.

## How the Graph Builder Works

1. Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`.
2. Each chunk is sent to **GPT-4o** with structured prompts to extract entities (nodes) and relationships (edges).
3. Nodes and edges are merged into Neo4j using `MERGE` — the same entity from different chunks becomes a single node.
4. A SHA-256 checkpoint tracks processed chunks so the script can resume after interruption.
