# RAG Knowledge Assistant

A dual-tool system for analysing qualitative document collections using **Retrieval-Augmented Generation (RAG)** and **LLM-powered knowledge graphs**.

## Tech Stack

| Layer | Technology |
|---|---|
| LLM (synthesis) | Claude Opus (Anthropic) |
| LLM (extraction) | GPT-4o (OpenAI) |
| LLM (translation / routing) | Claude Haiku / GPT-4o-mini |
| Vector store | ChromaDB |
| Embeddings | OpenAI `text-embedding-3-large` |
| Graph database | Neo4j |
| Frontend | Streamlit |
| Orchestration | LangChain |

## Architecture

```
Documents (.docx)
      │
      ├─► create_demo_data.py ──► data/  (synthetic demo set)
      │
      ├─► vectorstore_creator.py ──► ChromaDB (vector store)
      │                                      │
      │                               chatbot.py (Streamlit)
      │                               ┌──────────────────────┐
      │                               │ 1. Translate query    │
      │                               │ 2. MMR retrieval (k=15)│
      │                               │ 3. Claude Opus synth  │
      │                               └──────────────────────┘
      │
      └─► graphbuilder.py ──► Neo4j knowledge graph
                GPT-4o extracts nodes + relationships per chunk
                Checkpointing allows interrupted runs to resume
```

---

## Quick Start (with synthetic demo data)

The repo ships with a demo data generator — no private documents needed.

**1. Install dependencies**
```bash
pip install -r dependencies.txt
```

**2. Configure API keys** — copy `.env.example` to `.env` and fill in your keys:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**3. Generate demo documents**
```bash
python create_demo_data.py
```
This creates three synthetic interview transcripts in `data/` covering cloud infrastructure, software development practices, and AI/ML adoption at a fictional company.

**4. Build the vector store**
```bash
python "vectorstore creator.py"
```

**5. Launch the chatbot**
```bash
streamlit run chatbot.py
```

**Try asking:**
- *"What cloud platforms are used and why was Azure chosen?"*
- *"How does the team manage CI/CD and code quality?"*
- *"What challenges came up with LLM adoption?"*
- *"How is model monitoring handled in production?"*

---

## Using Your Own Documents

Place any `.docx` files in the `data/` folder (subfolders supported), then re-run steps 4 and 5 above. The system supports any document collection — the demo uses English documents; for Swedish content the chatbot automatically translates queries before retrieval.

---

## Optional: Build the Knowledge Graph

Requires a running Neo4j instance (default: `bolt://localhost:7687`). Edit `neo4j_config.json` with your credentials, then:

```bash
python graphbuilder.py
```

Progress is saved to `checkpoint.json` after each chunk — interrupted runs resume automatically.

---

## Project Structure

```
chatbot.py               — Streamlit RAG chatbot
graphbuilder.py          — Neo4j knowledge-graph builder
vectorstore creator.py   — Builds the ChromaDB vector store from documents
create_demo_data.py      — Generates synthetic demo documents in data/
dependencies.txt         — Python package requirements
neo4j_config.json        — Neo4j connection settings (localhost defaults)
query_nodes.txt          — LLM prompt for entity extraction
query_relations.txt      — LLM prompt for relationship extraction
tests/                   — Unit tests (no API keys required)
data/                    — Place .docx documents here (not committed)
vectorstore/             — Auto-generated ChromaDB storage (not committed)
```

---

## Running Tests

```bash
pytest tests/
```

Tests cover core utility functions (hashing, label sanitisation, Pydantic models, separator patterns) and require no API keys or external services.

---

## How the Chatbot Works

1. The user's question is translated to the document language by Claude Haiku.
2. The translated query retrieves the 15 most relevant chunks from ChromaDB using **Maximal Marginal Relevance (MMR)** search.
3. The retrieved chunks and the original question are sent to **Claude Opus**, which produces a structured analytical response.
4. The last 10 messages are retained as chat history for follow-up questions.

## How the Graph Builder Works

1. Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter` with speaker-aware separators.
2. Each chunk is sent to **GPT-4o** with structured prompts to extract entities (nodes) and relationships (edges).
3. Nodes and edges are merged into Neo4j using `MERGE` — the same entity appearing in different chunks becomes a single node.
4. A SHA-256 checkpoint tracks processed chunks so the script can resume after interruption.
