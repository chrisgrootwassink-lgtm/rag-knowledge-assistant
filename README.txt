================================================================================
  RAG Knowledge Assistant — README
================================================================================

OVERVIEW
--------
This project provides two complementary tools for analysing qualitative
interview documents (.docx), written primarily in Swedish:

  1. Vector RAG Chatbot  (chatbot.py)
     A Streamlit web app that lets you ask questions about your documents.
     Queries are translated to Swedish, matched against a ChromaDB vector store,
     and answered in English by Claude (Anthropic) with inline citations.

  2. Knowledge Graph Builder  (graphbuilder.py)
     Extracts entities and relationships from document chunks using GPT-4o and
     stores them as a graph in a local Neo4j database. Supports checkpointing
     so interrupted runs can resume where they left off.

  The vector store is built separately with "vectorstore creator.py" before
  running the chatbot.


PROJECT STRUCTURE
-----------------
  chatbot.py               — Streamlit RAG chatbot (run this to use the app)
  graphbuilder.py          — Neo4j knowledge-graph builder
  vectorstore creator.py   — Builds the ChromaDB vector store from documents
  dependencies.txt         — All required Python packages (pip install -r)
  neo4j_config.json        — Neo4j connection settings
  query_nodes.txt          — LLM prompt for node extraction (graphbuilder)
  query_relations.txt      — LLM prompt for edge extraction (graphbuilder)
  checkpoint.json          — Auto-generated; tracks graph-builder progress
  data/                    — Place your .docx interview files here
  vectorstore/             — Auto-generated; ChromaDB persistent storage


PREREQUISITES
-------------
  - Python 3.10+
  - A running Neo4j instance (for graphbuilder.py only)
      Default connection: bolt://localhost:7687
      Edit neo4j_config.json to change host, port, or credentials.
  - API keys for OpenAI and Anthropic


SETUP
-----
1. Install dependencies:

     pip install -r dependencies.txt

2. Create a .env file in this folder with your API keys:

     OPENAI_API_KEY=sk-...
     ANTHROPIC_API_KEY=sk-ant-...

3. Place your .docx interview documents inside the data/ folder.
   Subfolders are supported.


USAGE
-----

  Step 1 — Build the vector store (only needed once, or when documents change):

     python "vectorstore creator.py"

  Step 2 — Launch the chatbot:

     streamlit run chatbot.py

  Step 3 (optional) — Build the Neo4j knowledge graph:

     Ensure Neo4j is running, then:
     python graphbuilder.py

     Progress is saved to checkpoint.json after each chunk. If the run is
     interrupted, re-running the script will skip already-processed chunks.


HOW THE CHATBOT WORKS
---------------------
  1. Your question is translated to Swedish by Claude Haiku.
  2. The Swedish query is used to retrieve the 15 most relevant chunks from
     ChromaDB (using Maximal Marginal Relevance search).
  3. The retrieved chunks plus your original question are sent to Claude Opus,
     which produces a structured analysis with inline citations.
  4. The last 10 messages are kept as chat history for follow-up questions.


NOTES
-----
  - The neo4j_config.json file contains database credentials. Do not share it
    or commit it to a public repository.
  - The .env file containing API keys should also be kept private.
  - The graphbuilder uses GPT-4o for graph extraction and GPT-4o-mini for
    translation of chunks prior to extraction.
  - The vector store uses OpenAI's text-embedding-3-large model.


NOTE: 
The graphbuilder uses a lot of resources due to its layered architecture. Therefore the cost of building graphs is relatively high. Make sure to only include relevant documents. 
================================================================================
