"""Graph building pipeline: document chunking, LLM extraction, Neo4j persistence."""

import json
import hashlib
import time
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from tqdm import tqdm

from rag_assistant.config import (
    CHECKPOINT_FILE,
    DATA_DIR,
    NEO4J_CONFIG_FILE,
    PROMPTS_DIR,
)
from rag_assistant.graph.models import (
    Edge,
    GraphResponse,
    Node,
    Property,
    Relationship,
)

# ---------------------------------------------------------------------------
# Module-level prompt loading
# ---------------------------------------------------------------------------

with open(PROMPTS_DIR / "query_nodes.txt", encoding="utf-8") as _f:
    _QUERY_NODES: str = _f.read()

with open(PROMPTS_DIR / "query_relations.txt", encoding="utf-8") as _f:
    _QUERY_RELATIONS: str = _f.read()

_COMBINED_PROMPT: str = _QUERY_NODES + "\n\n" + _QUERY_RELATIONS

_SPEAKER_SEPARATORS: list[str] = [
    r"\nRESPONDENT:\t",
    r"\nINFORMANT 1:\t",
    r"\nINFORMANT 2:\t",
    r"\nINFORMANT:\t",
    r"\nINFORMANT:",
    r"\nINTERVJUARE 1:\t",
    r"\nINTERVJUARE 2:\t",
    r"\nINTERVJUARE:\t",
    r"\nINTERVJUARE:",
    r"\nRespondent\n",
    r"\nInterviewer\n",
    r"\nR:\t",
    r"\nI:\t",
    r"\n[A-ZÅÄÖ][A-ZÅÄÖ\s\.]+:[\t ]",
    r"\n[A-ZÅÄÖ]{1,4}:[\t ]",
    r"\n",
]


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------


def _chunk_id(text: str) -> str:
    """Return a SHA-256 hex digest for the given text."""
    return hashlib.sha256(text.encode()).hexdigest()


def _sanitize_label(label: str) -> str:
    """Return a backtick-escaped Neo4j label with spaces/commas as underscores."""
    label = label.strip().replace(",", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return f"`{label.strip('_')}`"


def _props_to_dict(properties: list[Property]) -> dict[str, str]:
    """Convert a list of Property objects to a plain key/value dictionary."""
    return {prop.key: prop.value for prop in properties}


def _node_to_dict(node: Node) -> dict:
    """Serialize a Node model to the dict format expected by Graph.build_graph."""
    return {
        "node_label": node.node_label,
        "node_properties": _props_to_dict(node.node_properties),
    }


def _relationship_to_dict(relationship: Relationship) -> dict:
    """Serialize a Relationship model to the dict format expected by Graph.build_graph."""
    return {
        "relationship_type": relationship.relationship_type,
        "relationship_properties": _props_to_dict(
            relationship.relationship_properties
        ),
    }


def _call_with_retry(fn, *args, retries: int = 3, delay: float = 5.0, **kwargs):
    """Call *fn* with retry logic, sleeping *delay* seconds between attempts."""
    last_exception = Exception("No attempts made.")
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            if attempt < retries:
                time.sleep(delay)
    raise last_exception


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------


class Checkpoint:
    """Tracks which document chunks have already been processed.

    Persists progress to disk so that an interrupted run can resume
    from where it left off rather than reprocessing all chunks.
    """

    def __init__(self, path: Path = CHECKPOINT_FILE) -> None:
        self._path = path
        self._processed: set[str] = self._load()

    def _load(self) -> set[str]:
        if self._path.exists():
            with open(self._path, encoding="utf-8") as file:
                return set(json.load(file))
        return set()

    def save(self) -> None:
        """Persist the current set of processed chunk IDs to disk."""
        with open(self._path, "w", encoding="utf-8") as file:
            json.dump(list(self._processed), file)

    def contains(self, chunk_id: str) -> bool:
        """Return True if *chunk_id* has already been processed."""
        return chunk_id in self._processed

    def mark_done(self, chunk_id: str) -> None:
        """Mark *chunk_id* as processed and persist immediately."""
        self._processed.add(chunk_id)
        self.save()

    @property
    def count(self) -> int:
        """Number of chunks already processed."""
        return len(self._processed)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


class DocumentLoader:
    """Loads and chunks .docx documents from the data directory.

    Uses speaker-aware separators so that chunks respect turn boundaries
    in interview transcripts, rather than splitting mid-sentence.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self._data_dir = data_dir
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=True,
            separators=_SPEAKER_SEPARATORS,
        )

    def load_chunks(self) -> list:
        """Load all .docx files and return a flat list of document chunks."""
        docx_files = sorted(self._data_dir.rglob("*.docx"))
        if not docx_files:
            raise FileNotFoundError(
                f"No .docx files found in {self._data_dir}"
            )

        print(f"Found {len(docx_files)} .docx file(s).")
        all_chunks: list = []

        for file_path in docx_files:
            print(f"  Loading: {file_path.name}")
            try:
                documents = Docx2txtLoader(str(file_path)).load()
                chunks = self._splitter.split_documents(documents)
                print(f"    Chunks created: {len(chunks)}")
                all_chunks.extend(chunks)
            except Exception as exc:
                print(f"    Error loading {file_path.name}: {exc}")

        print(f"Total chunks: {len(all_chunks)}")
        return all_chunks


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------


class GraphExtractor:
    """Extracts nodes and relationships from text chunks using GPT-4o.

    Uses OpenAI's structured output API to guarantee that responses
    conform to the GraphResponse Pydantic schema.
    """

    def __init__(self, client) -> None:
        self._client = client

    def extract(self, chunk_text: str) -> GraphResponse:
        """Extract a GraphResponse (nodes + edges) from a single text chunk."""
        response = _call_with_retry(
            self._client.beta.chat.completions.parse,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _COMBINED_PROMPT},
                {"role": "user", "content": chunk_text},
            ],
            response_format=GraphResponse,
        )
        return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Neo4j graph
# ---------------------------------------------------------------------------


class Graph:
    """Manages all read/write interactions with a Neo4j graph database.

    Nodes are merged by label + properties so that the same entity
    appearing in multiple chunks is deduplicated into a single node.
    The originating chunk text is appended to each node's
    ``source_chunks`` list property for traceability.
    """

    def __init__(self, config_file: Path = NEO4J_CONFIG_FILE) -> None:
        self._driver = self._create_driver(config_file)
        if not self._ping():
            self._driver.close()
            raise RuntimeError("Neo4j is unreachable. Is it running?")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _create_driver(config_file: Path):
        with open(config_file, encoding="utf-8") as file:
            cfg = json.load(file)
        url = f"{cfg['scheme']}://{cfg['host_name']}:{cfg['port']}"
        return GraphDatabase.driver(url, auth=(cfg["user"], cfg["password"]))

    def _ping(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the underlying Neo4j driver connection."""
        self._driver.close()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def create_node(
        self,
        label: str,
        properties: dict[str, str],
        source_chunk: str | None = None,
    ) -> str | None:
        """MERGE a node by label + properties and return its element ID.

        If *source_chunk* is provided it is appended to the node's
        ``source_chunks`` list without affecting the MERGE identity.
        """
        safe_label = _sanitize_label(label)
        prop_clause = ", ".join(f"{k}: ${k}" for k in properties)

        if source_chunk:
            query = f"""
                MERGE (n:{safe_label} {{ {prop_clause} }})
                SET n.source_chunks = coalesce(n.source_chunks, []) + $source_chunk
                RETURN elementId(n) AS element_id
            """
            params = {**properties, "source_chunk": source_chunk}
        else:
            query = f"""
                MERGE (n:{safe_label} {{ {prop_clause} }})
                RETURN elementId(n) AS element_id
            """
            params = properties

        try:
            with self._driver.session() as session:
                record = session.run(query, params).single()
                return record["element_id"] if record else None
        except Exception as exc:
            print(f"Error creating node {safe_label}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    def create_relationship(
        self,
        start_element_id: str,
        end_element_id: str,
        rel_type: str,
        properties: dict[str, str] | None = None,
    ) -> bool:
        """MERGE a directed relationship between two nodes by element ID."""
        safe_type = _sanitize_label(rel_type)
        params: dict = {
            "start_id": start_element_id,
            "end_id": end_element_id,
        }

        if properties:
            prop_clause = ", ".join(f"{k}: ${k}" for k in properties)
            query = f"""
                MATCH (a), (b)
                WHERE elementId(a) = $start_id AND elementId(b) = $end_id
                MERGE (a)-[r:{safe_type} {{ {prop_clause} }}]->(b)
                RETURN type(r)
            """
            params.update(properties)
        else:
            query = f"""
                MATCH (a), (b)
                WHERE elementId(a) = $start_id AND elementId(b) = $end_id
                MERGE (a)-[r:{safe_type}]->(b)
                RETURN type(r)
            """

        try:
            with self._driver.session() as session:
                return session.run(query, params).single() is not None
        except Exception as exc:
            print(f"Error creating relationship {safe_type}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(
        self,
        nodes: list[dict],
        edges: list[tuple[dict, dict, dict]],
        source_chunk: str | None = None,
    ) -> None:
        """Create all nodes then wire up relationships.

        Caches element IDs locally to avoid redundant database round-trips
        when the same node appears on both sides of multiple edges.
        """
        def _cache_key(node: dict) -> tuple:
            return (
                node["node_label"],
                tuple(sorted(node["node_properties"].items())),
            )

        id_cache: dict[tuple, str] = {}
        for node in nodes:
            element_id = self.create_node(
                node["node_label"],
                node["node_properties"],
                source_chunk,
            )
            if element_id:
                id_cache[_cache_key(node)] = element_id

        for source_node, target_node, relationship in edges:
            source_id = id_cache.get(_cache_key(source_node)) or self.create_node(
                source_node["node_label"],
                source_node["node_properties"],
                source_chunk,
            )
            target_id = id_cache.get(_cache_key(target_node)) or self.create_node(
                target_node["node_label"],
                target_node["node_properties"],
                source_chunk,
            )

            if source_id and target_id:
                self.create_relationship(
                    source_id,
                    target_id,
                    relationship["relationship_type"],
                    relationship["relationship_properties"],
                )
            else:
                print(
                    f"Warning: skipping edge — "
                    f"{source_node} -> {target_node}"
                )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


class GraphBuildingPipeline:
    """Orchestrates the end-to-end graph construction process.

    Ties together document loading, LLM extraction, Neo4j persistence,
    and checkpoint management into a single reusable pipeline.
    """

    def __init__(
        self,
        graph: Graph,
        extractor: GraphExtractor,
        checkpoint: Checkpoint,
    ) -> None:
        self._graph = graph
        self._extractor = extractor
        self._checkpoint = checkpoint

    def run(self, chunks: list) -> None:
        """Process each chunk and write extracted nodes/edges to Neo4j."""
        skipped = sum(
            1 for chunk in chunks
            if self._checkpoint.contains(_chunk_id(chunk.page_content))
        )
        if skipped:
            print(f"Checkpoint: skipping {skipped} already-processed chunk(s).")

        for index, chunk in enumerate(
            tqdm(chunks, desc="Building graph", unit="chunk")
        ):
            chunk_hash = _chunk_id(chunk.page_content)
            if self._checkpoint.contains(chunk_hash):
                continue

            try:
                graph_response = self._extractor.extract(chunk.page_content)

                nodes = [_node_to_dict(node) for node in graph_response.nodes]
                edges = [
                    (
                        _node_to_dict(edge.source),
                        _node_to_dict(edge.target),
                        _relationship_to_dict(edge.relationship),
                    )
                    for edge in graph_response.edges
                ]

                self._graph.build_graph(nodes, edges, source_chunk=chunk.page_content)
                self._checkpoint.mark_done(chunk_hash)

            except Exception as exc:
                print(f"Error on chunk {index + 1}: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: load documents, build graph, close connection."""
    from openai import OpenAI

    client = OpenAI()
    graph = Graph()
    extractor = GraphExtractor(client)
    checkpoint = Checkpoint()
    loader = DocumentLoader()

    chunks = loader.load_chunks()
    pipeline = GraphBuildingPipeline(graph, extractor, checkpoint)
    pipeline.run(chunks)
    graph.close()
