import json
import hashlib
import time
from pathlib import Path
from dateutil.parser import parse as date_parse
from neo4j import GraphDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from pydantic import BaseModel
from tqdm import tqdm

## Checkpointing ensures the graph builder resumes from where it left off instead of reprocessing all chunks on every run.
class Property(BaseModel):
    key: str
    value: str


class Node(BaseModel):
    node_label: str
    node_properties: list[Property]


class Relationship(BaseModel):
    relationship_type: str
    relationship_properties: list[Property]


class Edge(BaseModel):
    source: Node
    target: Node
    relationship: Relationship


class NodesResponse(BaseModel):
    nodes: list[Node]


class EdgesResponse(BaseModel):
    edges: list[Edge]


class GraphResponse(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


# here you are loading the necessary documents into the query variable, change this to iterate over an entire map of documents
_HERE = Path(__file__).parent
with open(_HERE / 'query_nodes.txt', 'r') as file:
    query_nodes = file.read()
with open(_HERE / 'query_relations.txt', 'r') as file:
    query_relations = file.read()

DATA_DIR = Path(__file__).parent / "data" #adjust if neccessary, this is where your .docx files should be placed
CHECKPOINT_FILE = Path(__file__).parent / "checkpoint.json"


def _chunk_id(chunk_text: str) -> str:
    return hashlib.sha256(chunk_text.encode()).hexdigest()


def _load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f))
    return set()


def _save_checkpoint(processed: set):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(processed), f)


def _call_with_retry(fn, *args, retries=3, delay=5, **kwargs):
    """Call fn(*args, **kwargs), retrying up to `retries` times on any exception."""
    last_exc: Exception = Exception("No attempts made")
    for attempt in range(1, retries + 2):  # attempts: 1 … retries+1
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt <= retries:
                time.sleep(delay)
    raise last_exc


def text_cutting():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        is_separator_regex=True,
        separators=[
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
            r"\n[A-ZÅÄÖ][A-ZÅÄÖ\s\.]+:[\t ]",  # name labels like "ANDERS S:\t"
            r"\n[A-ZÅÄÖ]{1,4}:[\t ]",           # short initials like "JK: ", "BÅ: ", "LR: "
            r"\n",
        ],
    )

    all_docx_files = sorted(DATA_DIR.rglob("*.docx"))
    if not all_docx_files:
        print(f"No .docx files found in {DATA_DIR}")
        exit(1)

    print(f"Found {len(all_docx_files)} .docx file(s) total")

    all_chunks = []
    for file_path in all_docx_files:
        print(f"--- Loading: {file_path.name} ---")
        try:
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            print(f"  Chunks created: {len(chunks)}")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def _sanitize_label(label: str) -> str:
    """Sanitize a Neo4j node label by replacing spaces/commas with underscores and backtick-escaping it."""
    label = label.strip().replace(",", "_").replace(" ", "_")
    # Remove consecutive underscores
    while "__" in label:
        label = label.replace("__", "_")
    label = label.strip("_")
    return f"`{label}`"


class Graph:
    """
    Graph class encapsulates all interactions with the Neo4j database.
    """

    def __init__(self, config_file=Path(__file__).parent / "neo4j_config.json"):
        """
        Initializes the Graph instance by loading configuration and setting up the Neo4j driver.
        
        Args:
            config_file (str): Path to the Neo4j configuration file.
        """
        self.driver = self.init_graph_DB(config_file)
        if not self.graph_DB_running():
            print("Neo4j database is not running!")
            self.driver.close()
            exit(1)

    def init_graph_DB(self, config_file):
        """
        Initializes the Neo4j driver using the provided configuration file.

        Args:
            config_file (str): Path to the Neo4j configuration file.

        Returns:
            neo4j.GraphDatabase.driver: Neo4j driver instance.
        """
        with open(config_file) as f:
            config = json.load(f)

        scheme = config["scheme"]
        host_name = config["host_name"]
        port = config["port"]
        url = f"{scheme}://{host_name}:{port}"
        user = config["user"]
        password = config["password"]

        driver = GraphDatabase.driver(url, auth=(user, password))
        return driver

    def graph_DB_running(self):
        """
        Checks if the Neo4j database is reachable.
        """
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def create_node(self, label, properties, source_chunk=None):
        """
        Creates or matches a node via MERGE on label + properties.
        If source_chunk is provided, it is appended to a 'source_chunks' list
        property using SET (so it never affects the MERGE identity).
        """
        label = _sanitize_label(label)

        if source_chunk:
            query = f"""
            MERGE (n:{label} {{
                {', '.join(f'{k}: ${k}' for k in properties.keys())}
            }})
            SET n.source_chunks = coalesce(n.source_chunks, []) + $source_chunk
            RETURN elementId(n) AS element_id
            """
            params = {**properties, "source_chunk": source_chunk}
        else:
            query = f"""
            MERGE (n:{label} {{
                {', '.join(f'{k}: ${k}' for k in properties.keys())}
            }})
            RETURN elementId(n) AS element_id
            """
            params = properties

        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                record = result.single()
                return record["element_id"] if record else None
        except Exception as e:
            print(f"Error creating node {label} with properties {properties}: {e}")
            return None
        


    def create_relationship(self, start_element_id, end_element_id, rel_type, properties=None):
        """
        Creates a relationship between two nodes using elementId, avoiding issues with empty property dictionaries.
        """
        
        rel_type = _sanitize_label(rel_type)
        # Check if properties exist
        if properties:
            query = f"""
            MATCH (a), (b)
            WHERE elementId(a) = $start_element_id AND elementId(b) = $end_element_id
            MERGE (a)-[r:{rel_type} {{ {', '.join(f'{k}: ${k}' for k in properties.keys())} }}]->(b)
            RETURN type(r) AS rel_type
            """
        else:
            query = f"""
            MATCH (a), (b)
            WHERE elementId(a) = $start_element_id AND elementId(b) = $end_element_id
            MERGE (a)-[r:{rel_type}]->(b)
            RETURN type(r) AS rel_type
            """

        if properties is None:
            properties = {}

        params = {
            "start_element_id": start_element_id,
            "end_element_id": end_element_id,
            **properties
        }

        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                return result.single() is not None  # Returns True if the relationship was created
        except Exception as e:
            print(f"Error creating relationship {rel_type} between {start_element_id} and {end_element_id}: {e}")
            return False

    def check_node_exists(self, label, properties):
        """
        Checks if a node with the given label and properties exists.

        Args:
            label (str): Label of the node.
            properties (dict): Properties to match the node.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        label = _sanitize_label(label)
        query = f"""
        MATCH (n:{label} {{
            {', '.join(f'{k}: ${k}' for k in properties.keys())}
        }})
        RETURN COUNT(n) > 0 AS exists
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, properties)
                record = result.single()
                return record["exists"] if record else False
        except Exception as e:
            print(f"Error checking existence of node {label} with properties {properties}: {e}")
            return False
        
    def check_relationship_exists(self, start_label, start_props, end_label, end_props, rel_type):
        """
        Checks if a relationship of the specified type exists between two nodes.

        Args:
            start_label (str): Label of the start node.
            start_props (dict): Properties to match the start node.
            end_label (str): Label of the end node.
            end_props (dict): Properties to match the end node.
            rel_type (str): Type of the relationship.

        Returns:
            bool: True if the relationship exists, False otherwise.
        """
        # Exclude the 'text' property from start_props for matching
        filtered_start_props = {k: v for k, v in start_props.items() if k != "text"}
        start_label = _sanitize_label(start_label)
        end_label = _sanitize_label(end_label)
        rel_type = _sanitize_label(rel_type)

        query = f"""
        MATCH (a:{start_label} {{
            {', '.join(f'{k}: $start_{k}' for k in filtered_start_props.keys())}
        }})-[r:{rel_type}]->(b:{end_label} {{
            {', '.join(f'{k}: $end_{k}' for k in end_props.keys())}
        }})
        RETURN COUNT(r) > 0 AS exists
        """
        params = {f"start_{k}": v for k, v in filtered_start_props.items()}
        params.update({f"end_{k}": v for k, v in end_props.items()})

        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                record = result.single()
                return record["exists"] if record else False
        except Exception as e:
            print(f"Error checking relationship {rel_type} between {start_label} and {end_label}: {e}")
            return False

    def get_node(self, element_id):
        """
        Retrieves a node's properties and label based on its internal Neo4j element ID.

        Args:
            element_id (int): The internal Neo4j element ID of the node.

        Returns:
            dict or None: A dictionary containing the node's label and properties if found, else None.
        """
        query = """
        MATCH (n)
        WHERE elementId(n) = $element_id
        RETURN labels(n) AS labels, n AS node
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, {"element_id": element_id})
                record = result.single()
                if record:
                    # Extract the label and properties
                    labels = record["labels"]
                    properties = dict(record["node"])
                    return {
                        "node_label": labels[0] if labels else None,  # Assuming a single label
                        "node_properties": properties
                    }
                else:
                    print(f"No node found with element_id: {element_id}")
                    return None
        except Exception as e:
            print(f"Error retrieving node with element_id {element_id}: {e}")
            return None

    def get_element_ids(self, label, properties=None):
        """
        Retrieves the element_id(s) of nodes with the given label and properties.
        If properties is None, retrieves all element_ids for the given label.

        Args:
            label (str): The label of the nodes to match.
            properties (dict, optional): A dictionary of properties to match the nodes. Defaults to None.

        Returns:
            list of int: A list of element_ids that match the criteria.
        """
        label = _sanitize_label(label)
        if properties:
            prop_query = ', '.join(f'{k}: ${k}' for k in properties.keys())
            match_clause = f"MATCH (n:{label} {{{prop_query}}})"
            query = f"""
            {match_clause}
            RETURN elementId(n) AS element_id
            """
            try:
                with self.driver.session() as session:
                    result = session.run(query, properties)
                    return [record["element_id"] for record in result]
            except Exception as e:
                print(f"Error retrieving element_ids for label {label} with properties {properties}: {e}")
                return []
        else:
            query = f"""
            MATCH (n:{label})
            RETURN elementId(n) AS element_id
            """
            try:
                with self.driver.session() as session:
                    result = session.run(query)
                    return [record["element_id"] for record in result]
            except Exception as e:
                print(f"Error retrieving all element_ids for label {label}: {e}")
                return []

    def get_element_ids_from_query(self, query):
        """
        Executes a Cypher query and retrieves the element_id(s) of matching nodes.

        Args:
            query (str): The Cypher query to execute.

        Returns:
            list of int: A list of element_ids that match the query.
        """
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return [record["element_id"] for record in result]
        except Exception as e:
            print(f"Error executing query: {query}: {e}")
            return []
        
    def build_graph(self, nodes, edges, source_chunk=None):
        """
        Builds the graph by first creating all nodes and then adding relationships.
        If source_chunk is provided it is stored as a list property on each node
        (appended on every MERGE hit) without affecting node identity.

        Args:
            nodes (list of dict): List of node dictionaries.
            edges (list of tuples): List of tuples (start_node, end_node, relationship).
            source_chunk (str, optional): The originating chunk text.
        """

        def _cache_key(node):
            return (node["node_label"], tuple(sorted(node["node_properties"].items())))

        # Step 1: Create all nodes, caching their element IDs
        id_cache = {}
        for node in nodes:
            element_id = self.create_node(node["node_label"], node["node_properties"], source_chunk=source_chunk)
            if element_id:
                id_cache[_cache_key(node)] = element_id

        # Step 2: Create relationships, using cached IDs to avoid redundant DB calls
        for source_node, target_node, relationship in edges:
            source_id = id_cache.get(_cache_key(source_node)) or self.create_node(
                source_node["node_label"], source_node["node_properties"], source_chunk=source_chunk
            )
            target_id = id_cache.get(_cache_key(target_node)) or self.create_node(
                target_node["node_label"], target_node["node_properties"], source_chunk=source_chunk
            )

            if source_id and target_id:
                self.create_relationship(
                    start_element_id=source_id,
                    end_element_id=target_id,
                    rel_type=relationship["relationship_type"],
                    properties=relationship["relationship_properties"]
                )
            else:
                print(f" Warning: Could not create relationship: {source_node} -> {target_node}")

    def close(self):
        """
        Closes the Neo4j driver connection.
        """
        self.driver.close()


def translate_chunks(chunks, client):
    """
    Translates all chunks from Swedish to English using the OpenAI API.
    Returns a list of translated chunk texts (strings).
    """
    translated = []
    for chunk in tqdm(chunks, desc="Translating chunks", unit="chunk"):
        try:
            response = _call_with_retry(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional Swedish-to-English translator. "
                            "Translate the following Swedish text to English. "
                            "Preserve the original structure, formatting, and meaning. "
                            "Keep proper nouns, product names, and technical terms as-is. "
                            "Return ONLY the translated text, nothing else."
                        ),
                    },
                    {"role": "user", "content": chunk.page_content},
                ],
            )
            translated.append(response.choices[0].message.content)
        except Exception as e:
            print(f"Error translating chunk after retries: {e}")
            translated.append(chunk.page_content)  # fallback to original
    return translated


def extract_nodes_from_chunk(chunk_text, prompt, client) -> NodesResponse:
    """
    Sends a chunk to gpt-4o-mini and returns a validated NodesResponse.
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk_text},
        ],
        response_format=NodesResponse,
    )
    return response.choices[0].message.parsed


def extract_edges_from_chunk(chunk_text, prompt, client) -> EdgesResponse:
    """
    Sends a chunk to gpt-4o-mini and returns a validated EdgesResponse.
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk_text},
        ],
        response_format=EdgesResponse,
    )
    return response.choices[0].message.parsed


def extract_graph_from_chunk(chunk_text, nodes_prompt, edges_prompt, client) -> GraphResponse:
    """
    Extracts both nodes and edges in a single gpt-4o call.
    """
    combined_prompt = nodes_prompt + "\n\n" + edges_prompt
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": combined_prompt},
            {"role": "user", "content": chunk_text},
        ],
        response_format=GraphResponse,
    )
    return response.choices[0].message.parsed


def _props_to_dict(props: list[Property]) -> dict:
    """Convert a list of Property objects to a flat dict for Neo4j."""
    return {p.key: p.value for p in props}


def _node_to_dict(node: Node) -> dict:
    return {"node_label": node.node_label, "node_properties": _props_to_dict(node.node_properties)}


def _relationship_to_dict(rel: Relationship) -> dict:
    return {"relationship_type": rel.relationship_type, "relationship_properties": _props_to_dict(rel.relationship_properties)}


def build_graph_from_chunks(chunks, graph, client):
    """
    Iterates over chunks, extracts nodes and edges via the LLM, and builds the graph.
    Nodes are merged by their properties (label + name) so the same entity from
    different chunks becomes a single node in the graph.
    Checkpoints progress after each chunk so a crashed run can resume where it left off.
    """
    processed = _load_checkpoint()
    skipped = sum(1 for chunk in chunks if _chunk_id(chunk.page_content) in processed)
    if skipped:
        print(f"Checkpoint found: skipping {skipped} already-processed chunk(s).")

    for i, chunk in enumerate(tqdm(chunks, desc="Building graph", unit="chunk")):
        chunk_hash = _chunk_id(chunk.page_content)
        if chunk_hash in processed:
            continue

        try:
            graph_response = _call_with_retry(
                extract_graph_from_chunk, chunk.page_content, query_nodes, query_relations, client
            )

            nodes = [_node_to_dict(n) for n in graph_response.nodes]
            edges = [
                (
                    _node_to_dict(e.source),
                    _node_to_dict(e.target),
                    _relationship_to_dict(e.relationship),
                )
                for e in graph_response.edges
            ]

            graph.build_graph(nodes, edges, source_chunk=chunk.page_content)
            processed.add(chunk_hash)
            _save_checkpoint(processed)
        except Exception as e:
            print(f"Error processing chunk {i+1} after retries: {e}")


if __name__ == "__main__":
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from environment
    graph = Graph()

    chunks = text_cutting()
    build_graph_from_chunks(chunks, graph, client)

    graph.close()