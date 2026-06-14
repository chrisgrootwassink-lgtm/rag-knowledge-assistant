from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
PROMPTS_DIR = Path(__file__).parent / "prompts"
NEO4J_CONFIG_FILE = PROJECT_ROOT / "neo4j_config.json"
CHECKPOINT_FILE = PROJECT_ROOT / "checkpoint.json"

VECTORSTORE_COLLECTION = "document_collection"
CHROMA_PATH = str(VECTORSTORE_DIR)
