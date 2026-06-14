"""Vector store ingestion pipeline: load, chunk, embed, persist to ChromaDB."""

import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from rag_assistant.config import (
    CHROMA_PATH,
    DATA_DIR,
    VECTORSTORE_COLLECTION,
)

load_dotenv()

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


class VectorStoreBuilder:
    """Builds and persists a ChromaDB vector store from .docx documents.

    Loads documents from *data_dir*, splits them into speaker-aware
    chunks, embeds them in batches using OpenAI embeddings, and writes
    the result to a ChromaDB persistent directory.

    Attributes:
        _data_dir: Source directory containing .docx files.
        _chroma_path: Destination directory for the ChromaDB files.
        _collection_name: Name of the ChromaDB collection to create.
        _batch_size: Number of chunks to embed per API call.
        _max_retries: How many times to retry a failed embedding batch.
    """

    def __init__(
        self,
        data_dir=DATA_DIR,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = VECTORSTORE_COLLECTION,
        batch_size: int = 20,
        max_retries: int = 5,
    ) -> None:
        self._data_dir = data_dir
        self._chroma_path = chroma_path
        self._collection_name = collection_name
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            is_separator_regex=True,
            separators=_SPEAKER_SEPARATORS,
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _load_documents(self) -> list:
        print("[1/3] Loading documents...")
        loader = DirectoryLoader(
            str(self._data_dir),
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=True,
        )
        documents = loader.load()
        print(f"      Loaded {len(documents)} document(s).\n")
        return documents

    def _split_documents(self, documents: list) -> list:
        print("[2/3] Splitting into chunks...")
        chunks = self._splitter.split_documents(documents)
        print(f"      Created {len(chunks)} chunk(s).\n")
        return chunks

    def _embed_with_retry(self, chunks: list) -> Chroma | None:
        """Embed *chunks* in batches, retrying on transient API errors."""
        print(
            f"[3/3] Embedding {len(chunks)} chunks "
            f"into '{self._collection_name}'..."
        )
        vectorstore = self._create_initial_batch(chunks)
        if vectorstore is None:
            return None

        remaining_range = range(
            self._batch_size, len(chunks), self._batch_size
        )
        for batch_start in tqdm(remaining_range, desc="      Batches"):
            batch = chunks[batch_start: batch_start + self._batch_size]
            self._add_batch_with_retry(vectorstore, batch, batch_start)

        print(
            f"\nDone. Stored in '{self._chroma_path}' "
            f"under '{self._collection_name}'."
        )
        return vectorstore

    def _create_initial_batch(self, chunks: list) -> Chroma | None:
        """Bootstrap the collection with the first batch of chunks."""
        for attempt in range(self._max_retries):
            try:
                return Chroma.from_documents(
                    documents=chunks[: self._batch_size],
                    embedding=self._embeddings,
                    persist_directory=self._chroma_path,
                    collection_name=self._collection_name,
                )
            except Exception as exc:
                if attempt == self._max_retries - 1:
                    print(f"Error on first batch: {exc}")
                    return None
                time.sleep(5)
        return None

    def _add_batch_with_retry(
        self, vectorstore: Chroma, batch: list, batch_start: int
    ) -> None:
        """Add a single batch to an existing vectorstore, retrying on failure."""
        for attempt in range(self._max_retries):
            try:
                vectorstore.add_documents(batch)
                return
            except Exception as exc:
                if attempt == self._max_retries - 1:
                    print(f"Error at batch {batch_start}: {exc}")
                    return
                time.sleep(5)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self) -> Chroma | None:
        """Run the full ingestion pipeline and return the vector store.

        Returns:
            A Chroma instance if ingestion succeeded, otherwise None.
        """
        documents = self._load_documents()
        chunks = self._split_documents(documents)

        if not chunks:
            print("No chunks to embed. Check the data/ folder.")
            return None

        return self._embed_with_retry(chunks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: build the vector store from documents in data/."""
    builder = VectorStoreBuilder()
    builder.build()
