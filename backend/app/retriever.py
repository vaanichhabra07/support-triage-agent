
"""
RAG Retriever using ChromaDB + Gemini embeddings.

Supports:
- Dynamic company document uploads
- Company-filtered semantic search
- Retrieval confidence scoring
- Persistent ChromaDB storage
- Bundled knowledge-base seeding

Uses Google's embedding API instead of local Sentence Transformers,
reducing memory requirements for Render deployment.
"""

import os
import re
import glob

import chromadb
from google import genai
from dotenv import load_dotenv


# ---------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(
    os.path.join(APP_DIR, "..", ".env")
)

DATA_DIR = os.path.abspath(
    os.environ.get(
        "CHROMA_DATA_DIR",
        os.path.join(APP_DIR, "storage")
    )
)

CHROMA_DIR = os.path.join(DATA_DIR, ".chroma")

SEED_DATA_DIR = os.path.join(APP_DIR, "seed_data")

COLLECTION_NAME = "support_corpus_gemini_v1"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001"
)

os.makedirs(CHROMA_DIR, exist_ok=True)


# ---------------------------------------------------------
# Company name normalization
# ---------------------------------------------------------

def slugify(name: str) -> str:
    """Convert a company name into a safe corpus identifier."""
    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        name.strip().lower()
    ).strip("-")

    return slug or "general"


# ---------------------------------------------------------
# Document chunking
# ---------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:

    """Split text into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------
# Gemini embedding function
# ---------------------------------------------------------

class GeminiEmbeddingFunction:
    """
    Custom embedding function compatible with ChromaDB 1.5.9.

    Generates embeddings remotely using Google's Gemini API.
    """

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. "
                "Configure it in backend/.env or Render."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = EMBEDDING_MODEL

    def __call__(self, input):
        """Generate embeddings for document chunks."""

        if not input:
            return []

        result = self.client.models.embed_content(
            model=self.model,
            contents=list(input)
        )

        if not result.embeddings:
            raise RuntimeError(
                "Gemini returned no embeddings."
            )

        return [
            embedding.values
            for embedding in result.embeddings
        ]

    def embed_query(self, input):
        """
        Generate embeddings for search queries.

        Required by ChromaDB 1.5.9.
        """

        return self.__call__(input)

    def name(self) -> str:
        """Return a stable embedding function identifier."""

        return "gemini_embedding"


# ---------------------------------------------------------
# Embedding function singleton
# ---------------------------------------------------------

_ef = None


def get_embedding_function():
    global _ef

    if _ef is None:
        _ef = GeminiEmbeddingFunction()

    return _ef


# ---------------------------------------------------------
# ChromaDB client and collection
# ---------------------------------------------------------

def get_client_and_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}
    )

    return client, collection


# ---------------------------------------------------------
# Add company documents
# ---------------------------------------------------------

def add_documents(
    company: str,
    docs: list[tuple[str, str]]
) -> int:

    """
    Add documents to a company-specific corpus.

    docs:
        [
            ("document.md", "document content"),
            ...
        ]

    Returns the number of chunks indexed.
    """

    source = slugify(company)

    _, collection = get_client_and_collection()

    all_chunks = []

    for filename, content in docs:

        if not content or not content.strip():
            continue

        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):

            all_chunks.append({
                "id": f"{source}::{filename}::chunk_{i}",
                "text": chunk,
                "source": source,
                "filepath": filename
            })

    if not all_chunks:
        return 0

    # Small batches reduce request size and peak memory usage.
    BATCH_SIZE = 20

    for i in range(0, len(all_chunks), BATCH_SIZE):

        batch = all_chunks[i:i + BATCH_SIZE]

        collection.upsert(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[
                {
                    "source": d["source"],
                    "filepath": d["filepath"]
                }
                for d in batch
            ]
        )

    return len(all_chunks)


# ---------------------------------------------------------
# Delete company documents
# ---------------------------------------------------------

def delete_company(company: str) -> int:

    """Delete all indexed documents for a company."""

    source = slugify(company)

    _, collection = get_client_and_collection()

    existing = collection.get(
        where={"source": source}
    )

    ids = existing.get("ids", [])

    if ids:
        collection.delete(ids=ids)

    return len(ids)


# ---------------------------------------------------------
# List indexed companies
# ---------------------------------------------------------

def list_companies() -> list[dict]:

    """
    Return:
        [
            {"name": "claude", "chunk_count": 8},
            ...
        ]
    """

    _, collection = get_client_and_collection()

    if collection.count() == 0:
        return []

    all_meta = collection.get(
        include=["metadatas"]
    )

    counts = {}

    for meta in all_meta.get("metadatas", []):

        if not meta:
            continue

        source = meta.get(
            "source",
            "unknown"
        )

        counts[source] = counts.get(source, 0) + 1

    return [
        {
            "name": company,
            "chunk_count": count
        }
        for company, count in sorted(counts.items())
    ]


# ---------------------------------------------------------
# Seed knowledge base from disk
# ---------------------------------------------------------

def seed_from_disk(force_rebuild: bool = False):

    """
    Load bundled Markdown knowledge documents on startup.

    Documents are loaded from:
        backend/app/seed_data/<company>/*.md

    Existing collections are preserved unless force_rebuild=True.
    """

    client, collection = get_client_and_collection()

    if collection.count() > 0 and not force_rebuild:
        return collection

    if force_rebuild:

        client.delete_collection(
            COLLECTION_NAME
        )

        _, collection = get_client_and_collection()

    if not os.path.isdir(SEED_DATA_DIR):
        return collection

    for company in sorted(os.listdir(SEED_DATA_DIR)):

        folder = os.path.join(
            SEED_DATA_DIR,
            company
        )

        if not os.path.isdir(folder):
            continue

        docs = []

        pattern = os.path.join(
            folder,
            "**",
            "*.md"
        )

        for filepath in glob.glob(
            pattern,
            recursive=True
        ):

            rel = os.path.relpath(
                filepath,
                folder
            )

            try:

                with open(
                    filepath,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as file:

                    docs.append(
                        (rel, file.read())
                    )

            except OSError:
                continue

        if docs:
            add_documents(company, docs)

    _, collection = get_client_and_collection()

    return collection


# ---------------------------------------------------------
# Semantic retrieval with confidence
# ---------------------------------------------------------

def query_with_scores(
    query_text: str,
    company: str | None = None,
    n_results: int = 5
) -> dict:

    """
    Retrieve relevant documents.

    Returns:
        {
            "context": str,
            "confidence": float,
            "matches": [
                {
                    "source": str,
                    "filepath": str,
                    "distance": float
                }
            ]
        }
    """

    _, collection = get_client_and_collection()

    if collection.count() == 0:

        return {
            "context": "[No documents indexed yet]",
            "confidence": 0.0,
            "matches": []
        }

    if not query_text or not query_text.strip():

        return {
            "context": "[No relevant documents found]",
            "confidence": 0.0,
            "matches": []
        }

    if n_results <= 0:
        raise ValueError("n_results must be positive")

    where_filter = None

    if company and company.lower().strip() not in (
        "none",
        "unknown",
        ""
    ):

        where_filter = {
            "source": slugify(company)
        }

    # Never remove a requested company filter.
    # Returning another company's documents would be unsafe.
    results = collection.query(
        query_texts=[query_text],
        n_results=min(
            n_results,
            collection.count()
        ),
        where=where_filter,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    if not docs:

        return {
            "context": "[No relevant documents found]",
            "confidence": 0.0,
            "matches": []
        }

    output_parts = []
    matches = []

    for index, doc in enumerate(docs):

        meta = metas[index] if index < len(metas) else {}
        meta = meta or {}

        dist = (
            distances[index]
            if index < len(distances)
            else None
        )

        source = meta.get(
            "source",
            "unknown"
        )

        filepath = meta.get(
            "filepath",
            ""
        )

        output_parts.append(
            f"--- [{source.upper()}] "
            f"(source: {filepath}) ---\n{doc}"
        )

        matches.append({
            "source": source,
            "filepath": filepath,
            "distance": dist
        })

    # Chroma cosine distance:
    # 0 = identical, 2 = opposite.
    # This is a similarity-derived heuristic,
    # not a calibrated probability of correctness.
    valid_distances = [
        d for d in distances
        if d is not None
    ]

    if valid_distances:

        best_distance = min(valid_distances)

        confidence = max(
            0.0,
            min(
                1.0,
                1.0 - best_distance / 2.0
            )
        )

    else:
        confidence = 0.5

    return {
        "context": "\n\n".join(output_parts),
        "confidence": round(confidence, 3),
        "matches": matches
    }


# ---------------------------------------------------------
# Backwards-compatible query
# ---------------------------------------------------------

def query(
    query_text: str,
    company: str | None = None,
    n_results: int = 5
) -> str:

    """Return retrieved context as plain text."""

    result = query_with_scores(
        query_text,
        company,
        n_results
    )

    return result["context"]
