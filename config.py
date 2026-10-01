"""Central configuration. Every value can be overridden with an environment variable."""
import os

DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Embeddings: "ollama" (default) or "st" (sentence-transformers)
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ollama")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "nomic-embed-text" if EMBED_BACKEND == "ollama" else "BAAI/bge-small-en-v1.5",
)

# Generation and evaluation models
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)
# Ollama's default context window is small; retrieved chunks get silently cut off without this.
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))

# Chunking and retrieval defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))
# rerank  = dense + keyword candidates, reordered by a cross-encoder (most accurate, slowest)
# hybrid  = dense + keyword rankings merged with Reciprocal Rank Fusion
# dense   = embeddings only;  keyword = BM25 only
RETRIEVAL = os.getenv("RETRIEVAL", "rerank")
RETRIEVAL_METHODS = ["rerank", "hybrid", "dense", "keyword"]
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANK_POOL = int(os.getenv("RERANK_POOL", "20"))  # candidates taken from EACH search for reranking
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "50"))  # how deep each ranking goes before fusion
RRF_K = 60  # standard Reciprocal Rank Fusion constant

# Optional: price per 1M tokens, to report "what this would cost on a paid API".
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M", "0"))
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M", "0"))


def collection_name(chunk_size: int, overlap: int, tag: str = "") -> str:
    """One Chroma collection per chunking config, so experiments don't overwrite each other.
    tag distinguishes corpora, e.g. "struct" for the article-split documents."""
    prefix = f"docs_{tag}_" if tag else "docs_"
    return f"{prefix}cs{chunk_size}_ov{overlap}"
