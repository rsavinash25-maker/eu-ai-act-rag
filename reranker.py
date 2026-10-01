"""Cross-encoder reranking.

Dense and keyword search each score a chunk without looking at the question and the chunk
together: embeddings compare two separately computed vectors, BM25 counts shared words.
A cross-encoder reads the question and one chunk as a single input and predicts how well the
chunk answers the question. It is far more accurate but too slow to run over every chunk, so it
only reorders a shortlist of candidates that the two cheap searches collected.
"""
from functools import lru_cache

import config


@lru_cache(maxsize=2)
def get_reranker(model_name: str = config.RERANKER_MODEL):
    """Loaded once per process (first call downloads the model, ~1 GB)."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise SystemExit(
            "Reranking needs sentence-transformers: run  pip install -r requirements.txt"
        ) from e
    return CrossEncoder(model_name, max_length=512)


def rerank(question: str, chunks: list[dict], model_name: str = config.RERANKER_MODEL) -> list[dict]:
    """Return all chunks sorted by cross-encoder relevance, highest first."""
    if not chunks:
        return []
    model = get_reranker(model_name)
    scores = model.predict([(question, c["text"]) for c in chunks], batch_size=16)
    ranked = sorted(zip(chunks, scores), key=lambda pair: -float(pair[1]))
    return [{**c, "rerank_score": round(float(s), 3)} for c, s in ranked]
