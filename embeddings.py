"""Embedding backends with a common interface.

Both models used here expect different prefixes for documents vs queries.
Skipping the prefix is a common silent quality loss, so it's handled here.
"""
import requests

import config


class Embedder:
    def __init__(self, backend: str = config.EMBED_BACKEND, model: str = config.EMBED_MODEL):
        self.backend = backend
        self.model = model
        self.max_tokens = None

        if backend == "st":
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(model)
            # Text beyond this is truncated before embedding.
            self.max_tokens = self._st.max_seq_length
        elif backend != "ollama":
            raise ValueError(f"Unknown EMBED_BACKEND: {backend}")

    # Model-specific prefixes
    def _doc_prefix(self) -> str:
        return "search_document: " if "nomic" in self.model else ""

    def _query_prefix(self) -> str:
        if "nomic" in self.model:
            return "search_query: "
        if "bge" in self.model and "-en" in self.model:
            return "Represent this sentence for searching relevant passages: "
        return ""

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "st":
            return self._st.encode(texts, normalize_embeddings=True).tolist()
        r = requests.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        p = self._doc_prefix()
        return self._embed([p + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._query_prefix() + text])[0]
