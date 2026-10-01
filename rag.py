"""Retrieve relevant chunks and generate a cited answer with a local Ollama model."""
import time

import requests

import config
from embeddings import Embedder
from ingest import get_client
from keyword_search import BM25
from reranker import rerank

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.

How to answer:
1. Start with a direct answer to the question in one or two sentences.
2. Then add only the conditions or exceptions that matter, briefly.
3. Put the passage number right after each fact it supports, like [1] or [2][3].
4. Keep figures, dates, percentages and limits exactly as written, but otherwise use your own words.
   Never copy whole passages, and never repeat the "(source: ...)" labels.
5. Use only the passages that actually answer the question; ignore the rest.
6. Do not use outside knowledge.
7. If passages give different dates, figures or rules for the same thing, prefer the passage
   marked "(Amended text)" or coming from an amending regulation, and say that it was amended.

If the passages do not contain the answer, reply exactly: "I can't find that in the documents."
Keep the answer under 120 words unless the question asks for a list."""

REFUSAL = "I can't find that in the documents."


def ollama_chat(model: str, messages: list[dict], json_mode: bool = False) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": config.NUM_CTX},
    }
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return {
        "text": data["message"]["content"].strip(),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
    }


def source_label(c: dict) -> str:
    """'Regulation (EU) 2024/1689 — Article 6: …' for structured files, 'file.pdf, p.12' otherwise."""
    if c.get("title"):
        return c["title"]
    pages = f"p.{c['page_start']}" + (f"-{c['page_end']}" if c["page_end"] != c["page_start"] else "")
    return f"{c['source']}, {pages}"


def rank_info(c: dict) -> str:
    """e.g. 'dense #3 · keyword #1' — shows which search method found a chunk."""
    parts = []
    if "rerank_score" in c:
        parts.append(f"rerank {c['rerank_score']}")
    if "dense_rank" in c or "keyword_rank" in c:
        parts.append(f"dense #{c['dense_rank']}" if "dense_rank" in c else "dense: –")
        parts.append(f"keyword #{c['keyword_rank']}" if "keyword_rank" in c else "keyword: –")
    elif c.get("distance") is not None:
        parts.append(f"distance {c['distance']}")
    elif "bm25" in c:
        parts.append(f"bm25 {c['bm25']}")
    return " · ".join(parts)


class RAG:
    def __init__(self, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP, top_k=config.TOP_K,
                 tag: str = "", collection: str | None = None, retrieval: str = config.RETRIEVAL):
        if retrieval not in config.RETRIEVAL_METHODS:
            raise ValueError(f"retrieval must be one of {config.RETRIEVAL_METHODS}, not {retrieval!r}")
        self.top_k = top_k
        self.retrieval = retrieval
        self.collection_name = collection or config.collection_name(chunk_size, overlap, tag)
        self.collection = get_client().get_collection(self.collection_name)
        self.embedder = Embedder()

        # Keyword index over every chunk, built in memory (fast at this corpus size).
        everything = self.collection.get(include=["documents", "metadatas"])
        self.ids = everything["ids"]
        self.chunks = {
            i: {"id": i, "text": d, **m}
            for i, d, m in zip(everything["ids"], everything["documents"], everything["metadatas"])
        }
        self.bm25 = BM25(everything["documents"])

    @staticmethod
    def available_collections() -> list[str]:
        cols = get_client().list_collections()
        return sorted(c if isinstance(c, str) else c.name for c in cols)

    def dense_search(self, question: str, n: int) -> list[dict]:
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(question)],
            n_results=min(n, len(self.ids)),
        )
        return [
            {**self.chunks[i], "distance": round(dist, 4)}
            for i, dist in zip(res["ids"][0], res["distances"][0])
        ]

    def keyword_search(self, question: str, n: int) -> list[dict]:
        scores = self.bm25.scores(question)
        order = sorted(range(len(self.ids)), key=lambda j: -scores[j])[:n]
        return [{**self.chunks[self.ids[j]], "bm25": round(scores[j], 3)} for j in order if scores[j] > 0]

    def candidates(self, question: str, pool: int) -> list[dict]:
        """Union of the dense and keyword top-`pool`, ordered by Reciprocal Rank Fusion.

        Each list votes 1/(60 + rank) for its results; chunks ranked well by both rise to the
        top. Rank-based, so the two very different scoring scales never need calibrating."""
        dense = self.dense_search(question, pool)
        keyword = self.keyword_search(question, pool)
        fused, info = {}, {}
        for method, results in (("dense", dense), ("keyword", keyword)):
            for rank, c in enumerate(results, 1):
                fused[c["id"]] = fused.get(c["id"], 0.0) + 1.0 / (config.RRF_K + rank)
                info.setdefault(c["id"], dict(self.chunks[c["id"]]))[f"{method}_rank"] = rank
                if "distance" in c:
                    info[c["id"]]["distance"] = c["distance"]
        order = sorted(fused, key=lambda i: -fused[i])
        return [{**info[i], "rrf": round(fused[i], 4)} for i in order]

    def rerank_search(self, question: str) -> list[dict]:
        """All candidates from both searches, reordered by the cross-encoder."""
        return rerank(question, self.candidates(question, config.RERANK_POOL))

    def retrieve(self, question: str) -> list[dict]:
        if self.retrieval == "dense":
            return self.dense_search(question, self.top_k)
        if self.retrieval == "keyword":
            return self.keyword_search(question, self.top_k)
        if self.retrieval == "hybrid":
            return self.candidates(question, config.CANDIDATE_POOL)[: self.top_k]
        return self.rerank_search(question)[: self.top_k]

    @staticmethod
    def build_context(chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{i}] (source: {source_label(c)})\n{c['text']}")
        return "\n\n".join(parts)

    def answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        chunks = self.retrieve(question)
        t1 = time.perf_counter()

        context = self.build_context(chunks)
        out = ollama_chat(
            config.LLM_MODEL,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        t2 = time.perf_counter()

        return {
            "question": question,
            "answer": out["text"],
            "sources": chunks,
            "context": context,
            "retrieve_s": round(t1 - t0, 3),
            "generate_s": round(t2 - t1, 3),
            "total_s": round(t2 - t0, 3),
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
        }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or input("Question: ")
    r = RAG().answer(q)
    print("\n" + r["answer"] + "\n")
    for i, s in enumerate(r["sources"], 1):
        print(f"[{i}] {source_label(s)}  {rank_info(s)}")
    print(f"\n{r['total_s']}s, {r['prompt_tokens']} prompt tokens")
