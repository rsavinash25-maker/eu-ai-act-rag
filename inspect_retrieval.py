"""Compare what each search method returns for one question.

Usage:
    python inspect_retrieval.py "What is the maximum fine for a prohibited AI practice?"
    python inspect_retrieval.py "..." --find "Article 99"      # where does a given chunk rank?
    python inspect_retrieval.py "..." --index docs_struct_cs300_ov30
    python inspect_retrieval.py "..." --no-rerank               # skip the (slower) reranker
"""
import argparse
import time

import config
from rag import RAG, source_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--index", default=config.collection_name(1000, 100, "struct"))
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--find", default=None, help='text in the chunk title, e.g. "Article 99"')
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()

    rag = RAG(collection=a.index, top_k=a.top, retrieval="hybrid")
    n = len(rag.ids)
    results = {
        "dense": rag.dense_search(a.question, n),
        "keyword": rag.keyword_search(a.question, n),
        "hybrid": rag.candidates(a.question, config.CANDIDATE_POOL),
    }
    if not a.no_rerank:
        t0 = time.perf_counter()
        results["rerank"] = rag.rerank_search(a.question)
        print(f"(reranked {len(results['rerank'])} candidates in {time.perf_counter() - t0:.1f}s)")

    for method, hits in results.items():
        print(f"\n=== {method.upper()} (top {a.top}) ===")
        for i, c in enumerate(hits[: a.top], 1):
            score = f"  [{c['rerank_score']}]" if "rerank_score" in c else ""
            print(f"{i:>3}. {source_label(c)}{score}")

    if a.find:
        print(f"\n=== Where does '{a.find}' rank? ===")
        for method, hits in results.items():
            ranks = [i for i, c in enumerate(hits, 1) if a.find.lower() in source_label(c).lower()]
            where = ", ".join(f"#{r}" for r in ranks) if ranks else "not found"
            print(f"{method:>8}: {where}   (out of {len(hits)})")


if __name__ == "__main__":
    main()
