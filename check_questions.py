"""Check a test set against the index before running the (slow) evaluation.

For every answerable question, confirms its expected_evidence phrase really exists in some chunk
(matching ignores case, spaces and line breaks) and shows where. For phrases that are not found,
shows the closest chunk by keyword search, so you can copy the real wording from it.

Usage:
    python check_questions.py
    python check_questions.py --questions eval/questions.jsonl --index docs_struct_cs1000_ov100
"""
import argparse
from collections import Counter

import config
from evaluate import load_questions, norm
from rag import RAG, source_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--index", default=config.collection_name(1000, 100, "struct"))
    a = ap.parse_args()

    rag = RAG(collection=a.index, retrieval="keyword")
    chunks = list(rag.chunks.values())
    normed = [norm(c["text"]) for c in chunks]
    questions = load_questions(a.questions)

    problems = 0
    ids = Counter(q.get("id") for q in questions)
    for q in questions:
        qid = q.get("id", "?")
        if ids[qid] > 1:
            print(f"✗ {qid}: duplicate id")
            problems += 1
        if not q.get("answerable", True):
            print(f"· {qid}: unanswerable (no evidence needed)")
            continue
        phrases = q.get("expected_evidence")
        if not phrases:
            print(f"? {qid}: no expected_evidence, so retrieval can't be scored for this question")
            continue
        phrases = [phrases] if isinstance(phrases, str) else phrases
        found = [c for c, t in zip(chunks, normed) if any(norm(p) in t for p in phrases)]
        if found:
            where = "; ".join(sorted({source_label(c) for c in found})[:2])
            more = f" (+{len(found) - 2} more)" if len(found) > 2 else ""
            print(f"✓ {qid}: found in {where}{more}")
        else:
            problems += 1
            best = rag.keyword_search(phrases[0], 1)
            print(f"✗ {qid}: evidence NOT FOUND: {phrases[0]!r}")
            if best:
                snippet = " ".join(best[0]["text"].split())[:300]
                print(f"      closest chunk: {source_label(best[0])}")
                print(f"      {snippet}…")

    types = Counter(q.get("type", "untyped") for q in questions)
    print(f"\n{len(questions)} questions: " + ", ".join(f"{n} {t}" for t, n in types.most_common()))
    print("All evidence phrases found ✓" if not problems else f"{problems} problem(s) to fix")


if __name__ == "__main__":
    main()
