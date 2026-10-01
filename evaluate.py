"""Evaluate the RAG pipeline against a test set.

Metrics per question:
  evidence_hit   a retrieved chunk contains the expected evidence phrase (the precise retrieval metric)
  retrieval_hit  the expected source document appears in the top-k chunks (coarse)
  correctness    LLM judge: correct / partial / incorrect vs. the expected answer
  grounded       LLM judge: every claim in the answer is supported by the retrieved text
  latency, tokens, (optional) equivalent API cost

Usage:
    python evaluate.py --questions eval/questions.jsonl --chunk-size 500 --overlap 50
"""
import argparse
import json
import re
import statistics
from pathlib import Path

import config
from rag import RAG, ollama_chat, source_label

CORRECTNESS_PROMPT = """You are grading an answer to a question about a set of documents.

Question: {question}
Reference answer: {expected}
Candidate answer: {answer}

Grade the candidate against the reference:
- "correct": contains the key facts of the reference with nothing contradicting it
- "partial": some key facts right, but important ones missing or imprecise
- "incorrect": wrong, contradicts the reference, or refuses when the reference has an answer
If the reference says the information is not in the documents, the candidate is "correct"
only if it also says it cannot find the answer.

Respond with JSON only: {{"verdict": "correct|partial|incorrect", "reason": "<one sentence>"}}"""

GROUNDEDNESS_PROMPT = """Check whether an answer is fully supported by the context passages.

Context:
{context}

Answer: {answer}

Is every factual claim in the answer supported by the context? Ignore citation markers like [1].
Respond with JSON only: {{"grounded": true|false, "unsupported_claim": "<quote or empty>"}}"""


def judge(prompt: str) -> dict:
    out = ollama_chat(config.JUDGE_MODEL, [{"role": "user", "content": prompt}], json_mode=True)
    try:
        return json.loads(out["text"])
    except json.JSONDecodeError:
        return {"parse_error": out["text"]}


def norm(s: str) -> str:
    """Lowercase, unify quotes, drop all whitespace: robust to PDF line breaks and '35 000 000'."""
    s = s.lower().translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "\u00ad": ""}))
    return re.sub(r"\s+", "", s)


def evidence_hit(expected, chunks: list[dict]):
    """True if any retrieved chunk contains any of the expected evidence phrases."""
    if not expected:
        return None
    phrases = [expected] if isinstance(expected, str) else expected
    texts = [norm(c["text"]) for c in chunks]
    return any(norm(p) in t for p in phrases for t in texts)


def is_refusal(answer: str) -> bool:
    return "can't find that in the documents" in answer.lower() or "cannot find" in answer.lower()


def load_questions(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip() and not l.strip().startswith("//")]


def evaluate_one(rag: RAG, q: dict) -> dict:
    r = rag.answer(q["question"])
    answerable = q.get("answerable", True)
    expected = q["expected_answer"] if answerable else "The information is not in the documents."

    # Retrieval: did the right document come back? (only if the question names one)
    exp_src = q.get("expected_source")
    retrieval_hit = None
    if exp_src and answerable:
        retrieval_hit = any(exp_src.lower() in s["source"].lower() for s in r["sources"])
    ev_hit = evidence_hit(q.get("expected_evidence"), r["sources"]) if answerable else None

    c = judge(CORRECTNESS_PROMPT.format(question=q["question"], expected=expected, answer=r["answer"]))
    verdict = c.get("verdict", "incorrect")

    if is_refusal(r["answer"]):
        grounded = True  # a refusal makes no claims
        g = {}
    else:
        g = judge(GROUNDEDNESS_PROMPT.format(context=r["context"], answer=r["answer"]))
        grounded = bool(g.get("grounded", False))

    return {
        "id": q.get("id"),
        "type": q.get("type", "untyped"),
        "question": q["question"],
        "answerable": answerable,
        "expected_answer": expected,
        "answer": r["answer"],
        "retrieved_sources": [source_label(s) for s in r["sources"]],
        "evidence_hit": ev_hit,
        "retrieval_hit": retrieval_hit,
        "verdict": verdict,
        "verdict_reason": c.get("reason", ""),
        "grounded": grounded,
        "unsupported_claim": g.get("unsupported_claim", ""),
        "total_s": r["total_s"],
        "retrieve_s": r["retrieve_s"],
        "prompt_tokens": r["prompt_tokens"],
        "completion_tokens": r["completion_tokens"],
    }


def pct(xs) -> str:
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "n/a"


def summarise(rows: list[dict], label: str) -> dict:
    lat = sorted(r["total_s"] for r in rows)
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    pin = sum(r["prompt_tokens"] for r in rows) / len(rows)
    pout = sum(r["completion_tokens"] for r in rows) / len(rows)
    cost = (pin * config.PRICE_IN_PER_M + pout * config.PRICE_OUT_PER_M) / 1e6
    return {
        "config": label,
        "n": len(rows),
        "evidence_hit@k": pct([r.get("evidence_hit") for r in rows]),
        "doc_hit@k": pct([r["retrieval_hit"] for r in rows]),
        "correct": pct([r["verdict"] == "correct" for r in ans]),
        "correct_or_partial": pct([r["verdict"] in ("correct", "partial") for r in ans]),
        "refused_when_should": pct([r["verdict"] == "correct" for r in unans]),
        "grounded": pct([r["grounded"] for r in rows]),
        "latency_p50_s": round(statistics.median(lat), 2),
        "latency_p95_s": round(p95, 2),
        "avg_prompt_tokens": round(pin),
        "cost_per_query": f"${cost:.5f}" if cost else "$0 (local)",
    }


def by_type(rows: list[dict]) -> dict:
    """{type: 'correct/total'} — shows which kinds of question the system struggles with."""
    out = {}
    for t in sorted({r["type"] for r in rows}):
        rs = [r for r in rows if r["type"] == t]
        out[t] = f"{sum(r['verdict'] == 'correct' for r in rs)}/{len(rs)}"
    return out


def markdown_table(summaries: list[dict]) -> str:
    keys = list(summaries[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(s[k]) for k in keys) + " |" for s in summaries]
    return "\n".join(lines)


def run(questions_path: str, chunk_size: int, overlap: int, top_k: int, tag: str = "",
        retrieval: str = config.RETRIEVAL) -> tuple[dict, list[dict]]:
    rag = RAG(chunk_size, overlap, top_k, tag=tag, retrieval=retrieval)
    label = f"{tag + ' ' if tag else ''}cs={chunk_size} ov={overlap} k={top_k} {retrieval}"
    qs = load_questions(questions_path)
    rows = []
    for i, q in enumerate(qs, 1):
        row = evaluate_one(rag, q)
        rows.append(row)
        mark = {"correct": "✓", "partial": "~", "incorrect": "✗"}.get(row["verdict"], "?")
        print(f"[{i}/{len(qs)}] {mark} {q['question'][:70]}")

    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    out = Path(config.RESULTS_DIR) / f"{rag.collection_name}_k{top_k}_{retrieval}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    summary = summarise(rows, label)
    print("\n" + markdown_table([summary]))
    print("\nCorrect by question type:")
    for t, score in by_type(rows).items():
        print(f"  {t:<18} {score}")
    print(f"\nPer-question results: {out}")
    return summary, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--tag", default="", help='e.g. "struct" for the article-split index')
    ap.add_argument("--retrieval", default=config.RETRIEVAL, choices=config.RETRIEVAL_METHODS)
    a = ap.parse_args()
    run(a.questions, a.chunk_size, a.overlap, a.top_k, a.tag, a.retrieval)
