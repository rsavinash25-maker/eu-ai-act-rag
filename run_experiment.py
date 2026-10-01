"""Run several configurations over the test set and compare them.

A configuration is  ARM:METHOD
  ARM     struct   -> article/definition-split corpus (data/docs_structured), chunks capped at 1000 tokens
          fixedN   -> plain token chunks of N tokens from --fixed-docs (e.g. fixed500)
  METHOD  rerank | hybrid | dense | keyword

Usage:
    python run_experiment.py                                   # the default comparison below
    python run_experiment.py --configs struct:rerank struct:dense
    caffeinate -i python run_experiment.py                     # macOS: stop the Mac sleeping mid-run

Writes results/experiment.md: overall table, correct-by-type table, and which questions flipped.
"""
import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import config
from evaluate import by_type, markdown_table, run
from ingest import build_index

DEFAULT_CONFIGS = ["struct:rerank", "struct:hybrid", "struct:dense", "fixed500:rerank"]


def parse_arm(arm: str, a) -> dict:
    if arm == "struct":
        cap = a.structured_cap
        return dict(chunk_size=cap, overlap=cap // 10, docs_dir=a.structured_dir, tag="struct")
    m = re.fullmatch(r"fixed(\d+)", arm)
    if not m:
        raise SystemExit(f"Unknown arm {arm!r}: use struct or fixedN (e.g. fixed500)")
    size = int(m.group(1))
    return dict(chunk_size=size, overlap=size // 10, docs_dir=a.fixed_docs, tag="fixed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--structured-dir", default="data/docs_structured")
    ap.add_argument("--structured-cap", type=int, default=1000)
    ap.add_argument("--fixed-docs", default="data/docs_fixed",
                    help="plain PDFs for fixed-size arms: one version of each document (no duplicates)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild indexes even if they exist")
    a = ap.parse_args()

    t0 = time.perf_counter()
    summaries, all_rows, index_stats, built = [], {}, [], set()
    for cfg in a.configs:
        arm_name, _, method = cfg.partition(":")
        method = method or config.RETRIEVAL
        arm = parse_arm(arm_name, a)
        if not Path(arm["docs_dir"]).is_dir():
            raise SystemExit(f"{cfg}: folder {arm['docs_dir']} not found")
        print(f"\n=== {cfg} ===")
        if arm_name not in built:
            index_stats.append({"arm": arm_name, **build_index(arm["chunk_size"], arm["overlap"], rebuild=a.rebuild,
                                                               docs_dir=arm["docs_dir"], tag=arm["tag"])})
            built.add(arm_name)
        summary, rows = run(a.questions, arm["chunk_size"], arm["overlap"], a.top_k, arm["tag"], method)
        summary["config"] = cfg
        summaries.append(summary)
        all_rows[cfg] = rows

    # Correct-by-type table
    types = sorted({r["type"] for rows in all_rows.values() for r in rows})
    type_rows = [{"config": cfg, **by_type(rows)} for cfg, rows in all_rows.items()]
    for tr in type_rows:
        for t in types:
            tr.setdefault(t, "–")

    # Questions where configurations disagree
    marks = {"correct": "✓", "partial": "~", "incorrect": "✗"}
    first = next(iter(all_rows.values()))
    flips = []
    for i, r in enumerate(first):
        verdicts = {cfg: rows[i]["verdict"] for cfg, rows in all_rows.items()}
        if len(set(verdicts.values())) > 1:
            flips.append({"id": r["id"], "type": r["type"], "question": r["question"][:70],
                          **{cfg: marks.get(v, "?") for cfg, v in verdicts.items()}})

    minutes = (time.perf_counter() - t0) / 60
    report = [
        f"# Evaluation ({datetime.now():%Y-%m-%d %H:%M}, {minutes:.0f} min)",
        "",
        f"LLM: `{config.LLM_MODEL}` · Judge: `{config.JUDGE_MODEL}` · Embeddings: `{config.EMBED_MODEL}` · "
        f"Reranker: `{config.RERANKER_MODEL}` · top-k: {a.top_k} · questions: `{a.questions}`",
        "",
        "## Overall",
        "",
        markdown_table(summaries),
        "",
        "## Correct answers by question type",
        "",
        markdown_table(type_rows),
        "",
        "## Questions where configurations disagree (✓ correct · ~ partial · ✗ incorrect)",
        "",
        markdown_table(flips) if flips else "_All configurations gave the same verdict on every question._",
        "",
        "## Indexes",
        "",
        markdown_table(index_stats),
        "",
        "## Notes",
        "",
        "_Write what you observed: why the flipped questions flipped (check results/*.jsonl)._",
    ]
    out = Path(config.RESULTS_DIR) / "experiment.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n" + markdown_table(summaries) + f"\n\nFull report: {out}  ({minutes:.0f} min)")
