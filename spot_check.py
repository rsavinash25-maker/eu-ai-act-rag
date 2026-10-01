"""Spot-check the LLM judge: read answers yourself and record whether you agree with its verdicts.

By default shows every answer the judge did NOT mark fully correct (where a judging mistake
would hide a real failure, or invent one) plus a few it marked correct (to check it isn't too
lenient). Saves results/spot_check.md with the agreement rate for the README.

Usage:
    python spot_check.py
    python spot_check.py --file results/docs_struct_cs1000_ov100_k4_dense.jsonl --n-correct 3
"""
import argparse
import json
import random
import textwrap
from pathlib import Path

import config


def wrap(label: str, text: str) -> str:
    body = textwrap.fill(" ".join(str(text).split()) or "(empty)", width=100,
                         initial_indent="    ", subsequent_indent="    ")
    return f"{label}\n{body}"


def ask(prompt: str, options: str) -> str:
    while True:
        a = input(prompt).strip().lower()
        if a in options:
            return a
        print(f"  Please type one of: {', '.join(options)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=f"{config.RESULTS_DIR}/docs_struct_cs1000_ov100_k4_rerank.jsonl")
    ap.add_argument("--n-correct", type=int, default=2, help="how many 'correct' verdicts to sample")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    not_correct = [r for r in rows if r["verdict"] != "correct"]
    correct = [r for r in rows if r["verdict"] == "correct"]
    random.Random(a.seed).shuffle(correct)
    sample = not_correct + correct[: a.n_correct]
    print(f"\nChecking {len(sample)} of {len(rows)} verdicts from {a.file}")
    print("For each one: do YOU agree with the grader's verdict?  y = agree, n = disagree, s = skip\n")

    records = []
    for i, r in enumerate(sample, 1):
        print("=" * 104)
        print(f"[{i}/{len(sample)}]  {r['id']} · {r['type']}")
        print(wrap("QUESTION:", r["question"]))
        print(wrap("EXPECTED:", r["expected_answer"] or "(should refuse: not in the documents)"))
        print(wrap("SYSTEM ANSWERED:", r["answer"]))
        print(wrap("SOURCES:", "; ".join(r.get("retrieved_sources", []))))
        print(f"\nGRADER SAID: {r['verdict'].upper()}  —  {r.get('verdict_reason', '')}")
        if not r.get("grounded", True):
            print(f"  (also flagged as not grounded: {r.get('unsupported_claim', '')})")
        agree = ask("\nDo you agree? (y/n/s): ", "yns")
        your_verdict, note = r["verdict"], ""
        if agree == "n":
            your_verdict = {"c": "correct", "p": "partial", "i": "incorrect"}[
                ask("  Your verdict? (c = correct, p = partial, i = incorrect): ", "cpi")]
            note = input("  Why, in a few words? (Enter to skip): ").strip()
        records.append({**r, "agree": agree, "your_verdict": your_verdict, "note": note})
        print()

    judged = [x for x in records if x["agree"] != "s"]
    agreed = sum(x["agree"] == "y" for x in judged)
    lines = [
        "# Judge spot-check",
        "",
        f"Source: `{a.file}`. Checked {len(judged)} verdicts by hand "
        f"(all {len(not_correct)} not marked correct, plus {min(a.n_correct, len(correct))} marked correct).",
        "",
        f"**Agreed with the judge on {agreed} of {len(judged)}"
        + (f" ({100 * agreed / len(judged):.0f}%)**" if judged else "**"),
        "",
        "| id | type | judge | me | note |",
        "|---|---|---|---|---|",
    ]
    for x in records:
        if x["agree"] == "s":
            continue
        lines.append(f"| {x['id']} | {x['type']} | {x['verdict']} | {x['your_verdict']} | {x['note']} |")
    out = Path(config.RESULTS_DIR) / "spot_check.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("=" * 104)
    print(f"You agreed with the judge on {agreed} of {len(judged)}. Saved to {out}")


if __name__ == "__main__":
    main()
