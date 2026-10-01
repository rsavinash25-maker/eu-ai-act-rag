# Evaluation (2026-10-01 02:19, 117 min)

LLM: `llama3.1:8b` · Judge: `llama3.1:8b` · Embeddings: `nomic-embed-text` · Reranker: `BAAI/bge-reranker-base` · top-k: 4 · questions: `eval/questions.jsonl`

## Overall

| config | n | evidence_hit@k | doc_hit@k | correct | correct_or_partial | refused_when_should | grounded | latency_p50_s | latency_p95_s | avg_prompt_tokens | cost_per_query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| struct:rerank | 30 | 69% | n/a | 73% | 96% | 75% | 97% | 39.4 | 58.96 | 2513 | $0 (local) |
| struct:hybrid | 30 | 58% | n/a | 50% | 96% | 100% | 100% | 29.76 | 45.0 | 2472 | $0 (local) |
| struct:dense | 30 | 54% | n/a | 73% | 88% | 100% | 97% | 23.75 | 39.96 | 2231 | $0 (local) |
| fixed500:rerank | 30 | 73% | n/a | 50% | 81% | 75% | 97% | 33.07 | 39.02 | 2137 | $0 (local) |

## Correct answers by question type

| config | cross-reference | definition | lookup | obligation | prohibition | scope | timeline | timeline-amended | transparency | unanswerable |
|---|---|---|---|---|---|---|---|---|---|---|
| struct:rerank | 4/5 | 2/3 | 3/4 | 4/5 | 2/2 | 1/2 | 1/1 | 0/2 | 2/2 | 3/4 |
| struct:hybrid | 4/5 | 1/3 | 1/4 | 3/5 | 2/2 | 1/2 | 0/1 | 0/2 | 1/2 | 4/4 |
| struct:dense | 4/5 | 2/3 | 1/4 | 4/5 | 2/2 | 2/2 | 1/1 | 1/2 | 2/2 | 4/4 |
| fixed500:rerank | 2/5 | 0/3 | 1/4 | 5/5 | 2/2 | 2/2 | 0/1 | 0/2 | 1/2 | 3/4 |

## Questions where configurations disagree (✓ correct · ~ partial · ✗ incorrect)

| id | type | question | struct:rerank | struct:hybrid | struct:dense | fixed500:rerank |
|---|---|---|---|---|---|---|
| q01 | lookup | What is the maximum fine for engaging in a prohibited AI practice? | ✓ | ~ | ✗ | ~ |
| q02 | lookup | What fine applies when an operator breaches AI Act obligations other t | ✓ | ~ | ✗ | ✗ |
| q06 | definition | Who is a 'provider' under the AI Act? | ✓ | ~ | ✓ | ~ |
| q07 | definition | Who counts as a deployer under the AI Act? | ✓ | ✓ | ✓ | ~ |
| q10 | transparency | What must providers ensure when people interact with an AI chatbot? | ✓ | ~ | ✓ | ~ |
| q12 | cross-reference | Is an AI system used to screen job applications considered high-risk? | ✓ | ✓ | ✓ | ✗ |
| q13 | cross-reference | Is AI used for credit scoring of individuals high-risk? | ✓ | ✓ | ✓ | ~ |
| q14 | cross-reference | Does the AI Act treat AI used to price life insurance as high-risk? | ~ | ✓ | ~ | ✗ |
| q15 | cross-reference | When is an AI system listed in Annex III not considered high-risk? | ✓ | ~ | ✓ | ✓ |
| q18 | obligation | Must an employer tell workers before using a high-risk AI system at th | ✓ | ~ | ✓ | ✓ |
| q19 | obligation | Who has to carry out a fundamental rights impact assessment? | ~ | ~ | ~ | ✓ |
| q23 | scope | Does the AI Act apply to research and development of AI before it is p | ~ | ~ | ✓ | ✓ |
| q24 | timeline-amended | From when do the obligations for high-risk AI systems listed in Annex  | ✗ | ✗ | ✓ | ~ |
| q25 | timeline-amended | From when do the high-risk rules apply to AI systems covered by the pr | ~ | ~ | ✗ | ✗ |
| q26 | timeline | When did the AI Act's prohibitions on AI practices start to apply? | ✓ | ~ | ✓ | ✗ |
| q28 | unanswerable | Which Irish authority enforces the AI Act in Ireland? | ✓ | ✓ | ✓ | ~ |
| q29 | unanswerable | How much does it cost to register a high-risk AI system in the EU data | ✗ | ✓ | ✓ | ✓ |

## Indexes

| arm | collection | documents | chunks | avg_chunk_tokens | index_seconds |
|---|---|---|---|---|---|
| struct | docs_struct_cs1000_ov100 | 383 | 443 | 387 | cached |
| fixed500 | docs_fixed_cs500_ov50 | 2 | 277 | 442 | 24.9 |

## Notes

_Write what you observed: why the flipped questions flipped (check results/*.jsonl)._