# EU AI Act Q&A: retrieval-augmented generation, measured honestly

A local question-answering system over the EU AI Act, as amended in 2026, that answers with
article-level citations, plus an evaluation harness that measures where it succeeds and fails.
Everything runs on a laptop for free: a local LLM through Ollama, Chroma for vector search,
and Streamlit for the interface.

**Why this problem.** Organisations now have to work out which AI Act obligations apply to them.
The regulation runs to 113 articles and 13 annexes with dense cross-references, and in July 2026
the AI Omnibus (Regulation (EU) 2026/1744) changed key deadlines, so answers based on the original
text can now be wrong. That makes it a demanding test for retrieval-augmented generation: exact
figures, definitions, cross-references between articles and annexes, and amended dates.

## Results

30 hand-written test questions (26 answerable, 4 deliberately unanswerable), four configurations,
same local model (Llama 3.1 8B), top 4 chunks passed to the model:

| Configuration | Answer passage retrieved | Correct | Correct or partial | Refused traps | Grounded | Median time |
|---|---|---|---|---|---|---|
| **Article-based chunks + reranker** | 69% | **73%** | **96%** | 3 of 4 | 97% | 39 s |
| Article-based chunks + hybrid search | 58% | 50% | 96% | 4 of 4 | 100% | 30 s |
| Article-based chunks + semantic search | 54% | 73% | 88% | 4 of 4 | 97% | 24 s |
| Fixed 500-token chunks + reranker | 73% | 50% | 81% | 3 of 4 | 97% | 33 s |

*Correct/partial/grounded are judged by an LLM; see "Checking the judge" below. After human
review of the main configuration, correct stays at 73% but correct-or-partial falls to 85%.*

### What I found

**1. Splitting by legal structure was the biggest win.** With the same reranker, chunking by
article and by definition answered 73% correctly, against 50% for fixed 500-token chunks
(definitions: 2 of 3 vs 0 of 3). Fixed chunks actually retrieved the answer passage *more* often
(73%), but buried among unrelated provisions, so the model picked the wrong one. Asked "who counts
as a deployer?", it answered with the definition of a *downstream provider* from the same chunk.

**2. The reranker improved retrieval, but not the final correct rate.** It raised retrieval of
the answer passage from 54% to 69% over semantic search, and reduced outright wrong answers
(96% vs 88% correct-or-partial). But fully correct answers tied at 73%, each answer took about
15 seconds longer, and it let one trap question through: retrieving more plausible-looking
passages tempts the model to answer instead of refusing.

**3. Retrieval, not the language model, is the main bottleneck.** I read the imperfect answers
by hand: in 6 of 8, the article containing the answer was never retrieved (Article 99, Annex III,
Article 2, Article 113 twice, Article 5). Both questions about amended deadlines failed this way.
Retrieval was also brittle: asking for "high-risk AI systems *in* Annex III" retrieved the
application-date article in the app, while "high-risk AI systems *listed in* Annex III" missed it.

**4. Small differences are noise.** With 26 answerable questions, one question is worth about
4 percentage points, and the judge is a small model. I treat gaps under about 10 points as
inconclusive, which is why I don't claim hybrid search is worse than semantic search.

### Case study: from rank #231 to #1

The first question I tried, "What is the maximum fine for a prohibited AI practice?", returned
"I can't find that in the documents." The answer is in Article 99(3), so I traced where it ranked:

| Search method | Rank of Article 99 |
|---|---|
| Semantic search (embeddings) | #231 of 362 |
| Keyword search (BM25) | #4 |
| Hybrid (reciprocal rank fusion of both) | #18 |
| Hybrid candidates + cross-encoder reranker | **#1** |

Semantic search failed because Article 99's chunk covers ten paragraphs of enforcement rules, so
its overall meaning looked like generic enforcement text, while the question's wording pulled
hard towards Article 5 ("Prohibited AI practices"). Keyword search found it but fusion couldn't
lift it: fusion rewards chunks both methods like, and one strong vote lost to several moderate
ones. Smaller chunks (300 tokens) moved it from #231 to #97 but split Article 5 into pieces that
crowded the results. The real problem was ordering, not recall: the answer was always in the
candidate pool. A cross-encoder reranker, which reads the question and each candidate together,
put it first. One good sign along the way: when retrieval failed, the system refused rather than
inventing a number. (Full output: `results/article99_diagnosis.txt`.)

## How it works

```
EUR-Lex PDFs ─► split_regulation.py ─► one file per recital, article, definition and annex,
                                        each with a context header ("Article 6: Classification…")
            ─► token-capped chunks, header repeated on every chunk
            ─► nomic-embed-text embeddings ─► Chroma

question ─► semantic search (top 20) + BM25 keyword search (top 20)
         ─► cross-encoder reranker (bge-reranker-base) ─► top 4 chunks
         ─► Llama 3.1 8B with a strict prompt ─► answer with [n] citations, or a refusal
```

## Design decisions

**Structure-aware chunking.** Legal text has natural boundaries, so `split_regulation.py` splits
the regulation into recitals, articles and annexes, and splits the definitions article into one
unit per definition. Every chunk starts with a header naming the regulation, article, title and
chapter, so even the second half of a long article says which article it is, and the model can
cite "Article 6" rather than "page 47". The splitter validates itself: it checks article numbers
run in sequence and warns about suspiciously long units, which caught a real bug where one
wrapped title made it swallow 52 articles.

**Handling the amendment.** An amending regulation is written as edits ("paragraph 2 is replaced
by…"), which retrieves poorly and contradicts the original text. Instead I index EUR-Lex's
*consolidated* text (CELEX 02024R1689-20260727), which merges the Omnibus changes into the Act.
When several versions are present, the splitter takes articles and annexes only from the
consolidated one, so outdated provisions never compete in retrieval, and labels amended
provisions using EUR-Lex's amendment markers, tracked per paragraph. The recitals come from the
original Act; the Omnibus stays in the corpus for its explanations of why things changed.

**Retrieval.** Semantic search handles paraphrase but misses exact terms; BM25 keyword search
catches terms like "fine" or "Article 5". The reranker reorders the combined candidates. Top 4
chunks keep the prompt around 2,500 tokens.

**Generation.** Temperature 0 and a prompt that requires a direct answer first, inline citations,
exact figures, no outside knowledge, preference for amended text when passages conflict, and an
exact refusal sentence so unanswerable questions can be detected. Ollama's default context window
is too small for four chunks, so it's raised to 8,192 tokens to avoid silent truncation.

**Embeddings.** `nomic-embed-text` accepts long inputs, so 1,000-token chunks are embedded in
full rather than silently truncated as by many small models (256–512 token limits). It expects
`search_document:` / `search_query:` prefixes, which the code adds.

## Evaluation method

- **Test set** (`eval/questions.jsonl`): 30 questions across lookups (fines), definitions,
  prohibitions, transparency duties, cross-references between articles and Annex III,
  obligations, scope, application dates (two changed by the Omnibus), and four unanswerable
  questions, including a near-miss trap about registration fees.
- **Answer passage retrieved:** a short phrase from the passage that answers the question appears
  in a retrieved chunk. `check_questions.py` verifies every phrase exists in the corpus first.
- **Correct / partial / incorrect** and **grounded** (every claim supported by the retrieved text)
  are judged by Llama 3.1 8B against my reference answers.
- **Refused traps:** the system says it can't find the answer for unanswerable questions.

### Checking the judge

I reviewed 10 of the judge's verdicts by hand (all 8 it didn't mark correct, plus 2 it did) and
agreed with only 5 (`results/spot_check.md`). It was inconsistent in both directions: it gave
partial credit to answers whose headline was wrong (e.g. claiming life-insurance pricing isn't
high-risk), marked a fully correct answer as partial because the wording differed, and its stated
reasons were sometimes false even when the label was right. My conclusion: the LLM judge is useful
for comparing configurations, since its errors affect them similarly, but individual verdicts need
human review. After review, the main configuration is **73% correct and 85% correct or partial**.

The most instructive failure: asked how much it costs to register a high-risk AI system (not in
the Act), the system answered "free of charge", citing a recital saying the EU database is free
to *access*. A real passage, misapplied, with a legitimate-looking citation.

## Limitations

- 30 questions is a small test set; differences under about 10 points are inconclusive.
- The judge is a small local model with about 50% agreement with me on the verdicts I checked.
- Answers take 25–40 seconds on a MacBook Air, too slow for interactive use at scale.
- One run per configuration; I didn't measure run-to-run variation.

## What I'd improve next

- **Query rewriting:** have the model rephrase each question several ways before searching, to
  fix the phrasing sensitivity that caused both amended-deadline failures.
- **Larger candidate pool and cross-reference following:** when a retrieved article cites
  Annex III, retrieve Annex III too.
- **A stronger judge, or a larger human-labelled set**, to make the correctness numbers reliable.
- **A larger answering model** for questions that need care with rules and exceptions.

## Running it

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # the first reranked query downloads the reranker (~1 GB)

# data/docs/ holds the original Act, the consolidated Act and the Omnibus (PDFs from EUR-Lex)
python split_regulation.py --dry-run   # check the article/definition/annex counts
python split_regulation.py             # writes data/docs_structured/
python ingest.py --docs-dir data/docs_structured --tag struct --chunk-size 1000 --overlap 100

python -m streamlit run app.py         # chat UI with search-method selector
python inspect_retrieval.py "your question" --find "Article 99"   # debug retrieval

python check_questions.py              # verify the test set against the index
mkdir -p data/docs_fixed && cp data/docs/ai_act_consolidated_2026.pdf data/docs/ai_omnibus_2026_1744.pdf data/docs_fixed/
caffeinate -i python run_experiment.py # all four configurations, about 2 hours on a MacBook Air
python spot_check.py                   # review the judge's verdicts by hand
```

| File | Purpose |
|---|---|
| `split_regulation.py` | Splits EU regulations into recitals, articles, definitions and annexes |
| `ingest.py` | Chunks, embeds and indexes documents in Chroma |
| `keyword_search.py`, `reranker.py`, `rag.py` | BM25, cross-encoder reranking, retrieval and generation |
| `app.py` | Streamlit chat interface |
| `evaluate.py`, `run_experiment.py` | Evaluation harness and multi-configuration experiment |
| `check_questions.py`, `spot_check.py`, `inspect_retrieval.py` | Test-set validation, judge review, retrieval debugging |
| `results/` | Experiment report, per-question results, spot-check, Article 99 diagnosis |

Source documents are from EUR-Lex, reused under the European Commission's reuse policy.
