# Contextual RAG #2: cricket on Wikipedia

Ask "how was Kohli out in the Adelaide day-night Test in December 2020?" and get "run out for 74 after a mix-up with Rahane: Hazlewood fielded and Lyon broke the stumps [2]", from the right paragraph of the right article. The search is built so that a chunk which never names its own match can still be found.

```
INDEX (once):  140 Wikipedia articles → section-aware chunks (300/50 tokens, never cross a heading): 4,650 chunks
               → index text = free label (title › headings + match box) + LLM note (what happens in it) + chunk
               → Qdrant: dense (bge-small) + BM25 + SPLADE vectors per chunk

QUERY:  question ─┬─ gpt-4o-mini writes a fake Wikipedia passage → dense search with it ─┐
                  ├─ dense search with the question itself ───────────────────────────────┤ weighted RRF (k=20,
                  ├─ BM25 on the question ────────────────────────────────────────────────┤ keyword lists ×1.5)
                  └─ SPLADE on the question ──────────────────────────────────────────────┘ → top 20
                  → gpt-4o-mini reads the 20 under their labels, ranks them → top 5
                  → gpt-4o-mini answers with [n] citations, or "I don't know"
```

Think of pages torn out of a book. The page that says "Rahane was run out by Labuschagne combining with Tim Paine for 112" is the answer, but it doesn't say which Test, which series or which year. Those are in the chapter titles. On their own, torn-out pages are found by the words on them, so the search keeps returning the *signposts* instead: scorecards, article leads and player bios that do name the match. Contextual retrieval writes the missing heading back on top of each page before it is filed.

In this corpus only **31%** of report paragraphs name their own year and series (10% in the India–Australia tours). With the article title and section headings added, that rises to 71%.

The parts, each measured as one rung of a ladder:

- **Contextual Retrieval** (Anthropic, 2024): add context to each chunk's *index* text; the LLM still reads the bare chunk. Here in two forms: a **free label** built by code from the article's own structure, and an **LLM note**.
- **BM25**, **SPLADE** (Formal et al., 2021), **Reciprocal Rank Fusion** (Cormack et al., 2009) and **HyDE** (Gao et al., 2022): Hybrid RAG #1's stack, carried over unchanged.
- **Listwise LLM reranking** (RankGPT, Sun et al., 2023): one call sees all 20 candidates and orders them.
- Tried and dropped: three **cross-encoder rerankers**, blending a reranker with the fused order, and a question **router** (that one belongs to Adaptive RAG).

## The corpus and the golden set

140 English Wikipedia articles, each pinned to a revision (`data/cricket/manifest.json`), about 1.02 M tokens. The articles aren't committed: `scripts/download_wiki.py` fetches them.

| Tier | Articles |
|---|---|
| IPL seasons and finals | 36 |
| Players | 21 |
| T20 World Cups | 18 |
| Grounds | 15 |
| ODI World Cups | 14 |
| Border–Gavaskar, Ashes, India in England | 9 each |
| Champions Trophy | 6 |
| World Test Championship finals | 3 |

The corpus was capped at about 5,000 chunks, so 9 player biographies (the least cited elsewhere) were left out. **Cricsheet** ball-by-ball data (11,252 matches) is used only to check facts: 86% of the corpus's match boxes were found in it, and every scorecard figure in the golden set agrees with it.

The golden set (in `eval_harness`) has 50 questions: **30 tune / 20 held-out test**. The test questions use matches that no tune question uses. Every setting below was chosen on tune; the test split was run once, at the end. The categories are the traps this corpus sets:

- **which**: which match? "how was Root out in England's second innings of the first Test of the 2023 Ashes?"
- **twin**: the same players in another match ("Kohli in the Adelaide Test of December **2020**", not 2014 or 2018)
- **pron**: the answer paragraph says "the visitors" or "he", never the name
- **num**: scorecard figures ("Hazlewood's figures when India were bowled out for 36")
- **multi**: two facts in one question
- **bio**: player and ground articles
- **none**: no answer exists (the *fifth* Test of a four-Test series). The right reply is "I don't know"

Each answer has a verbatim evidence quote, checked by `scripts/check_golden.py`: in its article, once in the whole corpus, inside one section. When another passage states the same fact in other words (a scorecard line and a match report both giving "208/7"), it is listed after ` || ` and also counts.

## How it works

**1. Clean text** (`scripts/clean_wiki.py`). HTML → text with `#` headings. Match boxes become labelled lines (`Match: 17–21 December 2020 (D/N) | India v Australia`, `Venue: Adelaide Oval`), tables become ` | ` rows, and references, navboxes and off-topic sections (rugby, concerts, personal life) are dropped.

**2. Section-aware chunks** (`cricket/chunks.py`). Each section is cut into 300-token windows with 50 overlapping. A chunk never crosses a heading, so it belongs to exactly one place: "1st Test › Day Three".

**3. The free label** (`chunks.py`), built by code, no API. The title and heading path, plus the match the chunk belongs to:

```
Indian cricket team in Australia in 2020–21 › Test series › 2nd Test › Day Three
26–30 December 2020 | Australia v India | Melbourne Cricket Ground, Melbourne
```

The match is the box in effect where the chunk starts. A "Day Three" section inherits the box printed under "2nd Test". A box that prints no year ("10–14 July") takes it from the article title, with the season rule: a 2020–21 tour's January match is 2021. In an article about one final, its one box labels every chunk.

**4. The LLM note** (`cricket/notes.py`). gpt-4o-mini reads the label, the chunk's whole section and the chunk, and writes what *this* chunk adds:

> India's first innings ends at 326, with Ajinkya Rahane scoring 112 before being run out, and Ravindra Jadeja contributing 57. Australia, in their first innings, reached 6/133 at the end of Day Three…

The prompt shows the model the label and forbids repeating it. The note's job is to tell Day Three apart from Day Four, not to restate the match. It cost $2.03 for all 4,650 chunks.

**5. Index text ≠ stored text** (`cricket/ingest.py`). The three searches read label + note + chunk. The answer step and the evidence check read the bare chunk, so every rung is scored on the same text.

**6. Four lists, one merge, one rerank** (`cricket/ladder.py`, `cricket/llm_rerank.py`):

```python
fake  = gpt-4o-mini("write the Wikipedia passage that answers: " + question)    # HyDE, cached per question
lists = {"dense":          dense_search(embed_as_passage(fake), 20),  # meaning, searched with the fake passage
         "dense_question": dense_search(question, 20),                # safety net when HyDE invents facts
         "bm25":           sparse_search(question, 20),               # exact names, scores, grounds
         "splade":         sparse_search(question, 20)}               # learned keywords
top20 = rrf_fuse(lists, k=20, weights={"bm25": 1.5, "splade": 1.5})[:20]
top5  = gpt-4o-mini("which of these 20 labelled passages answer the question? best first")[:5]
```

**7. Answer** (`cricket/answer.py`). The top 5 go to gpt-4o-mini under their labels. The prompt teaches three cricket reading rules: `87* (112)` is 87 not out off 112 balls; a partnership is the figure the text gives, never the sum of the two batters' scores; a bowler's figures belong to the *other* side's innings. The model cites `[n]` or replies exactly "I don't know based on the provided documents."

## Run it

From `contextual_rag_in_depth/`. Qdrant runs in Docker (`open -a Docker`, then `docker start qdrant`). The OpenAI key is needed for HyDE, notes, the reranker and answers.

```bash
uv run scripts/build_manifest.py && uv run scripts/download_wiki.py && uv run scripts/clean_wiki.py
uv run scripts/measure.py --apply           # pick articles to fit ~5,000 chunks
uv run scripts/download_cricsheet.py        # only for fact checks
uv run scripts/check_golden.py              # every evidence quote: verbatim, unique, one section, Cricsheet
uv run cricket/ingest.py plain header header_match    # rungs 0-2 (~1 h of embedding on a laptop)
uv run cricket/notes.py --pilot             # the note prompt on 61 chunks, both context modes (~$0.02)
uv run cricket/notes.py                     # notes for every chunk (~$2, once)
uv run cricket/ingest.py header_match_note  # rungs 3-4
uv run cricket/ladder.py r0 r2b r4j         # retrieval-only ladder on tune
uv run cricket/answer.py r4j                # answers + judge on tune
uv run cricket/answer.py r0 r1 r4j --test   # the final held-out run
```

## Results

### The ladder (tune, 28 answerable questions, retrieval only)

Each rung carries the best rung before it. One question moves MRR by about 0.04. p50 latency includes the HyDE call (~3.2 s).

| Rung | What changed | recall@5 | MRR | recall@20 | p50 | Kept? |
|---|---|---|---|---|---|---|
| r0 | dense only (bge-small) | 0.34 | 0.20 | 0.50 | 12 ms | baseline |
| r1a | + BM25 + SPLADE, weighted RRF | 0.41 | 0.24 | 0.59 | 64 ms | ✅ |
| r1 | + HyDE on the dense list (Hybrid #1's best stack) | 0.45 | 0.28 | 0.59 | 3.4 s | ✅ |
| r2a | + free label: title › heading path | 0.75 | 0.45 | 0.80 | 3.3 s | ✅ |
| **r2b** | **+ free label: the match box** | **0.84** | **0.62** | **0.96** | 3.3 s | ✅ |
| r3a / r3b | + LLM note (in all lists / in dense only) | 0.82 / 0.82 | 0.62 / 0.61 | 0.95 | 3.3 s | tie, kept for r4j |
| r4a–c | + cross-encoder rerank of the top 20 (MiniLM / jina-turbo / bge-base) | 0.88 / 0.84 / 0.75 | 0.63 / 0.73 / 0.53 | 0.96 | 4.0–10.3 s | ❌ |
| r4d | notes + bge-base rerank | 0.79 | 0.52 | 0.95 | 12.2 s | ❌ |
| r4e / r4f | jina's order RRF'd with the fused order (1:1 / jina ×2) | 0.84 / 0.80 | 0.71 / 0.72 | 0.96 | 4.3 s | ❌ |
| r4h | + LLM listwise rerank of the top 20 (on r2b) | 0.95 | 0.84 | 0.96 | 4.8 s | ✅ |
| **r4j** | **r4h on the note index + a dense list with the question itself** | **0.91** | **0.89** | **0.95** | **4.6 s** | ✅ **winner** |
| r4k | r4j without the notes | 0.91 | 0.84 | 0.91 | 4.4 s | ❌ |

r4h–r4k were scored after the ` || ` evidence alternatives were added; the rungs above them, before. Re-scored, the earlier rungs could only go up slightly.

MRR by category:

| | which | twin | pron | num | multi | bio |
|---|---|---|---|---|---|---|
| r0 | 0.17 | 0.25 | 0.00 | 0.25 | 0.19 | 0.40 |
| r1 | 0.21 | 0.15 | 0.00 | 0.55 | 0.30 | 0.58 |
| r2b | 0.62 | 0.47 | 0.50 | 0.67 | 0.83 | 0.67 |
| r4b (jina) | 0.88 | 0.50 | 0.84 | 0.32 | 1.00 | 1.00 |
| r4j | 0.75 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 |

The free label took **pron** from 0.00 to 0.50 and **multi** from 0.30 to 0.83. The LLM reranker fixed what the cross-encoders broke: scorecard questions (**num**) went from 0.32 to 1.00.

On tune, r4j's answers were **0.92** correct: 0.85 with the first answer prompt, 0.92 once it had the three cricket reading rules.

### The final run (held-out test, 18 answerable + 2 no-answer, judged)

`cricket/results/answers_test.json` holds every number and answer.

| | r0 (dense only) | r1 (Hybrid #1 stack) | **r4j** |
|---|---|---|---|
| **answers correct** | 0.60 | 0.55 | **0.95** |
| evidence in top 5 (recall@5) | 0.50 | 0.56 | **1.00** |
| MRR | 0.32 | 0.46 | **0.91** |
| faithfulness | 0.81 | 0.82 | **0.91** |
| answer relevancy | 0.68 | 0.58 | **0.94** |
| context precision | 0.43 | 0.53 | **0.93** |
| context recall | 0.52 | 0.51 | **0.94** |
| answerable questions refused | 28% | 39% | **0%** |
| no-answer questions refused | 100% | 100% | 100% |
| cost / question | $0.00025 | $0.00037 | $0.00126 |
| p50 latency | 1.2 s | 3.7 s | ~5 s * |

\* Measured 2.5 s, but r4j reused the HyDE passages that r1's run had just cached; with a live HyDE call (~2.4 s) it is about 5 s.

r4j is 1.00 in every category except **twin** (0.67, one question of three). Test came out higher than tune (0.95 vs 0.92), the opposite of Hybrid #1. With 20 questions one answer is worth 0.05, so read both as one band: about 0.92–0.95.

"Correct" is our own grade: RAGAS has no correctness metric, so the judge (gpt-4o-mini) compares each answer with the golden reference: correct 1, partial 0.5, wrong 0. A refusal of an answerable question scores 0.

The whole build cost about **$2.60** in API calls, $2.07 of it for the notes. The judge cost $0.13 for the final run.

## What went wrong, and why

- **Signposts win until chunks carry their context (r0–r1).** Hits were scorecard boxes, article leads, bios and ground pages: the chunks that *name* the match. The day-by-day report chunks that hold the answers named nothing, and **pron** scored 0.00 on every rung until r2.
- **Match boxes are irregular.** Many print no year ("10–14 July"); WTC finals print the box in a sibling section of the report; ground articles have exactly one box that belongs to one match only; IPL final articles have none. Each needed its own rule in `chunks.py`.
- **One label per match can't tell days apart (r2b).** Every chunk of a Test gets the same label, so the remaining misses were the right match but the wrong day: Rahane's 112 at #18 behind Day Four, Root's stumping behind "Day one", which also mentions Carey's stumpings.
- **The LLM notes didn't pay on their own (r3).** The pilot's first prompt skipped each chunk's opening sentences and began most notes "This chunk covers…" (the same boilerplate that sank Hybrid #1's notes). The fixed prompt lifted buried answers (#18 → #7) but every rival chunk got a rich note too, so the top ranks slipped and MRR stayed at 0.62. The notes only paid in r4j, together with the question-dense list: MRR 0.84 → 0.89, pron 1.00. The pilot also underestimated the full run (estimated $0.95, actual $2.03): its chunks came from smaller sections than average.
- **Cross-encoders bury scorecards (r4a–c).** They are trained on web prose, and `Josh Hazlewood 5/8 (5 overs)` is a table row, not a sentence. All three dropped **num** to 0.32–0.40. bge-reranker-base, the best in Hybrid #1, was the worst here (MRR 0.53). Blending jina's order back with the fused order saved scorecards but gave back the prose gains.
- **A router helped, but it isn't this architecture.** Sending number questions past the reranker reached MRR 0.77, but gpt-4o-mini misrouted 2 of 5 number questions, and routing by question type *is* Adaptive RAG. It was removed and kept as an idea for that build.
- **HyDE can invent the wrong facts.** For the 2017 IPL final its fake passage says "*Bumrah* bowled the last over" (it was Johnson), and dense search pushed the answer chunk to #164. The question-dense list is the fix in general, but this one question is still unreachable: ~40 chunks from the same final and season name both teams, and the answer sits at fused #42 even with 40 per list. A deeper pool doesn't help.
- **The scoring needed fixes too.** The evidence rule credited only one exact quote, so a top chunk saying "set up a big total of 208/7" counted as a miss: hence ` || ` alternatives (checked against Cricsheet). And the judge graded some "I don't know" answers as *correct*, which inflated r0 to 0.85 and r1 to 0.90 before refusals were scored in code.
- **The answer model misreads cricket.** It summed two batters' scores as their partnership ("91 + 97 = 188"), refused `135* (219)` because it didn't read the balls, and named Archer, whose five wickets came in the *other* innings. The reading rules fixed most of it, but they also made the model bolder: when the answer isn't in the top 5 it now guesses (Bumrah for the IPL final) instead of refusing.
- **The one test miss:** twin-08, Joe Root's first century in Australia. Right Test (the Gabba, December 2025), wrong score: 104 instead of 138\*. Retrieval found the right passage; the answer model took a number from another excerpt.

## What it gets right

- **Chunks that don't name their match are found anyway.** The free label alone took recall@5 from 0.45 to 0.84, at no API cost and no extra search time.
- **Scorecards and prose both.** The LLM reranker reads `5/8 (5 overs)` and "the visitors" equally well: num, pron, multi and bio all at 1.00 on tune.
- **The right passage reaches the answer step.** recall@5 on test is 1.00; the answer is in the top 20 for 27 of 28 tune questions.
- **Honest refusals.** Both no-answer questions refused on tune and test, with no answerable question refused.
- **Cheap.** About $0.0013 a question, on an 8 GB laptop with local embedders. Only HyDE, the reranker and the answer call reach OpenAI.
- **Every step is measured.** Each rung changed one thing on the same chunks and the same golden set.

## Ideas not built

- **An innings and day label per chunk.** The headings give the day; the innings ("India's first innings") could come from the box's team order and the text, for free, instead of from the notes.
- **A match line for IPL finals.** Their articles print no box, so the title is the only label. Building one from the infobox would give them what every Test chunk has.
- **Refuse when the excerpts don't name the asked match.** A check step (or Corrective RAG) would turn the two remaining guesses into "I don't know".
- **A stronger answer model** for the reading traps (gpt-4o: ~15× the answer cost).
- **Re-tune RRF for four lists.** The weights (k=20, keyword ×1.5) come from Hybrid #1's three-list stack.
- **Route number questions** (Adaptive RAG): the router idea from r4, with a measured router accuracy.

## When I'd reach for it

When answers sit in paragraphs that **don't name their own subject**: match reports, meeting minutes, case notes, long policies, any document where "the second Test" or "the defendant" only makes sense under its headings. Start with the **free label**: if the documents have titles, headings or structured boxes, writing them onto each chunk's index text is free and here gave the biggest single gain (MRR 0.28 → 0.62). Add **LLM notes** only when the structure runs out, and make the prompt say what is *different* about each chunk, or the notes make neighbours look alike. Add an **LLM reranker** when the top 20 already holds the answer (recall@20 ≥ 0.9) and the documents mix prose with tables that cross-encoders misread. Skip all this for documents whose paragraphs already name what they're about: plain hybrid search is enough there.

## Files

| File | Purpose |
|---|---|
| `scripts/wiki.py` | Polite Wikipedia API client (1 s between calls, retries) |
| `scripts/build_manifest.py` | Candidate articles by tier → `manifest.json` with pinned revisions |
| `scripts/download_wiki.py` | Fetches each article at its pinned revision into `data/cricket/raw/` (git-ignored) |
| `scripts/clean_wiki.py` | HTML → text with `#` headings, labelled match boxes, ` \| ` tables; writes `articles.json` |
| `scripts/measure.py` | Estimates chunks per article; picks articles to fit ~5,000 chunks |
| `scripts/download_cricsheet.py` | Cricsheet zips → per-innings batting/bowling index (fact checks only) |
| `scripts/coverage.py` | How many match boxes Cricsheet finds; date and team matching |
| `scripts/anchor_stats.py` | How many paragraphs name their own year, series, match and ground |
| `scripts/find.py` | Search helper for writing golden questions |
| `scripts/check_golden.py` | Every evidence quote: verbatim, unique, one section; scorecard figures vs Cricsheet |
| `cricket/chunks.py` | Section-aware 300/50 chunks, heading paths and match labels |
| `cricket/ingest.py` | One Qdrant collection per index-text variant, dense + BM25 + SPLADE; caches model outputs |
| `cricket/retrieve.py` | Dense and sparse search, cross-encoder rerank, weighted RRF, per-stage timing |
| `cricket/hyde.py` | The fake-passage prompt, its cache and the HyDE query vector |
| `cricket/notes.py` | LLM notes per chunk (chunk or section mode), the pilot and its similarity check |
| `cricket/llm_rerank.py` | Listwise LLM reranker, cached per question and shortlist |
| `cricket/ladder.py` | Every rung as a search function; scores them on tune (recall@5, MRR, nDCG@5, recall@20, by category) |
| `cricket/answer.py` | Answers on any rung, the harness judge, the correctness grade; tune or test |
| `cricket/results/ladder_tune.json` | Every rung's tune numbers |
| `cricket/results/answers_tune.json`, `answers_test.json` | Judged answers, by category and per question |
