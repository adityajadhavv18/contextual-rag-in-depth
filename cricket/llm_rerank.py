"""r4h: an LLM reranker. gpt-4o-mini reads the question and the fused shortlist and lists the passages that answer it.

The cross-encoders (r4a-c) are small models trained on web-search prose. They made reading mistakes an LLM doesn't:
  "third morning" of a Test  -> they took the Day Two chunk (no link from "third morning" to the Day Three section)
  Root, Edgbaston 2023        -> they took Root's 2022 chunk, because it also says "second innings ... first Test"
  "Hazlewood 5/8 (5 overs)"   -> a scorecard row, not a sentence: buried under prose about Hazlewood
This is listwise reranking (RankGPT, Sun et al., 2023): one call sees every candidate at once, each under its r2b label
(article › section, match), and returns their numbers best first. Candidates it leaves out keep their fused order
after the ones it picked, so a bad reply can never lose a chunk.

Replies are cached per question and shortlist (data/cricket/cache/llm_rerank_<model>.json) with their tokens and time,
like HyDE's passages: re-runs cost nothing and the ladder still counts the original call.
"""

import hashlib
import json
import re
import time
from functools import cache

from eval_harness import settings, traced_openai

import hyde
from ingest import CACHE
from retrieve import timed

PROMPT = """You rank passages from cricket Wikipedia articles for a search engine. Each passage starts with its label:
the article › section, then the match (date | teams | venue) when it belongs to one.

Question: {question}

{passages}

Which passages answer the question? List their numbers, best first: passages that answer it fully, then ones that
answer part of it. At most {keep} numbers, comma-separated, nothing else.
- The passage must be about the exact match, innings, day or season the question asks about. A day named in words
  ("the second morning") is that day's section.
- Scorecard lines answer number questions: "Bowler 4/31 (10 overs)" is wickets/runs (overs), "Batter 87* (112)" is
  runs (balls), * means not out."""
KEEP = 10
CACHE_FILE = CACHE / f"llm_rerank_{settings.llm_model}.json"


@cache
def _cache() -> dict:
    return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}


def llm_rerank(question: str, candidates: list, text_of) -> list:
    """Candidates re-sorted: the LLM's picks in its order, then the rest in their fused order."""
    texts = [text_of(h) for h in candidates]
    key = hashlib.sha1("\n\n".join([question, *texts]).encode()).hexdigest()
    store = _cache()
    cached = key in store
    if not cached:
        passages = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, start=1))
        start = time.perf_counter()
        resp = traced_openai().chat.completions.create(
            model=settings.llm_model, temperature=0, max_tokens=60,
            messages=[{"role": "user", "content": PROMPT.format(question=question, passages=passages, keep=KEEP)}])
        store[key] = {"reply": resp.choices[0].message.content.strip(), "tokens_in": resp.usage.prompt_tokens,
                      "tokens_out": resp.usage.completion_tokens, "ms": (time.perf_counter() - start) * 1000}
        CACHE_FILE.write_text(json.dumps(store, indent=1))
    hit = store[key]
    hyde.USAGE["tokens_in"] += hit["tokens_in"]
    hyde.USAGE["tokens_out"] += hit["tokens_out"]
    hyde.USAGE["llm_ms"] += hit["ms"] if cached else 0.0  # a live call is already inside the harness's timing
    with timed("llm_rerank"):
        picks = list(dict.fromkeys(int(n) - 1 for n in re.findall(r"\d+", hit["reply"])
                                   if 1 <= int(n) <= len(candidates)))
        return [candidates[i] for i in picks] + [h for i, h in enumerate(candidates) if i not in picks]
