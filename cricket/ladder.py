"""The retrieval ladder: score each rung on the TUNE golden set, retrieval only (no answer LLM, so ~free).

Each rung is a search function. The harness runs it as a pipeline that returns the top TOP_K chunks and an empty
answer (the harness then scores retrieval only), and this script adds the lab-only numbers:
  nDCG@5     like MRR, but credits every useful chunk by position, not only the first
  recall@20  is the evidence anywhere in the top 20? (what a reranker could still rescue)
  by category (the golden id prefix) and latency per stage.
Results accumulate in cricket/results/ladder_tune.json; the test split is only touched at the very end.

Run from contextual_rag_in_depth/ (Qdrant running; HyDE rungs need the OpenAI key):
    set -a && source ../rag_sprints/.env && set +a && uv run cricket/ladder.py            # all rungs
    ... uv run cricket/ladder.py r0 r1                                                     # just these
"""

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from eval_harness import PipelineOutput, evaluate, load_golden

import hyde
from ingest import BM25, SPLADE, collection
from retrieve import PLAIN, STAGE_MS, dense_search, rrf_fuse, sparse_search

TUNE = "golden_cricket_tune_v1.jsonl"
TOP_K = 5  # chunks the LLM will read
CANDIDATES = 20  # how deep each retriever's list goes into fusion
RESULTS = Path(__file__).resolve().parent / "results" / "ladder_tune.json"

RRF_K = 20  # Hybrid #1's tuned fusion (fusion_sweep.py there), carried over unchanged
KEYWORD_WEIGHTS = {"bm25": 1.5, "splade": 1.5}  # keyword lists count 1.5x the dense list


def hybrid_stack(q: str, k: int, collection: str = PLAIN, use_hyde: bool = True, dense_collection: str = "") -> list:
    """Hybrid #1's best stack (r6a there): dense + BM25 + SPLADE, weighted RRF (k=20, keyword x1.5), and the dense
    list searching with a HyDE fake passage. The keyword lists always search with the question itself.
    dense_collection: let the dense list search another collection of the same chunks (r3b)."""
    vector = hyde.hyde_vector(q) if use_hyde else None
    return rrf_fuse({"dense": dense_search(q, CANDIDATES, dense_collection or collection, vector=vector),
                     "bm25": sparse_search(q, CANDIDATES, BM25, collection),
                     "splade": sparse_search(q, CANDIDATES, SPLADE, collection)},
                    k=RRF_K, weights=KEYWORD_WEIGHTS)[:k]


RUNGS = {
    "r0": ("dense only", lambda q, k: dense_search(q, k)),
    "r1a": ("Hybrid #1 stack without HyDE", lambda q, k: hybrid_stack(q, k, use_hyde=False)),
    "r1": ("Hybrid #1 best stack (+ HyDE on dense)", lambda q, k: hybrid_stack(q, k)),
    # r2: the r1 stack over the same chunks with a FREE label (built by code) in the index text
    "r2a": ("r1 + title and heading path", lambda q, k: hybrid_stack(q, k, collection("header"))),
    "r2b": ("r2a + the match box (date, teams, venue)", lambda q, k: hybrid_stack(q, k, collection("header_match"))),
    # r3: r2b plus an LLM note on what happens in the chunk (notes.py), in all three lists or the dense list only
    "r3a": ("r2b + LLM note (section mode)", lambda q, k: hybrid_stack(q, k, collection("header_match_note"))),
    "r3b": ("r2b, LLM note in dense only", lambda q, k: hybrid_stack(
        q, k, collection("header_match"), dense_collection=collection("header_match_note"))),
}


def norm(text: str) -> str:  # the harness's matching rule
    return re.sub(r"\s+", " ", text).strip().lower()


def fragments(item) -> list[str]:
    return [norm(f) for f in re.split(r" \.\.\. | / ", item.evidence or "") if f.strip()]


def first_rank(frags: list[str], texts: list[str]) -> int | None:
    return next((i for i, t in enumerate(texts, start=1) if any(f in norm(t) for f in frags)), None)


def ndcg(frags: list[str], texts: list[str], k: int = TOP_K) -> float:
    """A chunk scores if it holds an evidence fragment no higher chunk already supplied."""
    found, dcg = set(), 0.0
    for i, t in enumerate(texts[:k], start=1):
        new = {f for f in frags if f in norm(t)} - found
        if new:
            dcg += 1 / math.log2(i + 1)
            found |= new
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(frags), k) + 1))
    return dcg / ideal


def run_rung(key: str) -> dict:
    label, search = RUNGS[key]
    STAGE_MS.clear()
    llm_ms = {}  # question -> LLM time (HyDE) of a passage served from the cache, so cached runs show the real cost

    def pipeline(question: str) -> PipelineOutput:
        hyde.USAGE.update(tokens_in=0, tokens_out=0, llm_ms=0.0)
        hits = search(question, TOP_K)
        llm_ms[question] = hyde.USAGE["llm_ms"]
        return PipelineOutput(answer="", contexts=[h.text for h in hits], context_ids=[h.doc_id for h in hits],
                              tokens_in=hyde.USAGE["tokens_in"], tokens_out=hyde.USAGE["tokens_out"])

    run = evaluate(pipeline, arch=f"cricket_{key}", golden_set=TUNE, use_llm_judge=False, compare_to=None)
    stages = {s: round(statistics.median(ms), 1) for s, ms in STAGE_MS.items()}
    if hyde.USAGE["tokens_in"]:  # an LLM rung: its call time, from the cache (the original call's time)
        stages["llm"] = round(statistics.median(hyde._cache()[q]["ms"] for q in llm_ms), 1)

    golden = {g.id: g for g in load_golden(TUNE)}
    rows, by_cat = [], defaultdict(list)
    for q in run.questions:
        g = golden[q.golden_id]
        if g.question_type == "no_answer":
            continue
        frags = fragments(g)
        deep = [h.text for h in search(g.question, CANDIDATES)]  # outside the timed run
        row = {"id": g.id, "recall5": q.scores["evidence_recall"], "mrr": q.scores["evidence_mrr"],
               "ndcg5": ndcg(frags, q.output.contexts),
               "recall20": sum(any(f in norm(t) for t in deep) for f in frags) / len(frags),
               "rank20": first_rank(frags, deep)}
        rows.append(row)
        by_cat[g.id.split("-")[0]].append(row)

    mean = lambda rs, m: round(statistics.mean(r[m] for r in rs), 3)
    lat = sorted(q.latency_ms + llm_ms.get(golden[q.golden_id].question, 0.0) for q in run.questions)
    result = {
        "rung": key, "label": label, "questions": len(rows),
        "recall5": mean(rows, "recall5"), "mrr": mean(rows, "mrr"), "ndcg5": mean(rows, "ndcg5"),
        "recall20": mean(rows, "recall20"),
        "latency_p50_ms": round(statistics.median(lat), 1), "latency_p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1),
        "stage_p50_ms": stages, "cost_per_query_usd": run.summary.get("cost_per_query_usd", 0.0),
        "by_category": {c: {"recall5": mean(rs, "recall5"), "mrr": mean(rs, "mrr"), "n": len(rs)}
                        for c, rs in sorted(by_cat.items())},
        "per_question": rows,
    }
    misses = [f"{r['id']}(#{r['rank20'] or '>20'})" for r in rows if r["mrr"] < 1]
    print(f"\n{key} {label}: not ranked #1 -> {', '.join(misses) or 'none'}")
    return result


def print_table(results: dict) -> None:
    print("\n| rung | what | recall@5 | MRR | nDCG@5 | recall@20 | p50 ms | $/query |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results.values():
        print(f"| {r['rung']} | {r['label']} | {r['recall5']:.2f} | {r['mrr']:.2f} | {r['ndcg5']:.2f} "
              f"| {r['recall20']:.2f} | {r['latency_p50_ms']:.0f} | {r['cost_per_query_usd']:.5f} |")
    cats = sorted({c for r in results.values() for c in r["by_category"]})
    first = next(iter(results.values()))
    print("\nMRR by category:\n| rung | " + " | ".join(f"{c} (n={first['by_category'][c]['n']})" for c in cats) + " |")
    print("|---|" + "---|" * len(cats))
    for r in results.values():
        print(f"| {r['rung']} | " + " | ".join(f"{r['by_category'][c]['mrr']:.2f}" for c in cats) + " |")


def main():
    keys = sys.argv[1:] or list(RUNGS)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for key in keys:
        results[key] = run_rung(key)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(dict(sorted(results.items())), indent=2))
    print_table(dict(sorted(results.items())))


if __name__ == "__main__":
    main()
