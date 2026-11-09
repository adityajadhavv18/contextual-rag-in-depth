"""Answer generation: a rung's top 5 chunks -> gpt-4o-mini -> an answer with [n] citations, judged.

The ladder scored retrieval only (empty answers). This adds the last step, so the harness can judge the ANSWER:
RAGAS faithfulness / relevancy / context precision and recall, and abstention on the no-answer questions (the fifth
Test of 2014-15: there were four). Each chunk is shown under its r2b label (article › section, match), so the model
knows which match a passage is about; the harness gets that same labelled text as the contexts, so faithfulness is
judged on exactly what the model read. Refusals use the exact line the harness counts: answers that START with
"I don't know".

RAGAS has no "is it right?" score, so this file adds one (correctness): the judge model compares the answer with the
golden reference answer: correct 1, partial 0.5, wrong 0. A no-answer question is correct when the harness's
abstention judge says it declined. Results by category go to cricket/results/answers_<split>.json.

Run from contextual_rag_in_depth/ (Qdrant running, OpenAI key sourced):
    uv run cricket/answer.py r4j                # tune split + judge (~$0.20)
    uv run cricket/answer.py r0 r1 r4j --test   # the final run: held-out test split
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from eval_harness import PipelineOutput, evaluate, load_golden, settings, traced_openai
from eval_harness.metrics import cost_usd

import hyde
from ladder import RUNGS, TOP_K, TUNE, labelled

TEST = "golden_cricket_test_v1.jsonl"
NO_ANSWER = "I don't know based on the provided documents."
RESULTS = Path(__file__).resolve().parent / "results"

PROMPT = """You answer cricket questions using ONLY the Wikipedia excerpts below. Each excerpt starts with its label:
the article › section, then the match (date | teams | venue) when it belongs to one.
Use the excerpts about the exact match, innings, day or season the question asks about; others may be about a
different match with the same teams or players. Answer every part of the question in one or two sentences, with
the names and numbers the excerpts give. Cite excerpts like [2].
How to read cricket excerpts:
- Scorecard lines: "Batter 87* (112)" is 87 runs off 112 balls, * means not out; "Bowler 4/31 (10 overs)" is 4
  wickets for 31 runs in 10 overs; "Team: 250 (80.2 overs)" is the team's total.
- A partnership or stand ("added", "put on", "shared") is the figure the text gives for it. Never add up the two
  batters' own scores: they include runs made with other partners.
- A bowler's figures belong to the innings the OTHER team batted. When a team "was bowled out for 67", the wickets
  went to the bowlers of the side in the field.
If the excerpts do not answer the question, reply exactly: "{no_answer}"

Excerpts:
{context}

Question: {question}"""

GRADE = """Question: {question}
Reference answer: {reference}
Answer given: {answer}

Judge only the facts the QUESTION asks for; the reference may add details the question didn't ask about, and an
answer may leave those out. Extra details in the answer are fine unless they contradict the reference. An answer
that declines ("I don't know") is wrong. Reply with exactly one word: correct (every asked-for part right), partial
(some asked-for parts right, others wrong or missing), or wrong (the asked-for facts are wrong or missing)."""
GRADES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def make_pipeline(rung: str):
    """question -> PipelineOutput for this rung: its retrieval, then one generation call."""
    search = RUNGS[rung][1]
    client = traced_openai()

    def pipeline(question: str) -> PipelineOutput:
        hyde.USAGE.update(tokens_in=0, tokens_out=0, llm_ms=0.0)  # HyDE and the LLM reranker add their calls here
        hits = search(question, TOP_K)
        texts = [labelled(h) for h in hits]
        context = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, start=1))
        resp = client.chat.completions.create(
            model=settings.llm_model, temperature=0,
            messages=[{"role": "user", "content": PROMPT.format(no_answer=NO_ANSWER, context=context,
                                                                question=question)}])
        return PipelineOutput(answer=resp.choices[0].message.content.strip(), contexts=texts,
                              context_ids=[h.doc_id for h in hits],
                              tokens_in=hyde.USAGE["tokens_in"] + resp.usage.prompt_tokens,
                              tokens_out=hyde.USAGE["tokens_out"] + resp.usage.completion_tokens)

    return pipeline


def grade(question: str, reference: str, answer: str) -> tuple[float, float]:
    """(correctness, judge cost USD) for an answerable question."""
    resp = traced_openai().chat.completions.create(
        model=settings.judge_model, temperature=0, max_tokens=3,
        messages=[{"role": "user", "content": GRADE.format(question=question, reference=reference, answer=answer)}])
    word = resp.choices[0].message.content.strip().lower()
    score = next((v for k, v in GRADES.items() if word.startswith(k)), 0.0)
    return score, cost_usd(resp.usage.prompt_tokens, resp.usage.completion_tokens, settings.judge_model) or 0.0


def main():
    args = sys.argv[1:]
    split = TEST if "--test" in args else TUNE
    out_path = RESULTS / f"answers_{'test' if split == TEST else 'tune'}.json"
    saved = json.loads(out_path.read_text()) if out_path.exists() else {}
    golden = {g.id: g for g in load_golden(split)}
    for rung in (a for a in args if not a.startswith("--")):
        run = evaluate(make_pipeline(rung), arch=f"cricket_{rung}_answer", golden_set=split, use_llm_judge=True,
                       compare_to=None)
        rows, judge_cost = [], run.summary.get("judge_cost_usd", 0.0)
        for q in run.questions:
            g = golden[q.golden_id]
            if g.question_type == "no_answer":
                correct = q.scores.get("abstention", 0.0)
            elif q.output is None:
                correct = 0.0
            else:
                correct, cost = grade(g.question, g.ground_truth, q.output.answer)
                judge_cost += cost
            rows.append({"id": g.id, "correct": correct, **{k: q.scores[k] for k in
                         ("evidence_recall", "faithfulness", "refused") if k in q.scores},
                         "answer": q.output.answer if q.output else q.error})
        by_cat = defaultdict(list)
        for r in rows:
            by_cat[r["id"].split("-")[0]].append(r)
        mean = lambda rs, k: round(statistics.mean(r[k] for r in rs if k in r), 3) if any(k in r for r in rs) else None
        saved[rung] = {"label": RUNGS[rung][0], "correct": mean(rows, "correct"),
                       "faithfulness": run.summary.get("faithfulness"), "recall5": run.summary.get("evidence_recall"),
                       "false_refusal_rate": run.summary.get("false_refusal_rate"),
                       "cost_per_query_usd": run.summary.get("cost_per_query_usd"),
                       "latency_p50_ms": run.summary.get("latency_p50_ms"), "judge_cost_usd": round(judge_cost, 4),
                       "by_category": {c: {"correct": mean(rs, "correct"), "faithfulness": mean(rs, "faithfulness"),
                                           "n": len(rs)} for c, rs in sorted(by_cat.items())},
                       "per_question": rows}
        out_path.write_text(json.dumps(saved, indent=2, ensure_ascii=False))
        s = saved[rung]
        print(f"\n{rung}: correct {s['correct']:.2f}, faithfulness {s['faithfulness'] or 0:.2f}, judge ${judge_cost:.3f}")
        print("| category | n | correct | faithfulness |\n|---|---|---|---|")
        for c, v in s["by_category"].items():
            print(f"| {c} | {v['n']} | {v['correct']:.2f} | {v['faithfulness'] if v['faithfulness'] is not None else '-'} |")
        for r in rows:
            if r["correct"] < 1:
                print(f"\n  [{r['id']}] correct={r['correct']}  {golden[r['id']].question}\n    -> {r['answer'][:300]}")


if __name__ == "__main__":
    main()
