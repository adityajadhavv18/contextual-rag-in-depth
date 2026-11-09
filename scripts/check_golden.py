"""Check the cricket golden sets against the cleaned corpus (and Cricsheet).

    uv run scripts/check_golden.py

For both splits (tune, test): items load under the harness schema, every source_doc is an included article, and
each evidence fragment
  - is in its source doc, matched the way the harness matches it (whitespace collapsed, lowercase),
  - appears only once in the whole corpus (otherwise a wrong chunk could count as a hit),
  - sits inside one section (chunks never cross a section, so it could never be retrieved whole otherwise).
Scorecard fragments ("Ben Stokes 135* (219)", "Josh Hazlewood 5/8 (5 overs)", "India: 36 (21.2 overs)") are also
looked up in Cricsheet: the match is found from the Match: line of the fragment's section, and the batter's
runs/balls, bowler's wickets/runs/overs or team's total must agree. A match Cricsheet lacks is reported, not failed.
A part may add other wordings of the same fact after " || " (another passage giving the same answer); those must be
verbatim inside one section of some included doc, may appear more than once, and get the same Cricsheet check.
Prints the mix by category (the id prefix), question type and doc.
"""

import re
import sys
from collections import Counter

from eval_harness import load_golden

from coverage import find_match
from find import included
from measure import sections

SPLITS = ["golden_cricket_tune_v1.jsonl", "golden_cricket_test_v1.jsonl"]
TITLES = {}  # doc_id -> title, filled in main()


def norm(text: str) -> str:  # same as eval_harness.metrics._norm
    return re.sub(r"\s+", " ", text).strip().lower()


def _surname_match(name: str, cricsheet_name: str) -> bool:  # "Ben Stokes" vs "BA Stokes"
    return name.split()[-1].lower() == cricsheet_name.split()[-1].lower()


def cricsheet_check(frag: str, section: str, doc: str) -> str | None:
    """None if the fragment isn't a scorecard figure; else a verdict line."""
    bat = re.fullmatch(r"(.+?) (\d+)(\*?) \((\d+)\)", frag)
    bowl = re.fullmatch(r"(.+?) (\d+)/(\d+) \(([\d.]+) overs\)", frag)
    total = re.fullmatch(r"(.+?): (\d+)(?:/(\d+))?d? \(([\d.]+) overs\)", frag)
    if not (bat or bowl or total):
        return None
    box = re.search(r"^Match: (.*) \| (.*) v (.*)$", section, re.M)
    m = box and find_match(*box.groups(), title=TITLES[doc])
    if not m:
        return "⚠️  not in Cricsheet (unchecked)"
    for inn in m["innings"]:
        if bat:
            for p, s in inn["batting"].items():
                if _surname_match(bat[1], p) and s == f"{bat[2]}{bat[3]} ({bat[4]})":
                    return f"✅ Cricsheet {p} {s}"
        if bowl:
            for p, s in inn["bowling"].items():
                w, r, ov = re.match(r"(\d+)/(\d+) \(([\d.]+) ov\)", s).groups()
                if _surname_match(bowl[1], p) and (w, r) == bowl.groups()[1:3] and float(ov) == float(bowl[4]):
                    return f"✅ Cricsheet {p} {s}"
        if total and inn["team"].lower() == total[1].lower() and str(inn["runs"]) == total[2] \
                and float(inn["overs"]) == float(total[4]):
            return f"✅ Cricsheet {inn['team']} {inn['runs']}/{inn['wickets']} ({inn['overs']} ov)"
    return f"❌ Cricsheet disagrees ({m['id']})"


def main() -> None:
    import json
    from build_manifest import MANIFEST
    TITLES.update({e["doc_id"]: e["title"] for e in json.loads(MANIFEST.read_text())})
    raw = included()
    corpus = {doc: norm(t) for doc, t in raw.items()}
    secs = {doc: [(s, norm(s)) for s in sections(t)] for doc, t in raw.items()}
    problems = 0
    for name in SPLITS:
        golden = load_golden(name, known_docs=set(corpus))
        print(f"\n{name}: {len(golden)} items")
        print("  by category:", dict(Counter(g.id.split("-")[0] for g in golden)))
        print("  by type:    ", dict(Counter(g.question_type for g in golden)))
        print("  by doc:     ", len({g.source_doc for g in golden if g.source_doc}), "articles")
        for g in golden:
            for part in (p for p in re.split(r" \.\.\. | / ", g.evidence or "") if p.strip()):
                frag, *alts = part.split(" || ")
                for alt in alts:  # other wordings of the same fact: verbatim in one section of any doc (may repeat)
                    a = norm(alt)
                    homes = [(doc, s) for doc, ss in secs.items() for s, n in ss if a in n]
                    if not homes:
                        print(f"  ❌ {g.id}: alternative not found whole in any section: {alt[:70]!r}")
                        problems += 1
                        continue
                    verdict = cricsheet_check(alt.strip(), homes[0][1], homes[0][0])
                    print(f"  {g.id}: alt {alt.strip()[:60]!r} in {sorted({d for d, _ in homes})}"
                          + (f" → {verdict}" if verdict else ""))
                    problems += bool(verdict and verdict.startswith("❌"))
                f = norm(frag)
                hits = {doc: text.count(f) for doc, text in corpus.items() if f in text}
                if g.source_doc not in hits:
                    print(f"  ❌ {g.id}: evidence not found in {g.source_doc}: {frag[:70]!r}")
                    problems += 1
                    continue
                if sum(hits.values()) > 1:
                    print(f"  ❌ {g.id}: evidence appears {sum(hits.values())}x ({hits}): {frag[:60]!r}")
                    problems += 1
                home = [s for s, n in secs[g.source_doc] if f in n]
                if not home:
                    print(f"  ❌ {g.id}: evidence crosses a section boundary: {frag[:60]!r}")
                    problems += 1
                elif verdict := cricsheet_check(frag.strip(), home[0], g.source_doc):
                    print(f"  {g.id}: {frag.strip()!r} → {verdict}")
                    problems += verdict.startswith("❌")
    print(f"\n{'✅ all evidence unique, verbatim and in one section' if not problems else f'❌ {problems} problem(s)'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
