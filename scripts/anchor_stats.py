"""How much context does the corpus lose? For every prose paragraph of the included articles: does it name
its year, its series or tournament, which match it is, the ground, a team?

    uv run scripts/anchor_stats.py

A paragraph is one prose line of data/cricket/text (match boxes, tables and lists are left out: they are counted
as structured data). A 300-token chunk is about one to two paragraphs, so this is what most chunks will look like.
"anchored" = the paragraph names both its year and its series/tournament, so it could be placed with no context.
The last column adds the article title and the section headings above the paragraph (what a free metadata
header could give each chunk) and asks the same question again.
"""

import json
import re
from collections import defaultdict

from build_manifest import MANIFEST
from clean_wiki import TEXT

YEAR = re.compile(r"\b(19|20)\d{2}\b")
SERIES = re.compile(r"Ashes|Border.Gavaskar|World Cup|World Twenty20|T20 World Cup|Champions Trophy|"
                    r"World Test Championship|Indian Premier League|\bIPL\b|Pataudi|Anderson.Tendulkar|Trophy",
                    re.I)
TOUR = re.compile(r"cricket team in \w+ in \d{4}")  # a tour title names its series: "Indian cricket team in Australia in 2014–15"
MATCH = re.compile(r"\b(1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth) (Test|ODI|T20I?)\b|\bfinal\b|"
                   r"semi-final|Qualifier|Eliminator|\bMatch \d+", re.I)
PRONOUN_START = re.compile(r"^(He|She|They|His|Their|The visitors|The hosts|The home side|The tourists|"
                           r"The pair|The duo|The side|The team)\b")


def names_from_boxes(texts: list[str]) -> tuple[re.Pattern, re.Pattern]:
    """Ground/city names and team names, taken from the corpus's own match boxes."""
    venues, teams = set(), set()
    for t in texts:
        for v in re.findall(r"^Venue: (.+)$", t, re.M):
            venues |= {p.strip() for p in v.split(",") if len(p.strip()) > 3}
        for a, b in re.findall(r"^Match: .* \| (.*) v (.*)$", t, re.M):
            teams |= {re.sub(r"^\([HA]\) ", "", x).strip() for x in (a, b)}
    alt = lambda names: re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)))
    return alt(venues), alt(teams | {"Australian", "Indian", "English"})


def main() -> None:
    entries = [e for e in json.loads(MANIFEST.read_text()) if e["include"]]
    texts = {e["doc_id"]: (TEXT / f"{e['doc_id']}.txt").read_text() for e in entries}
    venue, team = names_from_boxes(list(texts.values()))
    checks = {"year": YEAR, "series": SERIES, "match": MATCH, "ground": venue, "team": team}
    stats = defaultdict(lambda: defaultdict(int))
    for e in entries:
        path = {}
        for line in texts[e["doc_id"]].splitlines():
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                path = {k: v for k, v in path.items() if k < level} | {level: line.lstrip("# ")}
                continue
            if (line.startswith(("- ", "  ")) or " | " in line or len(line.split()) < 25 or
                    re.match(r"^(Match|Result|Venue|Umpires|Player of the match|\w[\w ]*): ", line)):
                continue  # not prose
            s = stats[e["tier"]]
            s["paragraphs"] += 1
            for name, rx in checks.items():
                s[name] += bool(rx.search(line))
            s["anchored"] += bool(YEAR.search(line) and SERIES.search(line))
            s["pronoun_start"] += bool(PRONOUN_START.match(line))
            header = " ".join(path.values())  # title + section headings above this paragraph
            s["anchored_with_header"] += bool(YEAR.search(header + line) and
                                              (SERIES.search(header + line) or TOUR.search(header)))
    cols = ["year", "series", "match", "ground", "team", "pronoun_start", "anchored", "anchored_with_header"]
    print("share of prose paragraphs that name ...")
    print(f"{'tier':17} {'paras':>6} " + " ".join(f"{c[:10]:>10}" for c in cols))
    total = defaultdict(int)
    for tier, s in stats.items():
        for k, v in s.items():
            total[k] += v
        print(f"{tier:17} {s['paragraphs']:>6} " + " ".join(f"{s[c] / s['paragraphs']:>10.0%}" for c in cols))
    print(f"{'all':17} {total['paragraphs']:>6} " + " ".join(f"{total[c] / total['paragraphs']:>10.0%}" for c in cols))


if __name__ == "__main__":
    main()
