"""Which of the corpus's match boxes can be checked against Cricsheet?

    uv run scripts/coverage.py

Every "Match: <date> | A v B" line in an included article is looked up in the Cricsheet index by its first day
and its teams. Prints the share found, per tier. find_match() is what the golden-set checks use to look a match up.
"""

import json
import re
from collections import defaultdict
from datetime import date
from functools import cache

from build_manifest import MANIFEST
from clean_wiki import TEXT
from download_cricsheet import INDEX

MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                       "September", "October", "November", "December"], start=1)}
ALIASES = {"UAE": "United Arab Emirates", "USA": "United States of America", "U.A.E.": "United Arab Emirates",
           "Royal Challengers Bengaluru": "Royal Challengers Bangalore", "Kings XI Punjab": "Punjab Kings",
           "Delhi Daredevils": "Delhi Capitals", "Rising Pune Supergiants": "Rising Pune Supergiant"}


def first_day(text: str, title: str = "") -> str | None:
    """'31 December 2014 – 4 January 2015' -> '2014-12-31'. Many match boxes print no year ('10–14 July'): then it
    comes from the article title, and a "2013–14" season's January-June matches are in its second year."""
    dm = re.search(r"\b(\d{1,2})(?:\s*[–-]\s*\d{1,2})?\s+(" + "|".join(MONTHS) + r")\b", text)  # "9–13 December"
    if not dm:
        return None
    month = MONTHS[dm.group(2)]
    if year := re.search(r"\b(\d{4})\b", text[dm.end():]):
        y = int(year.group(1))
    elif season := re.search(r"\b(\d{4})(–\d{2})?\b", title):
        y = int(season.group(1)) + (bool(season.group(2)) and month <= 6)
    else:
        return None
    try:
        return date(y, month, int(dm.group(1))).isoformat()
    except ValueError:
        return None


def _team(name: str) -> str:
    name = re.sub(r"^\([HA]\) |\s*\([HA]\)$", "", name.strip())  # IPL boxes mark the home side "(H)"
    return ALIASES.get(name, name)


@cache
def _by_day() -> dict[str, list[dict]]:
    out = defaultdict(list)
    for m in json.loads(INDEX.read_text()):
        out[m["dates"][0]].append(m)
    return out


def find_match(when: str, team_a: str, team_b: str, title: str = "") -> dict | None:
    day = first_day(when, title)
    teams = {_team(team_a), _team(team_b)}
    for m in _by_day().get(day, []) if day else []:
        if teams & {_team(t) for t in m["teams"]}:
            return m
    return None


def main() -> None:
    found, total = defaultdict(int), defaultdict(int)
    for e in json.loads(MANIFEST.read_text()):
        if not e["include"]:
            continue
        for line in (TEXT / f"{e['doc_id']}.txt").read_text().splitlines():
            if m := re.match(r"^Match: (.*) \| (.*) v (.*)$", line):
                total[e["tier"]] += 1
                found[e["tier"]] += find_match(*m.groups(), title=e["title"]) is not None
    print(f"{'tier':18} {'match boxes':>12} {'in Cricsheet':>13}")
    for tier in total:
        print(f"{tier:18} {total[tier]:>12} {found[tier]:>8} ({found[tier] / total[tier]:.0%})")
    print(f"{'total':18} {sum(total.values()):>12} {sum(found.values()):>8} "
          f"({sum(found.values()) / sum(total.values()):.0%})")


if __name__ == "__main__":
    main()
