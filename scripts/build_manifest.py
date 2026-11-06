"""Resolve the candidate Wikipedia articles to pinned revisions and write data/cricket/manifest.json.

    uv run scripts/build_manifest.py            # resolve every candidate, keep existing pins
    uv run scripts/build_manifest.py --repin    # move every article to its current revision

Each entry: doc_id, title (after redirects), tier, format, revid, timestamp, include. The revid is the edition the
corpus and the golden set are built from: Wikipedia pages change daily, and a golden evidence quote must stay
verbatim. Adding an article later never moves the pins of the ones already in. `include` is set by measure.py
(the corpus is cut to ~5k chunks); a candidate left out stays listed so it can be switched on later.
"""

import json
import re
import sys
from pathlib import Path

from wiki import api

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "cricket" / "manifest.json"

BGT = ["Australian cricket team in India in 2010–11", "Indian cricket team in Australia in 2011–12",
       "2012–13 Border–Gavaskar Trophy", "2014–15 Border–Gavaskar Trophy", "2016–17 Border–Gavaskar Trophy",
       "Indian cricket team in Australia in 2018–19", "Indian cricket team in Australia in 2020–21",
       "Australian cricket team in India in 2022–23", "2024–25 Border–Gavaskar Trophy"]
ASHES = ["2010–11 Ashes series", "2013 Ashes series", "2013–14 Ashes series", "2015 Ashes series",
         "2017–18 Ashes series", "2019 Ashes series", "2021–22 Ashes series", "2023 Ashes series",
         "2025–26 Ashes series"]
IND_ENG = ["Indian cricket team in England in 2011", "English cricket team in India in 2012–13",
           "Indian cricket team in England in 2014", "English cricket team in India in 2016–17",
           "Indian cricket team in England in 2018", "English cricket team in India in 2020–21",
           "Indian cricket team in England in 2021", "English cricket team in India in 2023–24",
           "Indian cricket team in England in 2025"]
WTC = ["2021 World Test Championship final", "2023 World Test Championship final",
       "2025 World Test Championship final"]
ODI_WC = ["2011 Cricket World Cup", "2011 Cricket World Cup Group A", "2011 Cricket World Cup Group B",
          "2011 Cricket World Cup knockout stage", "2011 Cricket World Cup final",
          "2015 Cricket World Cup", "2015 Cricket World Cup Pool A", "2015 Cricket World Cup Pool B",
          "2015 Cricket World Cup knockout stage", "2015 Cricket World Cup final",
          "2019 Cricket World Cup", "2019 Cricket World Cup group stage", "2019 Cricket World Cup knockout stage",
          "2019 Cricket World Cup final",
          "2023 Cricket World Cup", "2023 Cricket World Cup group stage", "2023 Cricket World Cup knockout stage",
          "2023 Cricket World Cup final"]
CT = ["2013 ICC Champions Trophy", "2013 ICC Champions Trophy final", "2017 ICC Champions Trophy",
      "2017 ICC Champions Trophy final", "2025 ICC Champions Trophy", "2025 ICC Champions Trophy final"]
T20_WC = ["2007 ICC World Twenty20", "2007 ICC World Twenty20 final", "2009 ICC World Twenty20",
          "2009 ICC World Twenty20 final", "2010 ICC World Twenty20", "2010 ICC World Twenty20 final",
          "2012 ICC World Twenty20", "2012 ICC World Twenty20 final", "2014 ICC World Twenty20",
          "2014 ICC World Twenty20 final", "2016 ICC World Twenty20", "2016 ICC World Twenty20 final",
          "2021 ICC Men's T20 World Cup", "2021 ICC Men's T20 World Cup final", "2022 ICC Men's T20 World Cup",
          "2022 ICC Men's T20 World Cup final", "2024 ICC Men's T20 World Cup",
          "2024 ICC Men's T20 World Cup final"]
IPL = [f"{y} Indian Premier League" for y in range(2008, 2026)] + \
      [f"{y} Indian Premier League final" for y in range(2008, 2026)]
PLAYERS = ["Virat Kohli", "Steve Smith (cricketer)", "Joe Root", "James Anderson (cricketer)", "Nathan Lyon",
           "Jasprit Bumrah", "Rohit Sharma", "Pat Cummins", "Ben Stokes", "David Warner (cricketer)",
           "Cheteshwar Pujara", "Ravichandran Ashwin", "Ravindra Jadeja", "Mitchell Starc", "Alastair Cook",
           "Stuart Broad", "MS Dhoni", "Michael Clarke (cricketer)", "Rishabh Pant", "Shubman Gill",
           "Travis Head", "Marnus Labuschagne", "Josh Hazlewood", "Kevin Pietersen", "Mitchell Johnson (cricketer)",
           "Ajinkya Rahane", "Kane Williamson", "Glenn Maxwell", "Jos Buttler", "Suryakumar Yadav"]
GROUNDS = ["Adelaide Oval", "Melbourne Cricket Ground", "Sydney Cricket Ground", "The Gabba", "Perth Stadium",
           "Lord's", "The Oval", "Edgbaston Cricket Ground", "Old Trafford Cricket Ground",
           "Headingley Cricket Ground", "Trent Bridge", "Eden Gardens", "Wankhede Stadium",
           "M. A. Chidambaram Stadium", "Narendra Modi Stadium"]

# tier -> (format, titles). The tier order is the priority when the corpus is cut to the chunk budget.
CANDIDATES = {
    "bgt": ("test", BGT), "ashes": ("test", ASHES), "ind_eng": ("test", IND_ENG), "wtc": ("test", WTC),
    "odi_wc": ("odi", ODI_WC), "champions_trophy": ("odi", CT), "t20_wc": ("t20i", T20_WC), "ipl": ("t20", IPL),
    "players": ("person", PLAYERS), "grounds": ("ground", GROUNDS),
}


def doc_id(title: str) -> str:  # "2014–15 Border–Gavaskar Trophy" -> "2014-15_border-gavaskar_trophy"
    t = title.lower().replace("–", "-").replace("'", "")
    return re.sub(r"[^a-z0-9-]+", "_", t).strip("_")


def resolve(titles: list[str]) -> dict[str, dict]:
    """requested title -> {title, revid, timestamp} (None if the page doesn't exist)."""
    out = {}
    for i in range(0, len(titles), 50):  # the API takes 50 titles per call
        batch = titles[i:i + 50]
        data = api(action="query", titles="|".join(batch), redirects=1, prop="revisions", rvprop="ids|timestamp")
        q = data["query"]
        alias = {n["from"]: n["to"] for n in q.get("normalized", [])}
        alias |= {n["from"]: n["to"] for n in q.get("redirects", [])}
        pages = {p["title"]: p for p in q["pages"]}
        for t in batch:
            final = t
            while final in alias:
                final = alias[final]
            p = pages.get(final)
            out[t] = None if p is None or p.get("missing") else {
                "title": p["title"], "revid": p["revisions"][0]["revid"], "timestamp": p["revisions"][0]["timestamp"]}
    return out


def main() -> None:
    old = {e["doc_id"]: e for e in json.loads(MANIFEST.read_text())} if MANIFEST.exists() else {}
    entries, missing = {}, []
    for tier, (fmt, titles) in CANDIDATES.items():
        for want, got in resolve(titles).items():
            if got is None:
                missing.append(want)
                continue
            d = doc_id(got["title"])
            if d in entries:  # two candidate titles redirect to one page
                continue
            pin = old[d] if d in old and "--repin" not in sys.argv else got
            entries[d] = {"doc_id": d, "title": got["title"], "tier": tier, "format": fmt,
                          "revid": pin["revid"], "timestamp": pin["timestamp"],
                          "include": old.get(d, {}).get("include", True)}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(list(entries.values()), indent=1, ensure_ascii=False) + "\n")
    print(f"{len(entries)} articles pinned -> {MANIFEST.relative_to(MANIFEST.parent.parent.parent)}")
    print(f"{len(missing)} titles not on Wikipedia:", *missing, sep="\n  ")


if __name__ == "__main__":
    main()
