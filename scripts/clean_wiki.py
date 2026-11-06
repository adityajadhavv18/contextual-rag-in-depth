"""Wikipedia HTML -> clean text that keeps the article's structure: data/cricket/text/<doc_id>.txt + articles.json.

    uv run scripts/clean_wiki.py

Output, one file per article:
    # Indian cricket team in Australia in 2014–15
    ## Test series
    ### 1st Test
    Match: 9–13 December 2014 | Australia v India
    Australia: 7/517d (120 overs); Steve Smith 162* (231); Mohammed Shami 2/120 (24 overs)
    ...
    Result: Australia won by 48 runs
    Australia won the toss and chose to bat on a wicket ...

Kept: headings (as #, ##, ###, #### lines, so the chunker can follow sections), paragraphs, lists, tables (one row
per line, cells split by " | "), the infobox (one "label: value" per line), and match boxes. A match box (date, the
two teams' scores, result, venue, umpires, player of the match) is turned into labelled lines, because its layout
is the only thing that says whose score is whose. Dropped: references and citation marks, navigation boxes,
images, hatnotes, the sections Notes / References / External links / See also / Further reading, and sections
that aren't cricket (a ground's rugby and football history, concerts, broadcasting, a player's personal life).
Every block is one line: paragraphs never wrap, so evidence quotes from the golden set stay on one line.
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from build_manifest import MANIFEST

RAW, TEXT = MANIFEST.parent / "raw", MANIFEST.parent / "text"
ARTICLES = MANIFEST.parent / "articles.json"

DROP_SECTIONS = {"notes", "references", "external links", "see also", "further reading", "bibliography",
                 "sources", "citations", "footnotes", "notes and references", "references and notes", "gallery",
                 # not cricket: the other sports a ground hosts, and a player's life away from the game
                 "rugby league", "rugby union", "australian rules football", "australian rules football timeline",
                 "mcg and the vfl/afl", "vfl/afl records", "sydney swans", "soccer", "football", "baseball",
                 "olympic games", "other sports", "concerts", "musical venue", "broadcasting", "personal life",
                 "early and personal life", "philanthropy", "public image and in media", "outside cricket",
                 "in popular culture", "endorsements", "in the media", "media"}
DROP_SELECTORS = ["style", "link", "script", "sup.reference", "sup.noprint", ".mw-ref", ".navbox", ".navbox-styles",
                  ".hatnote", ".shortdescription", ".mw-empty-elt", "figure", ".thumb", ".metadata", ".ambox",
                  ".reflist", ".references", ".mw-references-wrap", ".noprint", ".sidebar", ".side-box", "img",
                  ".gallery", ".toc", ".portalbox", ".sistersitebox", ".mw-cite-backlink", ".plainlinks.hlist"]


def inline(el: Tag | NavigableString) -> str:
    """Text of an element on one line; <br> becomes '; '."""
    if isinstance(el, NavigableString):
        return _space(str(el))
    for br in el.find_all("br"):
        br.replace_with(" ")  # placeholder, so '; ' is only added between non-empty parts
    parts = [_space(p) for p in el.get_text("").split(" ")]
    return "; ".join(p for p in parts if p)


def _space(s: str) -> str:
    s = re.sub(r"\s+", " ", s.replace("\xa0", " ").replace("​", "")).strip()
    return re.sub(r" ([,.;:)])", r"\1", s).replace("( ", "(")


def is_match_box(el: Tag) -> bool:
    tables = el.find_all("table", recursive=False)
    return len(tables) >= 3 and ("Umpires" in el.get_text() or "Player of the match" in el.get_text())


def match_box(el: Tag) -> list[str]:
    date_t, teams_t, result_t = el.find_all("table", recursive=False)[:3]
    date = inline(date_t).replace("Scorecard", "").strip("; ")
    rows = teams_t.find_all("tr")
    head = [c.find("b") for c in rows[0].find_all("td")]
    names = [inline(b) if b else "" for b in head]
    a, b = names[0], names[-1]
    lines = [f"Match: {date} | {a} v {b}"]
    for r in rows:
        cells = r.find_all("td", recursive=False)
        if len(cells) < 3:
            continue
        for name, cell in ((a, cells[0]), (b, cells[-1])):
            if r is rows[0]:  # the team cell may hold the score after the name (one-day box)
                for bold in cell.find_all("b"):
                    bold.decompose()
            text = inline(cell)
            if text:
                lines.append(f"{name}: {text}")
    for i, part in enumerate(inline(result_t).split("; ")):
        label = "Result" if i == 0 else ("Venue" if i == 1 and ":" not in part else "")
        lines.append(f"{label}: {part}" if label else part)
    for li in el.find_all("li"):
        lines.append(f"- {inline(li)}")
    return lines


def table(el: Tag) -> list[str]:
    if "infobox" in (el.get("class") or []):
        out = []
        for tr in el.find_all("tr"):
            label, data = tr.find("th"), tr.find("td")
            if label and data:
                out.append(f"{inline(label)}: {inline(data)}")
            elif label or data:
                out.append(inline(label or data))
        return [x for x in out if x]
    out = []
    for tr in el.find_all("tr"):
        cells = [inline(c) for c in tr.find_all(["th", "td"], recursive=False)]
        if any(cells):
            out.append(" | ".join(cells))
    return out


def list_items(el: Tag, depth: int = 0) -> list[str]:
    out = []
    for li in el.find_all("li", recursive=False):
        nested = [n.extract() for n in li.find_all(["ul", "ol"], recursive=False)]
        if text := inline(li):
            out.append(f"{'  ' * depth}- {text}")
        for n in nested:
            out += list_items(n, depth + 1)
    return out


def blocks(el: Tag) -> list[str]:
    """Lines for one block-level element (recursing into plain wrapper divs)."""
    if el.name == "p" or el.name == "blockquote":
        return [t] if (t := inline(el)) else []
    if el.name in ("ul", "ol"):
        return list_items(el)
    if el.name == "dl":
        return [t for c in el.find_all(["dt", "dd"], recursive=False) if (t := inline(c))]
    if el.name == "table":
        return table(el)
    if el.name == "div":
        if is_match_box(el):
            return match_box(el)
        out = []
        for c in el.children:
            if isinstance(c, Tag):
                out += blocks(c)
            elif t := _space(str(c)):
                out.append(t)
        return out
    return []


def clean(html: str, title: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for sel in DROP_SELECTORS:
        for x in soup.select(sel):
            x.decompose()
    root = soup.select_one("div.mw-parser-output")
    lines, drop_level = [f"# {title}"], None
    for el in root.find_all(recursive=False):
        h = el.find(re.compile("^h[2-6]$")) if "mw-heading" in (el.get("class") or []) else None
        if h:
            level, name = int(h.name[1]), inline(h)
            if drop_level is not None and level > drop_level:
                continue  # a subsection of a dropped section
            drop_level = level if name.lower() in DROP_SECTIONS else None
            if drop_level is None:
                lines.append(f"{'#' * level} {name}")
            continue
        if drop_level is None:
            lines += blocks(el)
    # a heading with nothing under it (all of its content was dropped) is noise
    keep = [l for i, l in enumerate(lines)
            if not (l.startswith("#") and (i + 1 == len(lines) or
                                           lines[i + 1].startswith("#") and _depth(lines[i + 1]) <= _depth(l)))]
    return "\n".join(keep) + "\n"


def _depth(heading: str) -> int:
    return len(heading) - len(heading.lstrip("#"))


def main() -> None:
    TEXT.mkdir(exist_ok=True)
    meta = []
    for e in json.loads(MANIFEST.read_text()):
        text = clean((RAW / f"{e['doc_id']}.html").read_text(), e["title"])
        (TEXT / f"{e['doc_id']}.txt").write_text(text)
        meta.append({k: e[k] for k in ("doc_id", "title", "tier", "format", "revid")} |
                    {"words": len(text.split()), "sections": sum(l.startswith("#") for l in text.splitlines())})
    ARTICLES.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(meta)} articles, {sum(m['words'] for m in meta):,} words -> {TEXT.relative_to(MANIFEST.parent.parent.parent)}/")


if __name__ == "__main__":
    main()
