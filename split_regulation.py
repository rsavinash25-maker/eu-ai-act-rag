"""Split EU regulations (e.g. the AI Act) into one file per recital, article and annex.

Reads every file in the input folder. PDFs that look like an EU regulation (Official Journal
or EUR-Lex consolidated layout) are split into structural units; everything else (guidelines,
FAQs, codes of practice) is copied across unchanged, so the output folder is the full corpus.

Each unit becomes a Markdown file that starts with a context header, e.g.

    # Regulation (EU) 2024/1689 — Article 6: Classification rules for high-risk AI systems
    Chapter III: HIGH-RISK AI SYSTEMS › Section 1: CLASSIFICATION OF AI SYSTEMS AS HIGH-RISK

ingest.py repeats this header on every chunk of the unit, so even the second half of a long
article still says which article it belongs to.

Usage:
    python split_regulation.py --dry-run                 # check what it finds first
    python split_regulation.py                           # data/docs -> data/docs_structured
    python split_regulation.py --no-recitals             # articles and annexes only
"""
import argparse
import re
import shutil
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path

# Page furniture from the Official Journal PDF layout
NOISE = re.compile(
    r"^(EN|OJ L,?\s.*|ELI:\s.*|\d{1,3}/\d{1,3}|L \d+/\d+.*|Official Journal of the European Union"
    # EUR-Lex consolidated layout: "02024R1689 — EN — 27.07.2026 — 001.001 — 12"
    r"|\d{5}[A-Z]\d{4}\s*[—–-]\s*EN\s*[—–-].*"
    r"|This text is meant purely as a documentation tool.*)$"
)
FOOTNOTE = re.compile(r"^\(\d+\)\s.*\bOJ [LC]\b")
# EUR-Lex consolidated-version markers: ▼B (base text), ▼M1 (amendment 1), ►M1 … ◄
MARKER_LINE = re.compile(r"^▼[A-Z]\d*$")
AMENDMENT = re.compile(r"▼M\d+|►M\d+")
INLINE_MARKERS = re.compile(r"►[A-Z]\d+\s*|\s*◄")

RECITAL = re.compile(r"^\((\d{1,3})\)\s+(\S.*)$")
CHAPTER = re.compile(r"^CHAPTER ([IVXLC]+)$")
SECTION = re.compile(r"^SECTION (\d+)$")
ARTICLE = re.compile(r"^Article (\d+)([a-z]?)$")
ANNEX = re.compile(r"^ANNEX ([IVXLC]+)$")
PARA_START = re.compile(r"^(\d+[a-z]?\.\s|\([a-z]{1,4}\)\s|\(\d+[a-z]?\)\s|[—–-]\s)")
TRAILING_JOINER = re.compile(r"\b(and|or|of|for|in|to|the|on|with|by|from|as)$")  # a title can't end like this
DEFINITION = re.compile(r"^\((\d+[a-z]?)\)\s")  # a numbered point in a Definitions article


def roman(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for i, ch in enumerate(s):
        v = vals[ch]
        total += -v if i + 1 < len(s) and vals[s[i + 1]] > v else v
    return total


@dataclass
class Unit:
    kind: str  # recital | article | annex | definition
    num: str
    title: str = ""
    chapter: str = ""
    section: str = ""
    lines: list = field(default_factory=list)
    amended: bool = False
    parent: str = ""  # for definitions: the article they belong to
    flags: list = field(default_factory=list)  # per line: inside an amendment marker?

    def add(self, line: str, amended: bool):
        self.lines.append(line)
        self.flags.append(amended)
        self.amended |= amended

    def paragraphs(self) -> list[list]:
        """[[text, amended], ...] — lines joined into paragraphs, with per-paragraph amendment flags."""
        paras = []
        for line, flag in zip_longest(self.lines, self.flags, fillvalue=False):
            if not paras or (PARA_START.match(line) and not (paras[-1][0].endswith("-") and not paras[-1][0].endswith(" -"))):
                paras.append([line, flag])
                continue
            text = paras[-1][0]
            if text.endswith("-") and not text.endswith(" -"):
                paras[-1][0] = text + line  # re-join words hyphenated across a line break
            else:
                paras[-1][0] = text + " " + line
            paras[-1][1] |= flag
        return paras

    def body(self) -> str:
        return "\n\n".join(t for t, _ in self.paragraphs()).strip()


def looks_like_title(line: str) -> bool:
    return bool(line) and line[0].isupper() and len(line) < 200 and not line.endswith((".", ";", ","))


def clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or NOISE.match(line) or FOOTNOTE.match(line):
            continue
        lines.append(line)
    return lines


def parse_regulation(text: str) -> tuple[str, list[Unit]]:
    m = re.search(r"REGULATION \(EU\) (\d{4}/\d+)", text[:5000], re.IGNORECASE)
    doc_title = f"Regulation (EU) {m.group(1)}" if m else "Regulation"

    lines = clean_lines(text)
    units: list[Unit] = []
    cur: Unit | None = None
    state = "preamble"
    last_recital, last_article, last_annex = 0, (0, ""), 0
    chapter = section = ""
    in_amendment = False  # between a ▼M marker and the next ▼B marker
    i = 0

    def next_line(j):
        while j < len(lines) and MARKER_LINE.match(lines[j]):
            j += 1
        if j >= len(lines):
            return "", j
        text = INLINE_MARKERS.sub(" ", lines[j]).strip()
        # Titles that wrap onto following lines: "high-" / "risk AI systems", or
        # "…, in particular SMEs," / "including start-ups". Join up to 3 continuation lines.
        for _ in range(3):
            if j + 1 >= len(lines):
                break
            nxt = INLINE_MARKERS.sub(" ", lines[j + 1]).strip()
            if not nxt or MARKER_LINE.match(nxt) or PARA_START.match(nxt) or ARTICLE.match(nxt):
                break
            if text.endswith("-") and not text.endswith(" -"):
                text += nxt
            elif text.endswith(",") or TRAILING_JOINER.search(text) or nxt[0].islower():
                text += " " + nxt
            else:
                break
            j += 1
        return text, j

    def new_unit(u: Unit) -> Unit:
        u.amended = in_amendment  # a whole new article/annex inserted by an amendment
        units.append(u)
        return u

    while i < len(lines):
        line = lines[i]

        if MARKER_LINE.match(line):
            in_amendment = bool(AMENDMENT.search(line))
            i += 1
            continue
        amended_line = in_amendment or bool(AMENDMENT.search(line))
        line = re.sub(r"\s{2,}", " ", INLINE_MARKERS.sub(" ", line)).strip()

        if state == "preamble":
            if re.search(r"HAVE ADOPTED THIS REGULATION", line, re.IGNORECASE):
                state, cur = "articles", None
                i += 1
                continue
            if ARTICLE.match(line) and ARTICLE.match(line).group(1) == "1" and looks_like_title(next_line(i + 1)[0]):
                state, cur = "articles", None
                continue  # re-process this line as an article heading
            if CHAPTER.match(line) and CHAPTER.match(line).group(1) == "I" and last_recital == 0:
                state, cur = "articles", None
                continue
            r = RECITAL.match(line)
            if r and int(r.group(1)) == last_recital + 1:
                last_recital += 1
                cur = new_unit(Unit("recital", str(last_recital)))
                cur.add(r.group(2), amended_line)
            elif cur:
                cur.add(line, amended_line)
            i += 1
            continue

        if re.match(r"^Done at \w+", line):
            state, cur = "signature", None
            i += 1
            continue

        c = CHAPTER.match(line)
        if c and state in ("articles", "signature"):
            title, j = next_line(i + 1)
            chapter, section = f"Chapter {c.group(1)}: {title}", ""
            i = j + 1
            continue
        s = SECTION.match(line)
        if s and state == "articles":
            title, j = next_line(i + 1)
            section = f"Section {s.group(1)}: {title}"
            i = j + 1
            continue

        a = ARTICLE.match(line)
        if a and state in ("articles", "signature"):
            n, suffix = int(a.group(1)), a.group(2)
            title, j = next_line(i + 1)
            sequential = n == last_article[0] + 1 or (n == last_article[0] and suffix > last_article[1])
            if sequential and looks_like_title(title):
                last_article = (n, suffix)
                cur = new_unit(Unit("article", f"{n}{suffix}", title, chapter, section))
                state = "articles"
                i = j + 1
                continue

        x = ANNEX.match(line)
        if x and roman(x.group(1)) == last_annex + 1:
            last_annex += 1
            title, j = next_line(i + 1)
            cur = new_unit(Unit("annex", x.group(1), title))
            state = "annexes"
            i = j + 1
            continue

        if cur and state != "signature":
            cur.add(line, amended_line)
        i += 1

    return doc_title, units


def expand_definitions(u: Unit) -> list[Unit] | None:
    """Split a 'Definitions' article into one unit per numbered definition.

    A definitions article is dozens of unrelated terms in one block; chunked as a whole, a
    question about one term retrieves a chunk full of others and the model can pick the wrong one."""
    if u.kind != "article" or u.title.strip().lower() != "definitions":
        return None
    defs = []
    for para, flag in u.paragraphs():
        m = DEFINITION.match(para)
        if m:
            defs.append([m.group(1), para, flag])
        elif defs:
            defs[-1][1] += "\n\n" + para  # sub-points (a), (b)… belong to the current definition
            defs[-1][2] |= flag
    if len(defs) < 5:
        return None
    out = []
    for num, text, flag in defs:
        term = re.search(r"[‘'“\"]([^’'”\"]{2,100})[’'”\"]", text)
        d = Unit("definition", num, term.group(1) if term else "", u.chapter, u.section, parent=u.num)
        d.add(text, flag)
        out.append(d)
    return out


def render(doc_title: str, u: Unit) -> str:
    if u.kind == "recital":
        head = f"# {doc_title} — Recital ({u.num})"
    elif u.kind == "definition":
        head = f"# {doc_title} — Article {u.parent}({u.num}): Definition" + (f" of ‘{u.title}’" if u.title else "")
    elif u.kind == "article":
        head = f"# {doc_title} — Article {u.num}: {u.title}"
    else:
        head = f"# {doc_title} — Annex {u.num}: {u.title}"
    context = " › ".join(p for p in (u.chapter, u.section) if p)
    header = "\n".join(p for p in (head, context, "(Amended text)" if u.amended else "") if p)
    return f"{header}\n\n{u.body()}\n"


def filename(stem: str, u: Unit) -> str:
    if u.kind == "annex":
        return f"{stem}__annex_{roman(u.num):02d}.md"
    if u.kind == "definition":
        d = re.match(r"(\d+)([a-z]?)", u.num)
        return f"{stem}__article_{int(u.parent):03d}_def_{int(d.group(1)):03d}{d.group(2)}.md"
    num = re.match(r"(\d+)([a-z]?)", u.num)
    return f"{stem}__{u.kind}_{int(num.group(1)):03d}{num.group(2)}.md"


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/docs")
    ap.add_argument("--output", default="data/docs_structured")
    ap.add_argument("--no-recitals", action="store_true")
    ap.add_argument("--min-articles", type=int, default=5, help="below this, treat the PDF as a normal document")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src, out = Path(a.input), Path(a.output)
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"})
    if not files:
        raise SystemExit(f"No documents found in {src}")
    if not a.dry_run:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

    regulations = []  # (path, doc_title, units)
    for path in files:
        units, doc_title = [], ""
        if path.suffix.lower() == ".pdf":
            doc_title, units = parse_regulation(pdf_text(path))
        if sum(u.kind == "article" for u in units) < a.min_articles:
            print(f"COPY   {path.name}")
            if not a.dry_run:
                shutil.copy2(path, out / path.name)
            continue
        regulations.append((path, doc_title, units))

    # Several versions of the same regulation (e.g. original + consolidated): take articles and
    # annexes from the consolidated one only, so outdated text never competes in retrieval.
    by_title = {}
    for item in regulations:
        by_title.setdefault(item[1], []).append(item)

    for doc_title, items in by_title.items():
        def is_consolidated(item):
            return "consolidated" in item[0].name.lower() or any(u.amended for u in item[2])

        main_item = next((it for it in items if is_consolidated(it)), items[-1])
        recital_item = max(items, key=lambda it: sum(u.kind == "recital" for u in it[2]))
        if len(items) > 1:
            print(f"NOTE   {len(items)} versions of {doc_title}: articles/annexes from {main_item[0].name}, "
                  f"recitals from {recital_item[0].name}; the rest is skipped")

        units = [u for u in main_item[2] if u.kind != "recital"]
        if not a.no_recitals:
            units = [u for u in recital_item[2] if u.kind == "recital"] + units
        expanded = []
        for u in units:
            expanded.extend(expand_definitions(u) or [u])
        units = expanded

        counts = {k: sum(u.kind == k for u in units) for k in ("recital", "article", "definition", "annex")}
        amended = sum(u.amended for u in units)
        print(f"SPLIT  {main_item[0].name} -> {doc_title}: {counts['recital']} recitals, "
              f"{counts['article']} articles, {counts['definition']} definitions, {counts['annex']} annexes"
              + (f", {amended} marked as amended" if amended else ""))

        nums = {int(re.match(r"\d+", u.num if u.kind == "article" else u.parent).group())
                for u in units if u.kind in ("article", "definition")}
        gaps = sorted(set(range(1, max(nums) + 1)) - nums)
        if gaps:
            print(f"       WARNING: missing article numbers {gaps[:15]}{'…' if len(gaps) > 15 else ''} "
                  f"(their text is probably inside the preceding article)")
        for u in units:
            words = len(u.body().split())
            if words > 6000:
                label = f"Article {u.num}" if u.kind == "article" else f"{u.kind} {u.num}"
                print(f"       WARNING: {label} has {words} words, so it probably swallowed the headings "
                      f"after it (a heading the splitter didn't recognise)")
        if a.dry_run:
            arts = [u for u in units if u.kind == "article"]
            for u in arts[:3] + arts[-2:]:
                print(f"       Article {u.num}: {u.title}  [{len(u.body().split())} words]"
                      + ("  (amended)" if u.amended else ""))
            defs = [u for u in units if u.kind == "definition"]
            for u in defs[:2] + defs[-1:]:
                print(f"       Article {u.parent}({u.num}): ‘{u.title}’")
            continue

        stem = re.sub(r"[^A-Za-z0-9]+", "_", main_item[0].stem).strip("_").lower()
        for u in units:
            (out / filename(stem, u)).write_text(render(doc_title, u), encoding="utf-8")

    if not a.dry_run:
        print(f"\nWrote {len(list(out.iterdir()))} files to {out}")


if __name__ == "__main__":
    main()
