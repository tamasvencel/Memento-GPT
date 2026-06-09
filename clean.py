"""
Clean a raw screenplay corpus for language-model training.

The raw scripts (scraped / OCR'd shooting scripts) carry a lot of production and
pagination noise that is NOT screenplay text: revision footers, page numbers,
(CONTINUED) markers, scene-number gutters, revision asterisks, OCR glitches, etc.
Left in, this boilerplate is wildly over-represented and the model learns to emit
it. This pass strips the noise but KEEPS the real signal: scene headings
(INT./EXT.), character cues, dialogue, and action description.

Cleans  input.txt  in place (overwrites it with the cleaned text).
The cleaning is idempotent, so re-running on an already-clean file is harmless.
"""

import re

SRC = "input.txt"
DST = "input.txt"   # overwrite the source in place

# ---------------------------------------------------------------------------
# 1. Whole-line patterns: if a line matches any of these, DROP the entire line.
# ---------------------------------------------------------------------------
DROP_LINE = [
    # Revision / draft footers, e.g. "MEMENTO Pink Revisions - 9/7/99".
    # Matches any line that mentions a revision/draft AND carries a date.
    re.compile(r"\b(revision|revisions|draft|shooting script)\b.*\d{1,2}/\d{1,2}/\d{2,4}", re.I),
    # A date-only footer line, e.g. "9/7/99" sitting alone.
    re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$"),
    # Standalone page numbers: "33.", "3A.", "10." (a number, optional letter, a dot).
    re.compile(r"^\s*\d{1,4}[A-Za-z]?\.\s*$"),
    # Bare page numbers with no dot sitting alone on a line.
    re.compile(r"^\s*\d{1,4}[A-Za-z]?\s*$"),
    # (CONTINUED), CONTINUED:, CONTINUED: (2), and scene-numbered gutter variants
    # like "4   CONTINUED: (2)   4"  -- pagination, not script. Leading/trailing
    # scene numbers are optional so the gutter-wrapped versions are caught too.
    re.compile(r"^\s*\d{0,4}[A-Za-z]?\s*\(?\s*continued\s*\)?\s*:?\s*(\(\d+\))?\s*\d{0,4}[A-Za-z]?\s*$", re.I),
    # (MORE)  -- dialogue-spills-to-next-page marker.
    re.compile(r"^\s*\(?\s*more\s*\)?\s*$", re.I),
]

# ---------------------------------------------------------------------------
# 2. In-line substitutions: applied to every surviving line to scrub markers
#    while keeping the line's real content.
# ---------------------------------------------------------------------------
def scrub(line: str) -> str:
    # Remove slug markers like "<>" and "##BLACK AND WHITE SEQUENCE##".
    line = line.replace("<>", "")
    line = re.sub(r"#+[^#]*#+", "", line)
    # Fix the most common OCR glitch: "(V.0.)" (zero) -> "(V.O.)" (letter O).
    line = line.replace("V.0.", "V.O.")
    # Strip trailing revision asterisks and surrounding whitespace ("...text   *").
    line = re.sub(r"[ \t]*\*+[ \t]*$", "", line)
    # Tabs -> spaces, then trim trailing whitespace.
    line = line.replace("\t", " ").rstrip()
    return line

# Scene-heading gutter: "  12   INT. CAR - DAY      12"  ->  "INT. CAR - DAY".
# Strips the leading scene number and the trailing scene number on slug lines only.
SLUG = re.compile(r"^\s*\d{1,4}[A-Za-z]?\s+((?:INT|EXT|INT\.?/EXT|EXT\.?/INT|I/E)\b.*)$", re.I)

def fix_slug(line: str) -> str:
    m = SLUG.match(line)
    if m:
        line = m.group(1)
        # Remove a trailing standalone scene number (e.g. "... - DAY   12").
        line = re.sub(r"\s+\d{1,4}[A-Za-z]?\s*$", "", line)
    return line


def clean(text: str) -> str:
    out_lines = []
    for raw in text.splitlines():
        if any(p.search(raw) for p in DROP_LINE):
            continue
        line = fix_slug(scrub(raw))
        if any(p.search(line) for p in DROP_LINE):   # re-check after scrubbing
            continue
        out_lines.append(line)

    text = "\n".join(out_lines)
    # Collapse 3+ blank lines into a single blank line (one paragraph break).
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    # Collapse runs of spaces (but not newlines) into one.
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip() + "\n"


if __name__ == "__main__":
    with open(SRC, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean(raw)

    with open(DST, "w", encoding="utf-8") as f:
        f.write(cleaned)

    before, after = len(raw), len(cleaned)
    removed = before - after
    print(f"before: {before:>8,} chars")
    print(f"after:  {after:>8,} chars  (overwrote {DST} in place)")
    print(f"removed:{removed:>8,} chars  ({removed / before:.1%} was boilerplate/whitespace)")
