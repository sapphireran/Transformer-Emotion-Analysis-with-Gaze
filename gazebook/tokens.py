"""Word-level token artifacts left by ``DataTransformer``'s punctuation strip.

``utils_ZuCo.py`` runs ``re.sub('[^\\w\\s]', '', word.content)`` and lowercases
only the first token of a sentence. Hyphens disappear, so
``murder-on-campus`` becomes ``murderoncampus``. A few MATLAB strings also
arrive already mangled (``emp11111ty`` for sentence 4's ``empty``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .csvio import read_dicts
from .paths import repo_root

# Hand-checked against ssts_ZuCo.csv + word_averages_v2.csv on this clone.
KNOWN_GLUED = (
    ("4_NR", "emp11111ty", "empty"),
    ("69_NR", "massmurdering", "mass-murdering"),
    ("80_NR", "murderoncampus", "murder-on-campus"),
    ("199_NR", "allwiseguysallthetime", "all-wise-guys-all-the-time"),
    ("22_NR", "20thcentury", "20th-century"),
    ("358_NR", "under10", "under-10"),
)

WORDLEN_MISMATCH_COUNT = 55  # Word == 'unknown' and WordLen == 0 in the average file
DIGIT_TOKEN_COUNT = 28


@dataclass(frozen=True)
class TokenHit:
    sent_id: str
    word_id: str
    word: str
    word_len: int
    n_fixations: float


def load_word_averages(root=None) -> list[dict[str, str]]:
    root = root or repo_root()
    _, rows = read_dicts(root / "ZuCo_et_csv_data/word/word_averages_v2.csv")
    return rows


def glued_hits(rows: list[dict[str, str]] | None = None) -> list[TokenHit]:
    rows = rows if rows is not None else load_word_averages()
    wanted = {token for _, token, _ in KNOWN_GLUED}
    out = []
    for row in rows:
        if row["Word"] in wanted:
            out.append(
                TokenHit(
                    sent_id=row["Sent_ID"],
                    word_id=row["Word_ID"],
                    word=row["Word"],
                    word_len=int(float(row["WordLen"])),
                    n_fixations=float(row["nFixations"]),
                )
            )
    return out


def wordlen_mismatches(rows: list[dict[str, str]] | None = None) -> list[TokenHit]:
    rows = rows if rows is not None else load_word_averages()
    hits = []
    for row in rows:
        word = row["Word"] or ""
        stored = int(float(row["WordLen"]))
        if stored != len(word):
            hits.append(
                TokenHit(
                    sent_id=row["Sent_ID"],
                    word_id=row["Word_ID"],
                    word=word,
                    word_len=stored,
                    n_fixations=float(row["nFixations"]),
                )
            )
    return hits


def digit_tokens(rows: list[dict[str, str]] | None = None) -> list[TokenHit]:
    rows = rows if rows is not None else load_word_averages()
    hits = []
    for row in rows:
        word = row["Word"] or ""
        if any(ch.isdigit() for ch in word):
            hits.append(
                TokenHit(
                    sent_id=row["Sent_ID"],
                    word_id=row["Word_ID"],
                    word=word,
                    word_len=int(float(row["WordLen"])),
                    n_fixations=float(row["nFixations"]),
                )
            )
    return hits


def unknown_tokens(rows: list[dict[str, str]] | None = None) -> list[TokenHit]:
    rows = rows if rows is not None else load_word_averages()
    hits = []
    for row in rows:
        if (row["Word"] or "").lower() in {"", "unknown"}:
            hits.append(
                TokenHit(
                    sent_id=row["Sent_ID"],
                    word_id=row["Word_ID"],
                    word=row["Word"],
                    word_len=int(float(row["WordLen"])),
                    n_fixations=float(row["nFixations"]),
                )
            )
    return hits


def sentence_words(sent_nr: str, rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    rows = rows if rows is not None else load_word_averages()
    return [row for row in rows if row["Sent_ID"] == sent_nr]
