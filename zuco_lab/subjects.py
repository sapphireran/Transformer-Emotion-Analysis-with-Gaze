"""Twelve-reader ZuCo tables and the subject-3 remapping.

``DataTransformer`` in ``utils_ZuCo.py`` drops task-1 sentences 150–249 and
399 for subject index 2 (file ``3_SR.csv``), then writes the remaining 299
rows with a compacted ``id`` of 0..298.

``get_average_sentence_level.py`` later concatenates the twelve CSVs and
groups by row index. From id 150 onward, reader 3 is contributing the
*wrong sentence* to the published averages.

This module:

* loads every ``{k}_SR.csv``
* remaps reader 3 back onto original sentence ids
* reports which rows are aligned vs contaminated
* estimates how far the published ``average_data.csv`` sits from a
  correctly aligned mean
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from . import csvio, numbers, paths
from .schema import SUBJECT_SENTENCE, SUBJECT_WORD, validate_rows

N_SUBJECTS = 12
N_SENTENCES = 400

# Original sentence ids dropped for task 1, subject index 2.
SUBJECT3_SKIPPED_ORIGINAL = frozenset(range(150, 250)) | {399}

# After compaction, file 3_SR.csv has this many rows.
SUBJECT3_COMPACT_ROWS = N_SENTENCES - len(SUBJECT3_SKIPPED_ORIGINAL)  # 299

SENTENCE_FEATURES = (
    "SentLen",
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)


def compact_to_original_subject3(compact_id: int) -> int:
    """Map compacted id 0..298 in ``3_SR.csv`` back to the original 0..399 id."""
    if compact_id < 0 or compact_id >= SUBJECT3_COMPACT_ROWS:
        raise ValueError(f"subject 3 compact id {compact_id} out of range")
    if compact_id <= 149:
        return compact_id
    return compact_id + 100  # 150 -> 250, 298 -> 398


def original_to_compact_subject3(original_id: int) -> int | None:
    if original_id in SUBJECT3_SKIPPED_ORIGINAL:
        return None
    if original_id <= 149:
        return original_id
    if 250 <= original_id <= 398:
        return original_id - 100
    raise ValueError(f"unexpected original id {original_id}")


@dataclass(frozen=True)
class SubjectSentence:
    subject: int
    compact_id: int
    original_id: int
    aligned: bool
    values: dict[str, float]


@dataclass
class SubjectBundle:
    sentences: dict[int, list[SubjectSentence]] = field(default_factory=dict)

    def readers_for(self, original_id: int, *, aligned_only: bool = True) -> list[SubjectSentence]:
        rows = self.sentences.get(original_id, [])
        if aligned_only:
            return [row for row in rows if row.aligned]
        return list(rows)


def load_subject_sentences(validate: bool = True) -> SubjectBundle:
    bundle = SubjectBundle()
    for subject in range(1, N_SUBJECTS + 1):
        rows = csvio.read_dicts(paths.subject_sentence_csv(subject))
        if validate:
            validate_rows(rows, SUBJECT_SENTENCE)
        for row in rows:
            compact_id = int(float(row["id"]))
            if subject == 3:
                original_id = compact_to_original_subject3(compact_id)
                aligned = compact_id <= 149
            else:
                original_id = compact_id
                aligned = True
            values = {name: csvio.as_float(row[name]) for name in SENTENCE_FEATURES}
            item = SubjectSentence(
                subject=subject,
                compact_id=compact_id,
                original_id=original_id,
                aligned=aligned,
                values=values,
            )
            bundle.sentences.setdefault(original_id, []).append(item)
            if subject == 3 and not aligned:
                # Also record the *claimed* compact id so contamination is visible.
                claimed = SubjectSentence(
                    subject=subject,
                    compact_id=compact_id,
                    original_id=compact_id,
                    aligned=False,
                    values=values,
                )
                bundle.sentences.setdefault(compact_id, []).append(claimed)
    return bundle


def index_aligned_mean(feature: str, compact_id: int, bundle: SubjectBundle | None = None) -> float:
    """Mean that ``groupby(level=0)`` would compute: everyone at the same CSV id."""
    bundle = bundle or load_subject_sentences()
    values: list[float] = []
    for subject in range(1, N_SUBJECTS + 1):
        rows = csvio.read_dicts(paths.subject_sentence_csv(subject))
        match = next((row for row in rows if int(float(row["id"])) == compact_id), None)
        if match is not None:
            values.append(csvio.as_float(match[feature]))
    return numbers.mean(values)


def correctly_aligned_mean(
    feature: str,
    original_id: int,
    bundle: SubjectBundle | None = None,
) -> float | None:
    bundle = bundle or load_subject_sentences()
    rows = [item for item in bundle.readers_for(original_id, aligned_only=True)]
    # For ids 250-398, aligned_only drops the false compact-id claim; we still
    # want remapped reader 3. Reload-style: take subject 3 only when original matches.
    values = []
    seen: set[int] = set()
    for item in bundle.sentences.get(original_id, []):
        if item.subject == 3 and item.original_id != original_id:
            continue
        if item.subject in seen:
            continue
        if item.subject == 3 and item.compact_id != original_to_compact_subject3(original_id):
            continue
        seen.add(item.subject)
        values.append(item.values[feature])
    if not values:
        return None
    return numbers.mean(values)


@dataclass(frozen=True)
class ContaminationRow:
    sentence_id: int
    feature: str
    index_mean: float
    aligned_mean: float
    delta: float
    n_index: int
    n_aligned: int


def contamination_table(
    feature: str = "nFixations",
    ids: Iterable[int] | None = None,
) -> list[ContaminationRow]:
    """Compare index-average vs remapped-average for one feature."""
    ids = range(N_SENTENCES) if ids is None else ids
    out: list[ContaminationRow] = []
    # Load raw maps once.
    raw: dict[int, dict[int, dict[str, float]]] = {}
    for subject in range(1, N_SUBJECTS + 1):
        rows = csvio.read_dicts(paths.subject_sentence_csv(subject))
        raw[subject] = {int(float(row["id"])): {name: csvio.as_float(row[name]) for name in SENTENCE_FEATURES} for row in rows}

    for sid in ids:
        index_vals = [raw[s][sid][feature] for s in raw if sid in raw[s]]
        aligned_vals: list[float] = []
        for subject, table in raw.items():
            if subject != 3:
                if sid in table:
                    aligned_vals.append(table[sid][feature])
                continue
            compact = original_to_compact_subject3(sid)
            if compact is not None and compact in table:
                aligned_vals.append(table[compact][feature])
        if not index_vals or not aligned_vals:
            continue
        index_mean = numbers.mean(index_vals)
        aligned_mean = numbers.mean(aligned_vals)
        out.append(
            ContaminationRow(
                sentence_id=sid,
                feature=feature,
                index_mean=index_mean,
                aligned_mean=aligned_mean,
                delta=index_mean - aligned_mean,
                n_index=len(index_vals),
                n_aligned=len(aligned_vals),
            )
        )
    return out


def alignment_regions() -> dict[str, tuple[int, int]]:
    return {
        "aligned_all_12": (0, 149),
        "contaminated_wrong_sentence": (150, 249),
        "contaminated_and_shifted": (250, 298),
        "reader3_missing_from_index": (299, 398),
        "reader3_skipped": (399, 399),
    }


def first_mismatch_id() -> int:
    """Smallest compact id where subject 3 SentLen != subject 1 SentLen."""
    one = {int(float(r["id"])): float(r["SentLen"]) for r in csvio.read_dicts(paths.subject_sentence_csv(1))}
    three = {int(float(r["id"])): float(r["SentLen"]) for r in csvio.read_dicts(paths.subject_sentence_csv(3))}
    for compact_id in sorted(three):
        if one.get(compact_id) != three[compact_id]:
            return compact_id
    raise RuntimeError("subject 3 SentLen matches subject 1 on every shared id")


def remapped_sentlen_matches() -> bool:
    """True if remapped subject 3 SentLen equals subject 1 on 0-149 and 250-398."""
    one = {int(float(r["id"])): float(r["SentLen"]) for r in csvio.read_dicts(paths.subject_sentence_csv(1))}
    three = {int(float(r["id"])): float(r["SentLen"]) for r in csvio.read_dicts(paths.subject_sentence_csv(3))}
    for compact_id, sent_len in three.items():
        original = compact_to_original_subject3(compact_id)
        if one[original] != sent_len:
            return False
    return True


@dataclass(frozen=True)
class ReaderAgreement:
    feature: str
    id_from: int
    id_to: int
    mean_cv: float
    mean_pairwise_r: float
    min_pairwise_r: float
    max_pairwise_r: float
    n_sentences: int
    n_pairs: int


def reader_agreement(
    feature: str = "nFixations",
    id_from: int = 0,
    id_to: int = 149,
) -> ReaderAgreement:
    """Inter-reader CV and pairwise Pearson on a clean id window."""
    raw: dict[int, dict[int, float]] = {}
    for subject in range(1, N_SUBJECTS + 1):
        rows = csvio.read_dicts(paths.subject_sentence_csv(subject))
        raw[subject] = {int(float(row["id"])): csvio.as_float(row[feature]) for row in rows}

    cvs: list[float] = []
    for sid in range(id_from, id_to + 1):
        values = [raw[s][sid] for s in raw if sid in raw[s]]
        if len(values) >= 2:
            cvs.append(numbers.coeff_of_variation(values))

    pairs: list[float] = []
    subjects = list(range(1, N_SUBJECTS + 1))
    for i, a in enumerate(subjects):
        for b in subjects[i + 1 :]:
            xs, ys = [], []
            for sid in range(id_from, id_to + 1):
                if sid in raw[a] and sid in raw[b]:
                    xs.append(raw[a][sid])
                    ys.append(raw[b][sid])
            if len(xs) >= 2:
                pairs.append(numbers.pearson(xs, ys))

    return ReaderAgreement(
        feature=feature,
        id_from=id_from,
        id_to=id_to,
        mean_cv=numbers.mean(cvs) if cvs else 0.0,
        mean_pairwise_r=numbers.mean(pairs) if pairs else 0.0,
        min_pairwise_r=min(pairs) if pairs else 0.0,
        max_pairwise_r=max(pairs) if pairs else 0.0,
        n_sentences=id_to - id_from + 1,
        n_pairs=len(pairs),
    )


def load_subject_words(subject: int, validate: bool = True) -> list[dict[str, str]]:
    rows = csvio.read_dicts(paths.subject_word_csv(subject))
    if validate:
        validate_rows(rows, SUBJECT_WORD)
    return rows


def remap_word_sent_id(subject: int, sent_id: str) -> str:
    """Rewrite subject-3 Sent_ID values such as ``150_NR`` → ``250_NR``."""
    if subject != 3:
        return sent_id
    number_s, suffix = sent_id.split("_", 1)
    original = compact_to_original_subject3(int(number_s))
    return f"{original}_{suffix}"
