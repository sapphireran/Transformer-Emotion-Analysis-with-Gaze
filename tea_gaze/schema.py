"""Expected columns and light validation for each personal dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from tea_gaze.features import SENTIMENT_LABELS, canonicalize_gaze_columns, resolve_fusion_columns


@dataclass(frozen=True)
class SchemaIssue:
    code: str
    message: str
    row_index: int | None = None


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    row_count: int
    issues: tuple[SchemaIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def raise_if_invalid(self) -> None:
        if self.issues:
            preview = "; ".join(issue.message for issue in self.issues[:5])
            raise ValueError(
                f"{self.dataset} failed validation ({len(self.issues)} issues). "
                f"First issues: {preview}"
            )


REQUIRED_SENTENCE_TEXT = ("sentence_id", "sentence", "sentiment_label")
REQUIRED_WORD = ("sentence_id", "word_id", "word")


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def validate_sentence_rows(
    dataset: str,
    rows: Sequence[Mapping[str, object]],
    *,
    require_fusion_features: bool = True,
    require_text: bool = True,
) -> ValidationReport:
    issues: list[SchemaIssue] = []
    if not rows:
        return ValidationReport(dataset, 0, (SchemaIssue("empty", "table has no rows"),))

    columns = list(rows[0].keys())
    if require_text:
        for column in REQUIRED_SENTENCE_TEXT:
            if column not in columns:
                issues.append(SchemaIssue("missing_column", f"missing column {column!r}"))
    if require_fusion_features:
        try:
            resolve_fusion_columns(columns)
        except KeyError as exc:
            issues.append(SchemaIssue("missing_fusion", str(exc)))

    for index, row in enumerate(rows):
        if require_text and "sentiment_label" in row:
            try:
                label = int(row["sentiment_label"])
            except (TypeError, ValueError):
                issues.append(
                    SchemaIssue("bad_label", "sentiment_label is not an integer", index)
                )
            else:
                if label not in SENTIMENT_LABELS:
                    issues.append(
                        SchemaIssue(
                            "bad_label",
                            f"sentiment_label {label} is not in {sorted(SENTIMENT_LABELS)}",
                            index,
                        )
                    )
        gaze_map = canonicalize_gaze_columns(row.keys())
        for original in gaze_map:
            parsed = _as_float(row[original])
            if parsed is None and row[original] not in (None, ""):
                issues.append(
                    SchemaIssue(
                        "bad_float",
                        f"non-numeric gaze value in {original!r}: {row[original]!r}",
                        index,
                    )
                )

    return ValidationReport(dataset, len(rows), tuple(issues))


def validate_word_rows(
    dataset: str,
    rows: Sequence[Mapping[str, object]],
) -> ValidationReport:
    issues: list[SchemaIssue] = []
    if not rows:
        return ValidationReport(dataset, 0, (SchemaIssue("empty", "table has no rows"),))
    columns = set(rows[0].keys())
    expected = {"sentence_id", "word_id", "word"}
    # ZuCo word averages use Sent_ID / Word_ID / Word instead.
    zuco_expected = {"Sent_ID", "Word_ID", "Word"}
    if not expected.issubset(columns) and not zuco_expected.issubset(columns):
        issues.append(
            SchemaIssue(
                "missing_column",
                "word table needs sentence_id/word_id/word or Sent_ID/Word_ID/Word",
            )
        )
    return ValidationReport(dataset, len(rows), tuple(issues))


def summarize_labels(rows: Iterable[Mapping[str, object]]) -> dict[int, int]:
    counts = {label: 0 for label in SENTIMENT_LABELS}
    for row in rows:
        counts[int(row["sentiment_label"])] += 1
    return counts
