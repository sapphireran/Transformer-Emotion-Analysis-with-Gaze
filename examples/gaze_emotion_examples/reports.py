"""Build a markdown dataset report from the catalogued CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .catalog import DATASETS, get_dataset
from .io import load_dataset
from .labels import label_counts, label_table_markdown
from .paths import repo_root
from .scaling import scaler_report
from .splits import audit_split, audit_to_markdown
from .stats import correlation_matrix, describe_numeric, high_correlations, markdown_table


def _heading(title: str, level: int = 2) -> str:
    return f"{'#' * level} {title}\n"


def catalog_section() -> str:
    lines = [_heading("Checked-in tables"), ""]
    lines.append("| key | file | kind | rows | gaze columns |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for spec in DATASETS:
        if not spec.path.exists():
            rows = "missing"
        else:
            rows = str(len(pd.read_csv(spec.path)))
        gaze = ", ".join(spec.gaze_columns) if spec.gaze_columns else "—"
        lines.append(
            f"| `{spec.key}` | `{spec.relative_path}` | {spec.kind} | {rows} | {gaze} |"
        )
    lines.append("")
    return "\n".join(lines)


def dataset_section(key: str) -> str:
    spec = get_dataset(key)
    frame = load_dataset(spec)
    parts = [
        _heading(spec.title, 2),
        spec.description,
        "",
        f"- file: `{spec.relative_path}`",
        f"- rows: {len(frame)}",
        f"- columns: {', '.join(frame.columns)}",
        "",
    ]
    for note in spec.notes:
        parts.append(f"- note: {note}")
    if spec.notes:
        parts.append("")
    if spec.label_column:
        parts.append(label_table_markdown(label_counts(frame[spec.label_column]), "Label mix"))
        parts.append("")
    if spec.gaze_columns:
        desc = describe_numeric(frame, spec.gaze_columns)
        keep = [col for col in ("column", "mean", "std", "min", "median", "max") if col in desc.columns]
        parts.append(_heading("Gaze summary", 3))
        parts.append(markdown_table(desc[keep]))
        parts.append("")
        if len(spec.gaze_columns) >= 2:
            corr = correlation_matrix(frame, spec.gaze_columns)
            pairs = high_correlations(corr, threshold=0.8)
            if not pairs.empty:
                parts.append("High correlations (|r| ≥ 0.80):")
                parts.append("")
                parts.append(markdown_table(pairs))
                parts.append("")
        report = scaler_report(frame, list(spec.gaze_columns))
        kind = (
            "standard-like"
            if report.standard_like
            else "min-max-like"
            if report.minmax_like
            else "raw or mixed"
        )
        parts.append(f"Scaling fingerprint: **{kind}**.")
        parts.append("")
    return "\n".join(parts)


def split_section() -> str:
    zuco_parent = load_dataset("zuco_sst_standard")
    sst_parent = load_dataset("sst_full")
    zuco = audit_split(
        "ZuCo ∩ SST 80/10/10",
        zuco_parent,
        load_dataset("zuco_sst_train"),
        load_dataset("zuco_sst_valid"),
        load_dataset("zuco_sst_test"),
        id_column="sentence_id",
        label_column="sentiment_label",
    )
    sst = audit_split(
        "Full SST 80/10/10",
        sst_parent,
        load_dataset("sst_train"),
        load_dataset("sst_valid"),
        load_dataset("sst_test"),
        id_column="sentence_id",
        label_column="sentiment_label",
    )
    return "\n".join(
        [
            _heading("Split audits"),
            audit_to_markdown(zuco),
            "",
            audit_to_markdown(sst),
            "",
        ]
    )


def build_dataset_report() -> str:
    """Full markdown report used by ``examples/08_write_dataset_report.py``."""
    sections = [
        "# Dataset report",
        "",
        "Generated from the CSVs in this personal repository. Re-run",
        "`python examples/08_write_dataset_report.py` after data changes.",
        "",
        catalog_section(),
        dataset_section("zuco_sst_standard"),
        dataset_section("zuco_sentence_raw"),
        dataset_section("zuco_word_raw"),
        dataset_section("sst_full"),
        dataset_section("provo_word"),
        split_section(),
    ]
    return "\n".join(sections).rstrip() + "\n"


def write_dataset_report(path: Path | None = None) -> Path:
    target = path or (repo_root() / "docs" / "generated" / "dataset-report.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_dataset_report(), encoding="utf-8")
    return target
