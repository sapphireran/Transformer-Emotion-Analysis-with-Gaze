"""Markdown and HTML reports for the personal examples."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Sequence

from tea_gaze.baselines import ExperimentResult
from tea_gaze.eval import MetricSet, format_confusion, confusion_matrix
from tea_gaze.features import SENTIMENT_LABELS, describe_features, list_features
from tea_gaze.io import DatasetInventoryItem
from tea_gaze.schema import summarize_labels


def markdown_inventory(items: Sequence[DatasetInventoryItem]) -> str:
    lines = [
        "# Dataset inventory",
        "",
        "Personal CSVs already in this repository. Row counts are data rows, not counting the header.",
        "",
        "| Key | Path | Kind | Gaze | Rows | Columns |",
        "|---|---|---|---|---:|---|",
    ]
    for item in items:
        rows = "—" if item.rows is None else str(item.rows)
        cols = ", ".join(f"`{name}`" for name in item.columns[:8])
        if len(item.columns) > 8:
            cols += f", … (+{len(item.columns) - 8})"
        if not item.exists:
            cols = "_missing_"
        lines.append(
            f"| `{item.key}` | `{item.path}` | {item.kind} | {item.gaze_source} | {rows} | {cols} |"
        )
    lines.extend(["", "## Descriptions", ""])
    for item in items:
        lines.append(f"### `{item.key}`")
        lines.append("")
        lines.append(item.description)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def markdown_label_table(counts: dict[int, int]) -> str:
    total = sum(counts.values()) or 1
    lines = [
        "| Label | Name | Count | Share |",
        "|---:|---|---:|---:|",
    ]
    for label, name in SENTIMENT_LABELS.items():
        n = counts.get(label, 0)
        lines.append(f"| {label} | {name} | {n} | {n / total:.1%} |")
    lines.append(f"| | **total** | **{sum(counts.values())}** | 100% |")
    return "\n".join(lines)


def markdown_feature_glossary() -> str:
    return (
        "# Gaze feature glossary\n\n"
        "Units below describe the *raw* ZuCo extracts. After `StandardScaler` "
        "or the z-score used in `combined_sst_et_standard.csv`, the same "
        "columns are dimensionless.\n\n"
        + describe_features(list_features())
        + "\n"
    )


def markdown_baseline_suite(
    results: Sequence[ExperimentResult],
    *,
    title: str,
    extra_notes: Sequence[str] | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        "Stratified 5-fold cross-validation on the personal ZuCo+SST table. "
        "Metrics are weighted the same way as `model_ZuCo_SST.py` "
        "(accuracy plus weighted precision / recall / F1).",
        "",
        "| Model | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(result.summary_row())
    lines.extend(["", "## Per-fold scores", ""])
    for result in results:
        lines.append(f"### `{result.name}`")
        lines.append("")
        for fold in result.folds:
            lines.append(f"- fold {fold.fold}: {fold.metrics.format_line('metrics')}")
        lines.append(f"- mean: {result.mean.format_line('metrics')}")
        lines.append("")
        all_true = [label for fold in result.folds for label in fold.y_true]
        all_pred = [label for fold in result.folds for label in fold.y_pred]
        lines.append("Pooled confusion matrix (all held-out folds):")
        lines.append("")
        lines.append("```")
        lines.append(format_confusion(confusion_matrix(all_true, all_pred)))
        lines.append("```")
        lines.append("")
    if extra_notes:
        lines.append("## Notes")
        lines.append("")
        for note in extra_notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def html_page(title: str, body_markdown_like_blocks: Sequence[str]) -> str:
    """Minimal HTML wrapper so a browser can open an example report."""
    inner = "\n".join(body_markdown_like_blocks)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1b1f24;
      --muted: #5b6570;
      --line: #d7dde3;
      --bg: #f7f4ef;
      --card: #fffdf8;
      --accent: #2f5d50;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      max-width: 920px;
      margin: 2.5rem auto 4rem;
      padding: 0 1.5rem;
    }}
    h1, h2, h3 {{ font-weight: 600; letter-spacing: -0.02em; }}
    h1 {{ font-size: 2rem; margin-bottom: 0.4rem; }}
    p.lead {{ color: var(--muted); font-size: 1.05rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      margin: 1rem 0 1.5rem;
      font-family: "Source Sans 3", "Helvetica Neue", sans-serif;
      font-size: 0.95rem;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 0.45rem 0.65rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #efe8dc; }}
    code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.9em; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 1rem 1.2rem;
      margin: 1rem 0;
    }}
    .accent {{ color: var(--accent); }}
    pre {{
      background: #1b1f24;
      color: #f4efe6;
      padding: 0.9rem 1rem;
      border-radius: 10px;
      overflow: auto;
      font-size: 0.85rem;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    {inner}
  </main>
</body>
</html>
"""


def html_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def html_inventory(items: Sequence[DatasetInventoryItem]) -> str:
    rows = []
    for item in items:
        rows.append(
            (
                item.key,
                item.path,
                item.kind,
                item.gaze_source,
                item.rows if item.rows is not None else "missing",
                len(item.columns),
            )
        )
    table = html_table(
        ("Key", "Path", "Kind", "Gaze", "Rows", "Columns"),
        rows,
    )
    cards = []
    for item in items:
        cards.append(
            "<div class='card'>"
            f"<h3><code>{escape(item.key)}</code></h3>"
            f"<p>{escape(item.description)}</p>"
            "</div>"
        )
    return html_page(
        "Personal dataset inventory",
        [
            "<p class='lead'>Checked-in CSVs for Sapphire's gaze + sentiment experiments. "
            "No company data. Measured gaze comes from ZuCo Task 1; predicted gaze "
            "fills the larger SST tables.</p>",
            table,
            "<h2>What each file is</h2>",
            "".join(cards),
        ],
    )


def html_baselines(results: Sequence[ExperimentResult]) -> str:
    rows = [
        (
            result.name,
            f"{result.mean.accuracy:.4f}",
            f"{result.mean.precision:.4f}",
            f"{result.mean.recall:.4f}",
            f"{result.mean.f1:.4f}",
        )
        for result in results
    ]
    table = html_table(
        ("Model", "Accuracy", "Precision", "Recall", "F1"),
        rows,
    )
    fold_blocks = []
    for result in results:
        items = "".join(
            f"<li>fold {fold.fold}: {escape(fold.metrics.format_line('metrics'))}</li>"
            for fold in result.folds
        )
        fold_blocks.append(
            f"<div class='card'><h3><code>{escape(result.name)}</code></h3>"
            f"<ul>{items}</ul></div>"
        )
    return html_page(
        "Text vs gaze vs fusion baselines",
        [
            "<p class='lead'>CPU-only logistic baselines on the 400-sentence ZuCo+SST "
            "table. This is a documentation companion to the transformer scripts, "
            "not a replacement for them.</p>",
            table,
            "<h2>Folds</h2>",
            "".join(fold_blocks),
        ],
    )


def metric_dict(metrics: MetricSet) -> dict[str, float | int]:
    return metrics.as_dict()


def label_markdown_from_rows(rows: Sequence[dict]) -> str:
    return markdown_label_table(summarize_labels(rows))
