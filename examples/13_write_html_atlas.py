#!/usr/bin/env python3
"""Assemble a single HTML atlas from the markdown outputs of the other examples."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.paths import OUTPUT_DIR, ROOT  # noqa: E402
from sidecar.reports import html_page, markdown_to_html_table, write_text  # noqa: E402

PARTS = [
    ("inventory.md", "Committed tables"),
    ("label_atlas.md", "Labels and majority baselines"),
    ("gaze_rank.md", "Sidecar rank (PCA)"),
    ("subject3_reindex.md", "Subject-3 packed ids"),
    ("split_fingerprint.md", "Splits and text leaks"),
    ("last_batch_metric.md", "Last-batch test metric"),
    ("length_confound.md", "Length confound"),
    ("fusion_forward.md", "Fusion geometry"),
    ("word_skips.md", "Word-level skips"),
    ("shuffle_control.md", "Shuffle-gaze control"),
    ("gaze_prediction_track.md", "Prediction-track scales"),
    ("reader_means.md", "Reader means"),
]


def _md_to_html(md: str) -> str:
    """Very small markdown subset: headings, lists, tables, fenced code, paragraphs."""
    chunks: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            fence = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                fence.append(html.escape(lines[i]))
                i += 1
            i += 1
            chunks.append("<pre><code>" + "\n".join(fence) + "</code></pre>")
            continue
        if re.match(r"^\|.+\|$", line) and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and re.match(r"^\|.+\|$", lines[i]):
                table_lines.append(lines[i])
                i += 1
            chunks.append(markdown_to_html_table("\n".join(table_lines)))
            continue
        if line.startswith("## "):
            chunks.append(f"<h3>{html.escape(line[3:])}</h3>")
            i += 1
            continue
        if line.startswith("# "):
            i += 1
            continue
        if line.startswith("### "):
            chunks.append(f"<h4>{html.escape(line[4:])}</h4>")
            i += 1
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            chunks.append("<ul>" + "".join(items) + "</ul>")
            continue
        if not line.strip():
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "-", "|", "`")):
            if lines[i].startswith("```"):
                break
            para.append(lines[i])
            i += 1
        chunks.append("<p>" + " ".join(_inline(p) for p in para) + "</p>")
    return "\n".join(chunks)


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def main() -> int:
    sections: list[tuple[str, str]] = []
    missing = []
    for filename, heading in PARTS:
        path = OUTPUT_DIR / filename
        if not path.is_file():
            missing.append(filename)
            sections.append((heading, f"<p class='note'>Missing {html.escape(filename)}. Run that example first.</p>"))
            continue
        md = path.read_text(encoding="utf-8")
        sections.append((heading, _md_to_html(md)))
    page = html_page("Gaze Sidecar Atlas — personal lab notes", sections)
    html_path = write_text("sidecar_atlas.html", page)
    index = "\n".join(
        [
            "# Generated atlas tables",
            "",
            f"Written under `{OUTPUT_DIR.relative_to(ROOT)}/` by `examples/run_all.py`.",
            "",
            *[f"- `{name}` — {heading}" for name, heading in PARTS],
            "",
            "Open `sidecar_atlas.html` in a browser for the combined view.",
            "",
        ]
    )
    write_text("README.md", index)
    print(f"wrote {html_path}")
    if missing:
        print("missing", missing)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
