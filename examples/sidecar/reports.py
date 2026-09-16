"""Markdown / HTML writers for the personal atlas outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import OUTPUT_DIR


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Tiny GitHub-flavored table without pulling in tabulate."""
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: floatfmt.format(x) if pd.notna(x) else "")
    cols = [str(c) for c in work.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in work.iterrows():
        body.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join([header, sep, *body])


def write_text(name: str, text: str) -> Path:
    path = ensure_output_dir() / name
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path


def html_page(title: str, sections: list[tuple[str, str]]) -> str:
    """Self-contained HTML so the atlas can be opened in a browser."""
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'/>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: Georgia, 'Times New Roman', serif; max-width: 960px;",
        "  margin: 2rem auto; padding: 0 1.2rem; line-height: 1.45; color: #1a1a1a; }",
        "h1 { font-size: 1.8rem; }",
        "h2 { margin-top: 1.8rem; border-bottom: 1px solid #ccc; padding-bottom: 0.2rem; }",
        "table { border-collapse: collapse; margin: 0.8rem 0 1.2rem; font-size: 0.92rem; }",
        "th, td { border: 1px solid #bbb; padding: 0.28rem 0.55rem; text-align: left; }",
        "th { background: #f3f1ea; }",
        "code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }",
        "pre { background: #f6f4ee; padding: 0.8rem; overflow-x: auto; }",
        ".note { background: #fff7d6; padding: 0.7rem 0.9rem; border-left: 4px solid #c9a227; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        "<p>Personal lab atlas for <code>Transformer-Emotion-Analysis-with-Gaze</code>.",
        " Numbers below are computed from the committed CSVs (no GPU, no Hub downloads).</p>",
    ]
    for heading, html in sections:
        parts.append(f"<h2>{heading}</h2>")
        parts.append(html)
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


def markdown_to_html_table(md: str) -> str:
    """Convert a GFM table (from markdown_table) into an HTML table."""
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return f"<pre>{md}</pre>"
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip("|").split("|")]
    header = cells(lines[0])
    rows = [cells(ln) for ln in lines[2:]]
    html = ["<table>", "<thead><tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr></thead>", "<tbody>"]
    for row in rows:
        html.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    html.append("</tbody></table>")
    return "\n".join(html)
