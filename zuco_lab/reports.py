"""Markdown / HTML writers for the personal lab notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def md_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _cell(value: object) -> str:
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.2f}"
        return f"{value:.4f}"
    return str(value)


def md_heading(text: str, level: int = 2) -> str:
    return f"{'#' * level} {text}"


def join_sections(parts: Sequence[str]) -> str:
    return "\n\n".join(part.rstrip() for part in parts if part) + "\n"


def write_text(path: Path | str, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    return path


def html_page(title: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --ink: #241c15;
      --accent: #8b3d2f;
      --card: #fffdf8;
      --line: #d8cfc0;
    }}
    body {{
      margin: 0 auto;
      max-width: 880px;
      padding: 2.2rem 1.4rem 3rem;
      background: var(--bg);
      color: var(--ink);
      font: 16px/1.55 "Iowan Old Style", "Palatino Linotype", Palatino, serif;
    }}
    h1, h2, h3 {{ font-weight: 650; letter-spacing: -0.02em; }}
    h1 {{ font-size: 2rem; margin-bottom: 0.2rem; }}
    .sub {{ color: #5c5146; margin-top: 0; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: var(--card);
      margin: 0.8rem 0 1.4rem;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 0.4rem 0.55rem;
      text-align: left;
      font-size: 0.92rem;
    }}
    th {{ background: #efe6d6; }}
    code, pre {{
      font-family: "IBM Plex Mono", "Source Code Pro", Consolas, monospace;
      font-size: 0.86rem;
    }}
    pre {{
      background: #2b241c;
      color: #f4ead8;
      padding: 0.9rem 1rem;
      overflow-x: auto;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      padding: 0.9rem 1rem;
      margin: 1rem 0;
    }}
    .warn {{ border-left: 4px solid var(--accent); }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {body_html}
</body>
</html>
"""


def md_table_to_html(md: str) -> str:
    lines = [line for line in md.splitlines() if line.startswith("|")]
    if len(lines) < 2:
        return ""
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]
    headers = cells(lines[0])
    rows = [cells(line) for line in lines[2:]]
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
