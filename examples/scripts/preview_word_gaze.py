"""Peek at word-level ZuCo averages, predicted SST gaze, and Provo."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import sys

from teag_examples.io import load_predicted_word_gaze, load_provo, load_word_averages
from teag_examples.paths import examples_output
from teag_examples.reports import write_json, write_markdown
from teag_examples.stats import frame_profile


def _head_md(df, n: int = 5) -> str:
    show = df.head(n)
    return show.to_markdown(index=False)


def main(argv: list[str] | None = None) -> int:
    del argv
    words = load_word_averages("v2")
    pred = load_predicted_word_gaze("test")
    provo = load_provo()

    # prediction_test_v2 is large; only count unique sentences via the small dump + schema.
    payload = {
        "word_averages_v2": frame_profile(words, "word_averages_v2"),
        "prediction_test": frame_profile(pred, "prediction_test"),
        "provo": frame_profile(provo, "provo"),
        "word_averages_unique_sent": int(words["Sent_ID"].nunique()),
        "prediction_test_unique_sent": int(pred["sentence_id"].nunique()),
        "provo_unique_sent": int(provo["sentence_id"].nunique()),
        "mean_wordlen": float(words["WordLen"].mean()),
        "mean_nFixations_word": float(words["nFixations"].mean()),
        "zero_nFixations_share": float((words["nFixations"] == 0).mean()),
    }

    md = f"""# Word-level gaze preview

## ZuCo `word_averages_v2.csv`

- rows: {payload['word_averages_v2']['rows']}
- distinct `Sent_ID`: {payload['word_averages_unique_sent']}
- mean `WordLen`: {payload['mean_wordlen']:.2f}
- mean `nFixations`: {payload['mean_nFixations_word']:.3f}
- share of words with `nFixations == 0`: {payload['zero_nFixations_share']:.3f}

First five rows:

{_table(words.head())}

## Predicted gaze `prediction_test.csv` (sentences 300–399)

- rows: {payload['prediction_test']['rows']}
- distinct sentences: {payload['prediction_test_unique_sent']}
- columns: {payload['prediction_test']['columns']}

First five rows:

{_table(pred.head())}

## Provo extract

- rows: {payload['provo']['rows']}
- distinct sentences: {payload['provo_unique_sent']}
- gaze columns include `fixProp` instead of `GD`

First five rows:

{_table(provo.head())}

`prediction_test_v2.csv` has 191,971 word rows for the full SST stream; skip
loading it in this preview. Sentence-level models never read these word tables.
"""
    write_markdown(md, examples_output() / "word_gaze_preview.md")
    write_json(payload, examples_output() / "word_gaze_preview.json")
    sys.stdout.write(md + "\n")
    return 0


def _table(df) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


if __name__ == "__main__":
    raise SystemExit(main())
