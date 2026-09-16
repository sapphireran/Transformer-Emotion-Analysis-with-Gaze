#!/usr/bin/env python3
"""Walk one ZuCo sentence from words + gaze through a tiny fusion prediction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tea_gaze.baselines import _classifier, _vectorizer, prepare_xy
from tea_gaze.features import CORE_FUSION_FEATURES, SENTIMENT_LABELS, label_name
from tea_gaze.io import load_frame
from tea_gaze.paths import repo_root
from tea_gaze.reports import write_text
from tea_gaze.text import tokenize


def _pick_examples(frame):
    """Prefer short sentences, one per label, so the write-up stays readable."""
    picked = {}
    ordered = frame.assign(_len=frame["sentence"].str.len()).sort_values("_len")
    for row in ordered.itertuples(index=False):
        label = int(row.sentiment_label)
        if label not in picked:
            picked[label] = row
        if len(picked) == 3:
            break
    return [picked[label] for label in sorted(picked)]


def main() -> int:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import csr_matrix, hstack

    sentences = load_frame("zuco_sst_standard")
    words = load_frame("zuco_word_averages")
    words["sentence_id"] = words["Sent_ID"].astype(str).str.split("_").str[0].astype(int)

    train_frame, _ = train_test_split(
        sentences,
        test_size=0.2,
        random_state=42,
        stratify=sentences["sentiment_label"],
    )
    texts_train, gaze_train, y_train = prepare_xy(train_frame)
    vectorizer = _vectorizer()
    scaler = StandardScaler()
    text_train = vectorizer.fit_transform(texts_train)
    gaze_train_s = scaler.fit_transform(gaze_train)
    model = _classifier()
    model.fit(hstack([text_train, csr_matrix(gaze_train_s)]), y_train)

    examples = _pick_examples(sentences)
    blocks = [
        "# Sentence walkthrough",
        "",
        "A fused logistic model is fit on an 80% stratified split of the 400 "
        "ZuCo+SST sentences. The sentences below are short examples from each "
        "label, chosen so the word-level gaze table stays readable. This is an "
        "explanation aid, not a held-out score.",
        "",
    ]

    for row in examples:
        sid = int(row.sentence_id)
        word_rows = words[words["sentence_id"] == sid].sort_values("Word_ID")
        text_vec = vectorizer.transform([str(row.sentence)])
        gaze_vec = scaler.transform(
            [[float(getattr(row, name)) for name in CORE_FUSION_FEATURES]]
        )
        fused = hstack([text_vec, csr_matrix(gaze_vec)])
        pred = int(model.predict(fused)[0])
        proba = model.predict_proba(fused)[0]
        tokens = tokenize(str(row.sentence))

        blocks.append(f"## Sentence {sid}: {label_name(int(row.sentiment_label))}")
        blocks.append("")
        blocks.append(f"> {row.sentence}")
        blocks.append("")
        blocks.append(
            f"Predicted label: **{label_name(pred)}** "
            + "("
            + ", ".join(
                f"{SENTIMENT_LABELS[int(cls)]}={proba[i]:.2f}"
                for i, cls in enumerate(model.classes_)
            )
            + ")"
        )
        blocks.append("")
        blocks.append("Sentence-level fusion features (z-scored join):")
        blocks.append("")
        blocks.append("| Feature | Value |")
        blocks.append("|---|---:|")
        for name in CORE_FUSION_FEATURES:
            blocks.append(f"| `{name}` | {float(getattr(row, name)):.3f} |")
        blocks.append("")
        blocks.append(f"Tokenizer view ({len(tokens)} tokens): `" + " ".join(tokens) + "`")
        blocks.append("")
        if word_rows.empty:
            blocks.append("_No word-level rows for this sentence id._")
        else:
            blocks.append("Word-level ZuCo averages (raw units, 12 readers):")
            blocks.append("")
            blocks.append("| Word | nFix | FFD | GD | TRT | GPT |")
            blocks.append("|---|---:|---:|---:|---:|---:|")
            for word in word_rows.itertuples(index=False):
                token = word.Word if str(word.Word).strip() else "unknown"
                blocks.append(
                    f"| `{token}` | {word.nFixations:.2f} | {word.FFD:.1f} | "
                    f"{word.GD:.1f} | {word.TRT:.1f} | {word.GPT:.1f} |"
                )
        blocks.append("")

    path = write_text(repo_root() / "examples" / "output" / "sentence_walkthrough.md", "\n".join(blocks))
    print(f"Wrote {path.relative_to(repo_root())}")
    for row in examples:
        print(f"sentence {int(row.sentence_id)} gold={label_name(int(row.sentiment_label))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
