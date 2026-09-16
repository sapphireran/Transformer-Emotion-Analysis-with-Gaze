#!/usr/bin/env python3
"""Walk sentence 0 from word-level averages up to the labeled training row."""

from __future__ import annotations

from _util import maybe_write, parser
from zuco_lab import csvio, paths
from zuco_lab.alignment import group_words, sentence_word_profile
from zuco_lab.reports import join_sections, md_heading, md_table
from zuco_lab.subjects import remap_word_sent_id


def build(sentence_id: int = 0) -> str:
    grouped = group_words()
    sent_key = f"{sentence_id}_NR"
    profile = sentence_word_profile(sent_key, grouped)
    words = grouped[sent_key]

    labeled = next(
        row
        for row in csvio.read_dicts(paths.ZUCO_COMBINED_STANDARD)
        if int(row["sentence_id"]) == sentence_id
    )
    raw_avg = next(
        row
        for row in csvio.read_dicts(paths.ET_AVERAGE)
        if int(float(row["id"])) == sentence_id
    )
    z_avg = next(
        row
        for row in csvio.read_dicts(paths.ET_AVERAGE_STANDARD)
        if int(float(row["id"])) == sentence_id
    )

    preview = [
        (
            int(float(row["Word_ID"])),
            row["Word"],
            float(row["nFixations"]),
            float(row["FFD"]),
            float(row["TRT"]),
            float(row["WordLen"]),
        )
        for row in words[:12]
    ]

    # Subject 3 word-level remapping check on this sentence.
    from zuco_lab.subjects import load_subject_words

    w1 = [r for r in load_subject_words(1) if r["Sent_ID"] == sent_key]
    w3 = [r for r in load_subject_words(3) if remap_word_sent_id(3, r["Sent_ID"]) == sent_key]

    return join_sections(
        [
            md_heading(f"Word → sentence walkthrough (id {sentence_id})", 1),
            f"**Text:** {labeled['sentence']}",
            f"**Label:** {labeled['sentiment_label']}  ",
            (
                f"Word-average tokens: {len(profile.words)} "
                f"(zeros in nFixations: {profile.n_zero_nfix}). "
                f"Mean word nFixations={profile.mean_nfix:.3f}, TRT={profile.mean_trt:.3f}."
            ),
            md_heading("First twelve word-average rows"),
            md_table(("word_id", "word", "nFixations", "FFD", "TRT", "WordLen"), preview),
            md_heading("Sentence-level published numbers"),
            md_table(
                ("table", "nFixations", "TRT", "FFD", "GPT"),
                [
                    (
                        "average_data.csv (raw-ish mean)",
                        float(raw_avg["nFixations"]),
                        float(raw_avg["TRT"]),
                        float(raw_avg["FFD"]),
                        float(raw_avg["GPT"]),
                    ),
                    (
                        "standard_scaled_average_data.csv",
                        float(z_avg["nFixations"]),
                        float(z_avg["TRT"]),
                        float(z_avg["FFD"]),
                        float(z_avg["GPT"]),
                    ),
                    (
                        "combined_sst_et_standard.csv (train)",
                        float(labeled["nFixations"]),
                        float(labeled["TRT"]),
                        float(labeled["FFD"]),
                        float(labeled["GPT"]),
                    ),
                ],
            ),
            md_heading("Reader 1 vs remapped reader 3, same sentence"),
            (
                f"Reader 1 words on {sent_key}: {len(w1)}. "
                f"Reader 3 words whose remapped Sent_ID is {sent_key}: {len(w3)}. "
                f"First tokens: reader1={w1[0]['Word'] if w1 else '—'}, "
                f"reader3={w3[0]['Word'] if w3 else '—'}."
            ),
            md_heading("Do not expect word-means to equal sentence-means"),
            (
                "``DataTransformer`` builds sentence features by summing word measures "
                "and dividing by the number of *fixated* words, not by ``SentLen``. "
                "Omitted words (nFixations=0) therefore drop out of the sentence "
                "average. The word table keeps those zeros. That is why "
                "``mean word nFixations`` above is not a copy of ``average_data``."
            ),
        ]
    )


def main() -> None:
    p = parser("Walk one sentence from word averages to the training row.")
    p.add_argument("--sentence-id", type=int, default=0)
    args = p.parse_args()
    text = build(args.sentence_id)
    print(text)
    maybe_write(args, "05_word_sentence_bridge.md", text)


if __name__ == "__main__":
    main()
