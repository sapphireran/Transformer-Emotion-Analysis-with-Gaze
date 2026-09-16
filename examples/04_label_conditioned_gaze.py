#!/usr/bin/env python3
"""Gaze means by sentiment label, plus collinearity of the fusion five."""

from __future__ import annotations

from _util import maybe_write, parser
from zuco_lab.alignment import feature_label_correlations, gaze_collinearity, label_conditioned_means
from zuco_lab.catalogs import FUSION_FEATURES
from zuco_lab.reports import join_sections, md_heading, md_table


def build() -> str:
    profiles = label_conditioned_means()
    features = list(profiles[0].means)
    header = ["label", "n", *features]
    rows = [
        (f"{p.label} {p.name}", p.n, *[p.means[name] for name in features])
        for p in profiles
    ]
    corrs = feature_label_correlations()
    pairs = gaze_collinearity()
    return join_sections(
        [
            md_heading("Label-conditioned gaze (ZuCo standard table)", 1),
            (
                "400 z-scored sentences. A mean of 0 is the corpus centre. Negative "
                "nFixations on the negative class means those sentences were, on "
                "average, fixated slightly less than the corpus mean — not that "
                "people closed their eyes."
            ),
            md_table(header, rows),
            md_heading("Pearson r with integer label 0/1/2"),
            md_table(("feature", "r"), [(k, v) for k, v in corrs.items()]),
            md_heading("Collinearity among fusion features"),
            md_table(("a", "b", "r"), pairs),
            md_heading("Why this matters for EyeTrackingModel"),
            (
                f"The fusion head sees {', '.join(FUSION_FEATURES)} through one "
                "Linear(5 → 16). If TRT and GPT are highly correlated, that layer "
                "is not getting five independent cues. Label correlations here are "
                "small; gaze-only linear baselines in example 09 are the better "
                "floor for 'does gaze separate sentiment on its own?'."
            ),
        ]
    )


def main() -> None:
    args = parser("Summarize gaze by label and feature collinearity.").parse_args()
    text = build()
    print(text)
    maybe_write(args, "04_label_conditioned_gaze.md", text)


if __name__ == "__main__":
    main()
