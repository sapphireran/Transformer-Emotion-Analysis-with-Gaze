#!/usr/bin/env python3
"""Print the feature glossary and which columns each trainer actually consumes."""

from __future__ import annotations

import pandas as pd

from examples._common import banner
from tea_gaze.reports import glossary_frame, markdown_table, sentiment_legend
from tea_gaze.schema import (
    FEATURE_GLOSSARY,
    MODEL_TYPES,
    SST_MODEL_FEATURES,
    ZUCO_MODEL_FEATURES,
    ZUCO_SENTENCE_ET_COLUMNS,
    ZUCO_WORD_ET_COLUMNS,
)


def main() -> None:
    banner("Sentiment legend used by every trainer in this repo")
    print(sentiment_legend())

    banner("model_type values accepted by get_model()")
    for name in MODEL_TYPES:
        uses_gaze = name.endswith("eye_tracking")
        print(f"  {name:22}  gaze={'yes' if uses_gaze else 'no (tensor loaded, then ignored)'}")

    banner("Columns fused by model_ZuCo_SST.py")
    print(markdown_table(glossary_frame(ZUCO_MODEL_FEATURES, FEATURE_GLOSSARY)))
    unused = [col for col in ZUCO_SENTENCE_ET_COLUMNS if col not in ZUCO_MODEL_FEATURES]
    print("Stored on the ZuCo combined CSV but not fused:", ", ".join(unused))

    banner("Columns fused by model_full_SST.py")
    print(markdown_table(glossary_frame(SST_MODEL_FEATURES, FEATURE_GLOSSARY)))

    banner("Word-level ZuCo measures (not aligned to BPE in the trainers)")
    print(markdown_table(glossary_frame(ZUCO_WORD_ET_COLUMNS + ("WordLen",), FEATURE_GLOSSARY)))

    banner("Full glossary")
    keys = list(FEATURE_GLOSSARY)
    print(markdown_table(glossary_frame(keys, FEATURE_GLOSSARY)))

    banner("Naming trap")
    print(
        pd.DataFrame(
            [
                {"repo spelling": "nFixations", "where": "ZuCo sentence + word CSVs, model_ZuCo_SST.py"},
                {"repo spelling": "nFix", "where": "Full SST CSVs, model_full_SST.py, predicted-word files"},
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
