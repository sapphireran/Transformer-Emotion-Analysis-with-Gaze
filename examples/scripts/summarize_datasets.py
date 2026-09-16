"""Print an inventory of every committed sentiment / gaze table."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from teag_examples.io import (
    load_full_sst_combined,
    load_full_sst_split,
    load_predicted_word_gaze,
    load_provo,
    load_raw_sst_sentences,
    load_sentence_average,
    load_subject_sentence_gaze,
    load_word_averages,
    load_zuco_combined,
    load_zuco_split,
    load_zuco_text,
)
from teag_examples.paths import docs_assets, examples_output
from teag_examples.reports import write_json, write_markdown
from teag_examples.stats import frame_profile, profiles_to_markdown
from teag_examples.viz import save_label_bars_panels
from teag_examples.schema import LABEL_NAMES


def _all_profiles() -> list[dict]:
    profiles = [
        frame_profile(load_zuco_text(), "ZuCo_SST_data/ssts_ZuCo.csv"),
        frame_profile(load_zuco_combined("standard"), "ZuCo_SST_data/combined_sst_et_standard.csv"),
        frame_profile(load_zuco_combined("min_max"), "ZuCo_SST_data/combined_sst_et_min_max.csv"),
    ]
    for split in ("train", "valid", "test"):
        profiles.append(frame_profile(load_zuco_split(split), f"ZuCo_SST_data/{split}.csv"))
    profiles.append(frame_profile(load_full_sst_combined(), "SST_data/combined_full_sst_et.csv"))
    for split in ("train", "valid", "test"):
        profiles.append(
            frame_profile(load_full_sst_split(split), f"SST_data/{split}_full_sst.csv")
        )
    raw = load_raw_sst_sentences()
    profiles.append(
        {
            "name": "SST_data/stts_all_sentence_level.csv",
            "rows": int(len(raw)),
            "columns": list(raw.columns),
            "n_numeric": 0,
            "n_missing": int(raw.isna().sum().sum()),
            "duplicate_rows": int(raw.duplicated().sum()),
            "label_counts": {
                0: int((raw["label_string"] == "NEGATIVE").sum()),
                1: int((raw["label_string"] == "NEUTRAL").sum()),
                2: int((raw["label_string"] == "POSITIVE").sum()),
            },
            "moments": {},
        }
    )
    for kind in ("raw", "standard", "min_max"):
        profiles.append(
            frame_profile(load_sentence_average(kind), f"ZuCo_et_csv_data/{kind}_average")
        )
    for subject in range(1, 13):
        df = load_subject_sentence_gaze(subject)
        profiles.append(
            {
                "name": f"ZuCo_et_csv_data/{subject}_SR.csv",
                "rows": int(len(df)),
                "columns": list(df.columns),
                "n_numeric": int(df.select_dtypes("number").shape[1]),
                "n_missing": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "moments": {},
            }
        )
    profiles.append(frame_profile(load_word_averages(), "word_averages_v2.csv"))
    profiles.append(frame_profile(load_provo(), "provo.csv"))
    profiles.append(frame_profile(load_predicted_word_gaze("test"), "prediction_test.csv"))
    return profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="also print JSON to stdout")
    args = parser.parse_args(argv)

    profiles = _all_profiles()
    md = "# Dataset inventory\n\n" + profiles_to_markdown(profiles)
    md += "\nSubject 3 is the only sentence table with 299 rows; the others have 400.\n"
    out = examples_output() / "dataset_inventory.md"
    assets = docs_assets() / "dataset_inventory.md"
    write_markdown(md, out)
    write_markdown(md, assets)
    write_json(profiles, examples_output() / "dataset_inventory.json")

    label_tables = {
        "ZuCo-400": load_zuco_combined()["sentiment_label"].value_counts().sort_index().to_dict(),
        "ZuCo-train": load_zuco_split("train")["sentiment_label"].value_counts().sort_index().to_dict(),
        "ZuCo-valid": load_zuco_split("valid")["sentiment_label"].value_counts().sort_index().to_dict(),
        "ZuCo-test": load_zuco_split("test")["sentiment_label"].value_counts().sort_index().to_dict(),
        "SST-all": load_full_sst_combined()["sentiment_label"].value_counts().sort_index().to_dict(),
        "SST-train": load_full_sst_split("train")["sentiment_label"].value_counts().sort_index().to_dict(),
        "SST-valid": load_full_sst_split("valid")["sentiment_label"].value_counts().sort_index().to_dict(),
        "SST-test": load_full_sst_split("test")["sentiment_label"].value_counts().sort_index().to_dict(),
    }
    # json keys may be numpy ints
    label_tables = {
        name: {int(k): int(v) for k, v in counts.items()} for name, counts in label_tables.items()
    }
    save_label_bars_panels(
        {
            "ZuCo-SST (400 sentences)": {
                "all": label_tables["ZuCo-400"],
                "train": label_tables["ZuCo-train"],
                "valid": label_tables["ZuCo-valid"],
                "test": label_tables["ZuCo-test"],
            },
            "Full SST (11,853 sentences)": {
                "all": label_tables["SST-all"],
                "train": label_tables["SST-train"],
                "valid": label_tables["SST-valid"],
                "test": label_tables["SST-test"],
            },
        },
        docs_assets() / "label_counts.png",
        dict(LABEL_NAMES),
    )

    sys.stdout.write(md + "\n")
    sys.stdout.write(f"wrote {out} and {assets}\n")
    if args.json:
        sys.stdout.write(json.dumps(profiles, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
