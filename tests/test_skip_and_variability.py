from examples.subject_variability import load_subjects, main as var_main
from examples.word_level_skip_analysis import is_skip, main as skip_main
from examples.lib.paths import subject_word_csv
import pandas as pd


def test_skip_predicate_on_subject_two_sentence_three():
    df = pd.read_csv(subject_word_csv(2))
    rows = df[df["Sent_ID"].astype(str) == "3_NR"]
    skip = is_skip(rows)
    # Subject 2 skipped "slow" and fixated the rest.
    assert int(skip.sum()) == 1
    assert rows.loc[skip, "Word"].str.lower().iloc[0] == "slow"


def test_skip_analysis_cli():
    assert skip_main() == 0


def test_subject_stack_shape():
    stacked = load_subjects()
    assert len(stacked) == 12 * 400
    assert set(stacked["subject"]) == set(range(1, 13))


def test_variability_cli_for_sentence_three():
    assert var_main(["--sentence-id", "3"]) == 0
