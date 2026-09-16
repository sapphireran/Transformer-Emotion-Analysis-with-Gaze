from teag_examples.alignment import (
    compacted_to_original_subject3,
    first_sentlen_mismatch,
    original_to_compacted_subject3,
    skip_list_matches_transformer,
    subject3_alignment_report,
)
from teag_examples.io import load_subject_sentence_gaze
from teag_examples.schema import (
    SUBJECT_3_SKIP_ORIGINAL,
    ZUCO_N_SENTENCES,
    ZUCO_SUBJECT_3_N_SENTENCES,
)


def test_skip_list_size():
    assert skip_list_matches_transformer()
    assert original_to_compacted_subject3(149) == 149
    assert original_to_compacted_subject3(150) is None
    assert original_to_compacted_subject3(249) is None
    assert original_to_compacted_subject3(250) == 150
    assert original_to_compacted_subject3(398) == 298
    assert original_to_compacted_subject3(399) is None
    assert compacted_to_original_subject3(0) == 0
    assert compacted_to_original_subject3(149) == 149
    assert compacted_to_original_subject3(150) == 250
    assert compacted_to_original_subject3(298) == 398


def test_round_trip_every_original_index():
    skip = set(SUBJECT_3_SKIP_ORIGINAL)
    mapped = []
    for orig in range(ZUCO_N_SENTENCES):
        compact = original_to_compacted_subject3(orig)
        if orig in skip:
            assert compact is None
            continue
        assert compacted_to_original_subject3(compact) == orig
        mapped.append(compact)
    assert mapped == list(range(ZUCO_SUBJECT_3_N_SENTENCES))


def test_sentlen_mismatch_starts_at_row_150():
    s1 = load_subject_sentence_gaze(1)
    s3 = load_subject_sentence_gaze(3)
    assert first_sentlen_mismatch(s1, s3) == 150
    report = subject3_alignment_report(s3, s1)
    assert report["aligned_prefix_rows"] == 150
    assert report["compacted_150_matches_original_250"] is True
    assert report["n_subject3"] == 299
