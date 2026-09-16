from tea_gaze.features import CORE_FUSION_FEATURES
from tea_gaze.io import load_rows
from tea_gaze.stats import pearson, summarize_column


def test_raw_trt_is_longer_than_ffd_on_average():
    rows = load_rows("zuco_et_average")
    trt = summarize_column("TRT", (float(row["TRT"]) for row in rows))
    ffd = summarize_column("FFD", (float(row["FFD"]) for row in rows))
    assert trt.mean > ffd.mean
    assert trt.minimum >= 0
    assert ffd.minimum >= 0


def test_nfixations_correlates_with_trt():
    rows = load_rows("zuco_et_average")
    nfix = [float(row["nFixations"]) for row in rows]
    trt = [float(row["TRT"]) for row in rows]
    assert pearson(nfix, trt) > 0.7


def test_standard_join_has_all_fusion_columns():
    rows = load_rows("zuco_sst_standard")
    for name in CORE_FUSION_FEATURES:
        assert name in rows[0]
