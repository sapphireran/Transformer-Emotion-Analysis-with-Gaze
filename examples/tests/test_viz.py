from pathlib import Path

from teag_examples.viz import save_corr_heatmap, save_fusion_shapes, save_label_bars_panels
import pandas as pd


def test_viz_writers_create_pngs(tmp_path: Path):
    corr = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["a", "b"], columns=["a", "b"])
    heat = save_corr_heatmap(corr, "toy", tmp_path / "corr.png")
    assert heat.is_file() and heat.stat().st_size > 0
    schematic = save_fusion_shapes(tmp_path / "fusion.png")
    assert schematic.is_file() and schematic.stat().st_size > 0
    bars = save_label_bars_panels(
        {
            "left": {"all": {0: 3, 1: 1, 2: 2}},
            "right": {"all": {0: 10, 1: 4, 2: 8}},
        },
        tmp_path / "bars.png",
        {0: "neg", 1: "neu", 2: "pos"},
    )
    assert bars.is_file() and bars.stat().st_size > 0
