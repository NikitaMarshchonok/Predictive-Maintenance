import numpy as np
import pandas as pd

from src.predict import make_features_last_cycle


def test_make_features_last_cycle_has_all_columns():
    # minimal fake df with 2 units and a few cycles
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2, 2],
            "cycle":   [1, 2, 3, 1, 2, 3],
            "s1":      [10, 11, 12, 20, 21, 22],
            "s2":      [ 1,  2,  3,  4,  5,  6],
        }
    )

    feature_cols = ["s1_rm3", "s1_rs3", "s1_sl3", "s2_rm3", "s2_rs3", "s2_sl3"]
    feat = make_features_last_cycle(df, feature_cols=feature_cols, window=3)

    assert list(feat.columns) == ["unit_id"] + feature_cols
    assert feat.shape[0] == 2  # 2 units
    assert np.isfinite(feat[feature_cols].to_numpy()).all()
