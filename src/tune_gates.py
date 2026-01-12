import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rul_status(v: float, red_thr: float, yellow_thr: float) -> str:
    if v <= red_thr:
        return "RED"
    if v <= yellow_thr:
        return "YELLOW"
    return "GREEN"


def load_gt_rul(gt_path: Path, cap: int) -> pd.DataFrame:
    y = pd.read_csv(gt_path, header=None).iloc[:, 0].astype(float).to_numpy()
    unit_id = np.arange(1, len(y) + 1, dtype=int)
    true_rul_cap = np.clip(np.clip(y, 0, None), 0, cap)
    return pd.DataFrame({"unit_id": unit_id, "true_rul_cap": true_rul_cap})
