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


def false_green(df: pd.DataFrame, pred_col: str) -> int:
    return int(len(df[(df["true_status"] != "GREEN") & (df[pred_col] == "GREEN")]))


def red_recall(df: pd.DataFrame, pred_col: str) -> float:
    d = df[df["true_status"] == "RED"]
    if len(d) == 0:
        return 0.0
    return float((d[pred_col] == "RED").mean())


def red_caught(df: pd.DataFrame, pred_col: str) -> float:
    d = df[df["true_status"] == "RED"]
    if len(d) == 0:
        return 0.0
    return float((d[pred_col] != "GREEN").mean())


