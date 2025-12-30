import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


def read_cmapss_txt(path: Path) -> pd.DataFrame:
    """
    Read C-MAPSS train/test TXT (space-separated) into a DataFrame with named columns.
    Expected columns: unit_id, cycle, op1, op2, op3, s1..s21
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)

    cols = ["unit_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    if df.shape[1] != len(cols):
        raise ValueError(f"Unexpected number of columns: got {df.shape[1]}, expected {len(cols)}. "
                         f"Check file format: {path}")

    df.columns = cols
    return df


def infer_base_signals(feature_cols: list[str]) -> list[str]:
    """
    From feature names like s3_rm30 / s3_rs30 / s3_sl30 infer base signals: s3, s9, ...
    """
    bases = []
    for c in feature_cols:
        base = c.split("_")[0]
        if base not in bases:
            bases.append(base)
    return bases


def slope_last_window(x: np.ndarray) -> float:
    """
    Simple slope of last window using linear fit (index vs value).
    """
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x), dtype=float)
    # slope of y ~ a*t + b
    a, _b = np.polyfit(t, x.astype(float), 1)
    return float(a)


def make_features_last_cycle(df: pd.DataFrame, feature_cols: list[str], window: int) -> pd.DataFrame:
    """
    Build mean/std/slope rolling features for LAST cycle of each unit.
    Returns DataFrame with columns exactly = feature_cols + ['unit_id'].
    """
    base_signals = infer_base_signals(feature_cols)

    rows = []
    for unit_id, g in df.groupby("unit_id", sort=True):
        g = g.sort_values("cycle")
        last_cycle = int(g["cycle"].iloc[-1])

        # take last `window` rows (or fewer if unit shorter)
        tail = g.tail(window)

        feat = {"unit_id": int(unit_id), "cycle": last_cycle}

        for s in base_signals:
            if s not in tail.columns:
                # If feature list contains something unexpected
                continue

            arr = tail[s].to_numpy()

            feat[f"{s}_rm{window}"] = float(np.mean(arr))
            feat[f"{s}_rs{window}"] = float(np.std(arr, ddof=0))
            feat[f"{s}_sl{window}"] = slope_last_window(arr)

        rows.append(feat)

    feat_df = pd.DataFrame(rows)

    # Ensure all required columns exist (fill missing with 0.0)
    for c in feature_cols:
        if c not in feat_df.columns:
            feat_df[c] = 0.0

    # keep only needed order
    feat_df = feat_df[["unit_id"] + feature_cols]
    return feat_df


def main():
    parser = argparse.ArgumentParser(description="C-MAPSS FD001 RUL prediction (last-cycle) using saved model.")
    parser.add_argument("--data", type=str, required=True, help="Path to train_FD001.txt or test_FD001.txt")
    parser.add_argument("--model", type=str, default="models/fd001_rf_mss_w30_cap125.joblib", help="Path to .joblib model")
    parser.add_argument("--features", type=str, default="models/fd001_rf_mss_w30_cap125_features.json", help="Path to features JSON")
    parser.add_argument("--meta", type=str, default="models/fd001_rf_mss_w30_cap125_meta.json", help="Path to meta JSON")
    parser.add_argument("--unit-id", type=int, default=None, help="If set, predict only this unit_id")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    feats_path = Path(args.features)
    meta_path = Path(args.meta)

    df = read_cmapss_txt(data_path)

    feature_cols = json.loads(feats_path.read_text(encoding="utf-8"))

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    window = int(meta.get("window", 30))
    rul_cap = meta.get("rul_cap", None)

    model = joblib.load(model_path)

    feat_df = make_features_last_cycle(df, feature_cols, window=window)

    if args.unit_id is not None:
        feat_df = feat_df[feat_df["unit_id"] == args.unit_id].copy()
        if feat_df.empty:
            raise ValueError(f"unit_id={args.unit_id} not found in {data_path}")

    X = feat_df[feature_cols].to_numpy()
    pred = model.predict(X)

    out = pd.DataFrame({"unit_id": feat_df["unit_id"].values, "pred_rul": pred})
    out = out.sort_values("unit_id").reset_index(drop=True)

    print("Model:", model_path)
    print("Data :", data_path)
    print("Window:", window, "| RUL_CAP:", rul_cap)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
