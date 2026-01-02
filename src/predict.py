import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

def rf_prediction_interval(model, X, low=10, high=90):
    """
    Prediction interval for RandomForest via per-tree predictions.
    Returns (p_low, p50, p_high) for each row in X.
    """
    if not hasattr(model, "estimators_"):
        return None, None, None

    # shape: (n_samples, n_estimators)
    preds = np.column_stack([est.predict(X) for est in model.estimators_])

    p_low = np.percentile(preds, low, axis=1)
    p50   = np.percentile(preds, 50, axis=1)
    p_high= np.percentile(preds, high, axis=1)
    return p_low, p50, p_high


def rul_status(rul_cap_value, red_thr=20.0, yellow_thr=50.0):
    """
    Simple traffic-light status based on capped RUL.
    """
    if rul_cap_value <= red_thr:
        return "RED"
    if rul_cap_value <= yellow_thr:
        return "YELLOW"
    return "GREEN"


def read_cmapss_txt(path: Path) -> pd.DataFrame:
    """
    Read C-MAPSS train/test TXT (space-separated) into a DataFrame with named columns.
    Expected columns: unit_id, cycle, op1, op2, op3, s1..s21
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)

    cols = ["unit_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    if df.shape[1] != len(cols):
        raise ValueError(
            f"Unexpected number of columns: got {df.shape[1]}, expected {len(cols)}. "
            f"Check file format: {path}"
        )

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
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x), dtype=float)
    a, _b = np.polyfit(t, x.astype(float), 1)
    return float(a)


def make_features_last_cycle(df: pd.DataFrame, feature_cols: list[str], window: int) -> pd.DataFrame:
    """
    Build mean/std/slope rolling features for LAST cycle of each unit.
    Returns DataFrame with columns: ['unit_id'] + feature_cols
    """
    base_signals = infer_base_signals(feature_cols)

    rows = []
    for unit_id, g in df.groupby("unit_id", sort=True):
        g = g.sort_values("cycle")
        last_cycle = int(g["cycle"].iloc[-1])

        tail = g.tail(window)

        feat = {"unit_id": int(unit_id), "cycle": last_cycle}

        for s in base_signals:
            if s not in tail.columns:
                continue
            arr = tail[s].to_numpy()

            feat[f"{s}_rm{window}"] = float(np.mean(arr))
            feat[f"{s}_rs{window}"] = float(np.std(arr, ddof=0))
            feat[f"{s}_sl{window}"] = slope_last_window(arr)

        rows.append(feat)

    feat_df = pd.DataFrame(rows)

    for c in feature_cols:
        if c not in feat_df.columns:
            feat_df[c] = 0.0

    feat_df = feat_df[["unit_id"] + feature_cols]
    return feat_df


def build_tag(fd: str, window: int, cap: int) -> str:
    return f"{fd.lower()}_rf_mss_w{window}_cap{cap}"


def main():
    parser = argparse.ArgumentParser(description="C-MAPSS RUL prediction (last-cycle) using saved model.")
    parser.add_argument("--fd", type=str, default="FD001", help="FD001/FD002/FD003/FD004")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--cap", type=int, default=125)

    parser.add_argument("--data", type=str, default=None, help="Path to test_FD00X.txt. If omitted, auto path is used.")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory (default: models/)")

    parser.add_argument("--unit-id", type=int, default=None, help="If set, predict only this unit_id")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV output path for predictions (all units)")
    args = parser.parse_args()

    fd = args.fd.upper()
    if fd not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError("fd must be one of: FD001, FD002, FD003, FD004")

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "raw" / "cmapss" / "CMAPSSData"

    tag = build_tag(fd, args.window, args.cap)
    models_dir = (root / args.models_dir).resolve()

    model_path = models_dir / f"{tag}.joblib"
    feats_path = models_dir / f"{tag}_features.json"
    meta_path  = models_dir / f"{tag}_meta.json"

    if args.data is None:
        data_path = data_dir / f"test_{fd}.txt"
    else:
        data_path = (root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"Features not found: {feats_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = read_cmapss_txt(data_path)

    feature_cols = json.loads(feats_path.read_text(encoding="utf-8"))

    # If meta exists, prefer window/cap from meta (sanity)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_w = int(meta.get("window", args.window))
        meta_c = int(meta.get("rul_cap", args.cap))
        if meta_w != args.window or meta_c != args.cap:
            print(f"[warn] meta has window/cap={meta_w}/{meta_c}, args={args.window}/{args.cap}. Using meta values.")
            args.window, args.cap = meta_w, meta_c

    model = joblib.load(model_path)

    feat_df = make_features_last_cycle(df, feature_cols, window=int(args.window))

    if args.unit_id is not None:
        feat_df = feat_df[feat_df["unit_id"] == args.unit_id].copy()
        if feat_df.empty:
            raise ValueError(f"unit_id={args.unit_id} not found in {data_path}")

    X = feat_df[feature_cols].to_numpy()
    pred = model.predict(X)

    # clipping
    pred_raw = np.clip(pred, 0, None)
    pred_cap = np.clip(pred_raw, 0, int(args.cap))

    out_df = pd.DataFrame({
        "unit_id": feat_df["unit_id"].values,
        "pred_rul_raw": pred_raw,
        "pred_rul_cap": pred_cap,
    }).sort_values("unit_id").reset_index(drop=True)

    print("FD    :", fd)
    print("Model :", model_path.relative_to(root))
    print("Data  :", data_path.relative_to(root) if data_path.is_relative_to(root) else data_path)
    print("Window:", int(args.window), "| CAP:", int(args.cap))
    print()
    print(out_df.to_string(index=False))

    if args.out:
        out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
