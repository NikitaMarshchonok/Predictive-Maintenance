import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# Safety / status helpers
# =========================================================

def rul_status(rul_cap_value: float, red_thr: float = 20.0, yellow_thr: float = 50.0) -> str:
    """Map RUL(cap) -> traffic-light status."""
    v = float(rul_cap_value)
    if v <= float(red_thr):
        return "RED"
    if v <= float(yellow_thr):
        return "YELLOW"
    return "GREEN"


def status_severity(s: str) -> int:
    """Lower is worse."""
    m = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    return m.get(str(s), 99)


def worst_status(a: str, b: str) -> str:
    """Return more conservative (worse) status."""
    return a if status_severity(a) <= status_severity(b) else b


def find_pi_low_col(df: pd.DataFrame) -> str | None:
    """
    Find lowest-percentile PI column like pi_p10_cap / pi_p5_cap.
    Returns the column name with smallest p.
    """
    cols = []
    for c in df.columns:
        if c.startswith("pi_p") and c.endswith("_cap"):
            try:
                p = int(c.replace("pi_p", "").replace("_cap", ""))
                cols.append((p, c))
            except Exception:
                pass
    if not cols:
        return None
    cols.sort(key=lambda x: x[0])
    return cols[0][1]


def safe_gate_status(point_status: str, pi_low_val: float, gate_red_thr: float, gate_yellow_thr: float) -> str:
    """
    SAFE gated policy:
    - If point is GREEN, forbid GREEN if PI-low indicates risk (using gate thresholds).
    - Optional: if point is YELLOW but PI-low is strongly RED (<= gate_red_thr), escalate to RED.
    - RED stays RED.
    """
    ps = str(point_status)
    pv = float(pi_low_val)

    if ps == "GREEN":
        if pv <= float(gate_red_thr):
            return "RED"
        if pv <= float(gate_yellow_thr):
            return "YELLOW"
        return "GREEN"

    if ps == "YELLOW" and pv <= float(gate_red_thr):
        return "RED"

    return ps


# =========================================================
# Model helpers
# =========================================================

def rf_prediction_interval(model, X, low=10, high=90):
    """
    Prediction interval for RandomForest via per-tree predictions.
    Returns (p_low, p50, p_high) for each row in X.
    """
    if not hasattr(model, "estimators_"):
        return None, None, None

    preds = np.column_stack([est.predict(X) for est in model.estimators_])  # (n_samples, n_estimators)
    p_low = np.percentile(preds, low, axis=1)
    p50 = np.percentile(preds, 50, axis=1)
    p_high = np.percentile(preds, high, axis=1)
    return p_low, p50, p_high


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

    # Ensure all expected feature columns exist
    for c in feature_cols:
        if c not in feat_df.columns:
            feat_df[c] = 0.0

    feat_df = feat_df[["unit_id"] + feature_cols]
    return feat_df


def build_tag(fd: str, window: int, cap: int) -> str:
    return f"{fd.lower()}_rf_mss_w{window}_cap{cap}"


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="C-MAPSS RUL prediction (last-cycle) using saved model.")

    parser.add_argument("--fd", type=str, default="FD001", help="FD001/FD002/FD003/FD004")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--cap", type=int, default=125)

    parser.add_argument("--data", type=str, default=None,
                        help="Path to test_FD00X.txt. If omitted, auto path is used.")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory (default: models/)")

    parser.add_argument("--unit-id", type=int, default=None, help="If set, predict only this unit_id")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV output path for predictions (all units)")

    # PI settings
    parser.add_argument("--pi-low", type=float, default=10.0, help="Prediction interval lower percentile (RF only)")
    parser.add_argument("--pi-high", type=float, default=90.0, help="Prediction interval upper percentile (RF only)")

    # Base business thresholds for status labels
    parser.add_argument("--red-thr", type=float, default=20.0, help="RED threshold for CAP RUL")
    parser.add_argument("--yellow-thr", type=float, default=50.0, help="YELLOW threshold for CAP RUL")

    # Status policy (what goes into column `status`)
    parser.add_argument(
        "--status-policy",
        type=str,
        default="point",
        choices=["point", "conservative", "worst", "gate"],
        help="Which status to output in `status` column."
    )

    # PI column selection + gate thresholds
    parser.add_argument(
        "--pi-col",
        type=str,
        default=None,
        help="Which PI low column to use (e.g., pi_p10_cap). If not set, auto-detect lowest pi_pXX_cap."
    )
    # IMPORTANT: defaults tuned nicely for FD002 in your experiments
    parser.add_argument(
        "--gate-red-thr",
        type=float,
        default=8.0,
        help="Gate threshold for forcing RED when point is GREEN (uses PI-low)."
    )
    parser.add_argument(
        "--gate-yellow-thr",
        type=float,
        default=37.0,
        help="Gate threshold for forcing YELLOW when point is GREEN (uses PI-low)."
    )

    args = parser.parse_args()

    fd = args.fd.upper()
    if fd not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError("fd must be one of: FD001, FD002, FD003, FD004")

    if args.pi_high <= args.pi_low:
        raise ValueError("--pi-high must be greater than --pi-low (e.g., 90 > 10)")

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "raw" / "cmapss" / "CMAPSSData"

    tag = build_tag(fd, args.window, args.cap)
    models_dir = (root / args.models_dir).resolve()

    model_path = models_dir / f"{tag}.joblib"
    feats_path = models_dir / f"{tag}_features.json"
    meta_path = models_dir / f"{tag}_meta.json"

    if args.data is None:
        data_path = data_dir / f"test_{fd}.txt"
    else:
        p = Path(args.data)
        data_path = (root / p).resolve() if not p.is_absolute() else p

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"Features not found: {feats_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = read_cmapss_txt(data_path)
    feature_cols = json.loads(feats_path.read_text(encoding="utf-8"))

    # Meta sanity: prefer window/cap from meta if exists
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

    # PI (RF only)
    pi_low, pi_med, pi_high = rf_prediction_interval(model, X, low=float(args.pi_low), high=float(args.pi_high))
    has_pi = pi_low is not None

    out_df = pd.DataFrame(
        {
            "unit_id": feat_df["unit_id"].values,
            "pred_rul_raw": pred_raw,
            "pred_rul_cap": pred_cap,
        }
    ).sort_values("unit_id").reset_index(drop=True)

    # Point status
    out_df["status_point"] = [rul_status(v, args.red_thr, args.yellow_thr) for v in out_df["pred_rul_cap"].tolist()]

    # Add PI columns + safety statuses if PI exists
    if has_pi:
        pi_low_raw = np.clip(pi_low, 0, None)
        pi_med_raw = np.clip(pi_med, 0, None)
        pi_high_raw = np.clip(pi_high, 0, None)

        pi_low_cap = np.clip(pi_low_raw, 0, int(args.cap))
        pi_med_cap = np.clip(pi_med_raw, 0, int(args.cap))
        pi_high_cap = np.clip(pi_high_raw, 0, int(args.cap))

        low_name = f"pi_p{int(args.pi_low)}_cap"
        high_name = f"pi_p{int(args.pi_high)}_cap"

        out_df[low_name] = pi_low_cap
        out_df["pi_p50_cap"] = pi_med_cap
        out_df[high_name] = pi_high_cap

        out_df["pi_width_cap"] = (out_df[high_name] - out_df[low_name]).round(3)
        out_df["risk_score"] = ((out_df["pi_width_cap"] / float(args.cap)).clip(0, 1)).round(3)

        # Choose PI-low column for policies
        pi_col = args.pi_col
        if pi_col is None:
            pi_col = find_pi_low_col(out_df)
        if pi_col is None or pi_col not in out_df.columns:
            # Fallback: use the generated low_name
            pi_col = low_name

        out_df["status_conservative"] = [
            rul_status(v, args.red_thr, args.yellow_thr) for v in out_df[pi_col].tolist()
        ]

        out_df["status_worst"] = [
            worst_status(a, b) for a, b in zip(out_df["status_point"].tolist(), out_df["status_conservative"].tolist())
        ]

        out_df["status_gate"] = [
            safe_gate_status(ps, pv, args.gate_red_thr, args.gate_yellow_thr)
            for ps, pv in zip(out_df["status_point"].tolist(), out_df[pi_col].tolist())
        ]

        out_df["pi_low_col_used"] = pi_col
        out_df["gate_red_used"] = float(args.gate_red_thr)
        out_df["gate_yellow_used"] = float(args.gate_yellow_thr)

    else:
        out_df["pi_width_cap"] = np.nan
        out_df["risk_score"] = np.nan
        out_df["status_conservative"] = np.nan
        out_df["status_worst"] = np.nan
        out_df["status_gate"] = np.nan
        out_df["pi_low_col_used"] = np.nan
        out_df["gate_red_used"] = np.nan
        out_df["gate_yellow_used"] = np.nan

    # Set main status according to policy
    policy = args.status_policy
    if policy == "point":
        out_df["status"] = out_df["status_point"]
    elif policy == "conservative":
        out_df["status"] = out_df["status_conservative"] if has_pi else out_df["status_point"]
    elif policy == "worst":
        out_df["status"] = out_df["status_worst"] if has_pi else out_df["status_point"]
    elif policy == "gate":
        out_df["status"] = out_df["status_gate"] if has_pi else out_df["status_point"]

    def _pretty_path(p: Path) -> str:
        try:
            return str(p.relative_to(root))
        except Exception:
            return str(p)

    # Print summary header
    print("FD    :", fd)
    print("Model :", _pretty_path(model_path))
    print("Data  :", _pretty_path(data_path))
    print("Window:", int(args.window), "| CAP:", int(args.cap))
    print("Base thresholds:", f"RED<={args.red_thr}, YELLOW<={args.yellow_thr}, else GREEN")
    print("Status policy  :", policy)

    if has_pi:
        print("Prediction interval:", f"p{args.pi_low:.0f}/p50/p{args.pi_high:.0f} (RF trees, clipped to CAP)")
        print("PI-low used     :", out_df['pi_low_col_used'].iloc[0])
        print("Gate thresholds :", f"gate_red={args.gate_red_thr}, gate_yellow={args.gate_yellow_thr}")
    else:
        print("Prediction interval: n/a (model has no estimators_)")

    print()

    # Avoid printing huge tables unless single unit
    if args.unit_id is not None or len(out_df) <= 50:
        print(out_df.to_string(index=False))
    else:
        print(out_df.head(20).to_string(index=False))
        print(f"\n[info] showing first 20/{len(out_df)} rows. Use --out to save full CSV.")

    # TOP-10 critical units (batch mode)
    if args.unit_id is None and len(out_df) > 1:
        severity_rank = {"RED": 0, "YELLOW": 1, "GREEN": 2}

        crit = out_df.copy()
        crit["severity"] = crit["status"].map(severity_rank).fillna(99).astype(int)

        # Sort: RED first, then higher risk_score, then smaller predicted RUL
        sort_cols = ["severity"]
        asc = [True]

        if "risk_score" in crit.columns:
            sort_cols.append("risk_score")
            asc.append(False)

        sort_cols.append("pred_rul_cap")
        asc.append(True)

        crit = crit.sort_values(by=sort_cols, ascending=asc).head(10)

        cols = ["unit_id", "status", "pred_rul_cap"]
        if has_pi:
            cols += ["pi_width_cap", "risk_score", "pi_low_col_used"]

        print("\nTOP-10 critical units (by status + uncertainty + low RUL):")
        print(crit[cols].to_string(index=False))

    # Save
    if args.out:
        out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
