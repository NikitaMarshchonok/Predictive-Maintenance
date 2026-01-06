import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rul_status(rul_cap_value: float, red_thr: float = 20.0, yellow_thr: float = 50.0) -> str:
    """Map capped RUL value to traffic-light status."""
    if rul_cap_value <= red_thr:
        return "RED"
    if rul_cap_value <= yellow_thr:
        return "YELLOW"
    return "GREEN"


def select_pi_col(df: pd.DataFrame, prefer_p: int | None = None) -> str | None:
    """
    Select PI-low column.

    - If prefer_p is provided (e.g., 10), return pi_p10_cap if it exists.
    - Otherwise fallback to the smallest available percentile (most conservative).
    """
    cols: list[tuple[int, str]] = []
    for c in df.columns:
        if c.startswith("pi_p") and c.endswith("_cap"):
            # pi_p10_cap -> 10
            try:
                p = int(c.replace("pi_p", "").replace("_cap", ""))
                cols.append((p, c))
            except Exception:
                pass

    if not cols:
        return None

    if prefer_p is not None:
        for p, c in cols:
            if p == prefer_p:
                return c

    cols.sort(key=lambda x: x[0])
    return cols[0][1]


def load_gt_rul(gt_path: Path, cap: int) -> pd.DataFrame:
    """
    CMAPSS RUL_FD00X.txt: one RUL per unit in order 1..N
    """
    y = pd.read_csv(gt_path, header=None).iloc[:, 0].astype(float).to_numpy()
    unit_id = np.arange(1, len(y) + 1, dtype=int)

    true_rul_raw = np.clip(y, 0, None)
    true_rul_cap = np.clip(true_rul_raw, 0, cap)

    return pd.DataFrame({"unit_id": unit_id, "true_rul_cap": true_rul_cap})


def confusion_df(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    return pd.crosstab(y_true, y_pred, rownames=["true_status"], colnames=["pred_status"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate CMAPSS predictions vs ground truth RUL.")
    parser.add_argument("--fd", type=str, required=True, help="FD001/FD002/FD003/FD004")
    parser.add_argument("--cap", type=int, default=125)
    parser.add_argument("--pred", type=str, required=True, help="Path to preds CSV from src.predict")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data dir (root/data/raw/cmapss/CMAPSSData)",
    )
    parser.add_argument("--red-thr", type=float, default=20.0)
    parser.add_argument("--yellow-thr", type=float, default=50.0)

    parser.add_argument(
        "--use-pi-low",
        action="store_true",
        help="Also compute conservative status from PI low bound and SAFE gated status",
    )
    parser.add_argument(
        "--pi-low-p",
        type=int,
        default=None,
        help="Which PI-low percentile to use (e.g., 10 -> pi_p10_cap). If not set, uses smallest available.",
    )
    parser.add_argument(
        "--gate-yellow-thr",
        type=float,
        default=None,
        help="Threshold for downgrading GREEN using PI-low (defaults to --yellow-thr).",
    )

    args = parser.parse_args()

    fd = args.fd.upper()
    if fd not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError("fd must be one of: FD001, FD002, FD003, FD004")

    root = Path(__file__).resolve().parents[1]
    pred_path = (root / args.pred).resolve() if not Path(args.pred).is_absolute() else Path(args.pred)

    if args.data_dir is None:
        data_dir = root / "data" / "raw" / "cmapss" / "CMAPSSData"
    else:
        data_dir = (root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)

    gt_path = data_dir / f"RUL_{fd}.txt"

    if not pred_path.exists():
        raise FileNotFoundError(f"Pred file not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    pred_df = pd.read_csv(pred_path)
    if "unit_id" not in pred_df.columns or "pred_rul_cap" not in pred_df.columns:
        raise ValueError("Pred file must contain columns: unit_id, pred_rul_cap")

    gt_df = load_gt_rul(gt_path, cap=int(args.cap))

    df = gt_df.merge(pred_df, on="unit_id", how="inner")
    if df.empty:
        raise ValueError("No matching unit_id between GT and pred file")

    # statuses (point prediction)
    df["true_status"] = df["true_rul_cap"].apply(lambda v: rul_status(v, args.red_thr, args.yellow_thr))
    df["pred_status"] = df["pred_rul_cap"].apply(lambda v: rul_status(v, args.red_thr, args.yellow_thr))

    # metrics
    y_true = df["true_rul_cap"].to_numpy()
    y_pred = df["pred_rul_cap"].to_numpy()

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    acc = float((df["pred_status"] == df["true_status"]).mean())

    print(f"FD: {fd} | CAP: {args.cap}")
    print(f"Pred: {pred_path}")
    print(f"GT  : {gt_path}")
    print()
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"Status accuracy (RED/YELLOW/GREEN): {acc:.3f}")
    print()

    conf = confusion_df(df["true_status"], df["pred_status"])
    print("Confusion (rows=true, cols=pred):")
    print(conf.to_string())
    print()

    # worst errors
    df["abs_err"] = (df["pred_rul_cap"] - df["true_rul_cap"]).abs()
    worst = df.sort_values("abs_err", ascending=False).head(10)[
        ["unit_id", "true_rul_cap", "pred_rul_cap", "abs_err", "true_status", "pred_status"]
    ]
    print("Top-10 worst errors:")
    print(worst.to_string(index=False))

    # safety focus: false GREEN count
    false_green = df[(df["true_status"] != "GREEN") & (df["pred_status"] == "GREEN")]
    print()
    print(f"False GREEN count (dangerous misses): {len(false_green)}")

    # PI-low conservative + SAFE-gated
    if args.use_pi_low:
        pi_low_col = select_pi_col(df, prefer_p=args.pi_low_p)
        if pi_low_col is None:
            print("\n[info] --use-pi-low set but no PI columns found (pi_pXX_cap). Skipping conservative status.")
            return

        # 1) Pure conservative status from PI low
        df["pred_status_conservative"] = df[pi_low_col].apply(lambda v: rul_status(v, args.red_thr, args.yellow_thr))
        acc_c = float((df["pred_status_conservative"] == df["true_status"]).mean())
        conf_c = confusion_df(df["true_status"], df["pred_status_conservative"])
        false_green_c = df[(df["true_status"] != "GREEN") & (df["pred_status_conservative"] == "GREEN")]

        print("\n--- Conservative status (from PI low bound) ---")
        print(f"PI low column: {pi_low_col}")
        print(f"Status accuracy (conservative): {acc_c:.3f}")
        print("Confusion (rows=true, cols=pred_conservative):")
        print(conf_c.to_string())
        print(f"False GREEN count (conservative): {len(false_green_c)}")

        # 2) SAFE gated: downgrade GREEN if PI low indicates risk
        gate_yellow_thr = args.gate_yellow_thr if args.gate_yellow_thr is not None else args.yellow_thr

        def safe_gate(point_status: str, pi_low_val: float) -> str:
            # Gate ONLY GREEN predictions (tunable via --gate-yellow-thr)
            if point_status == "GREEN":
                if pi_low_val <= args.red_thr:
                    return "RED"
                if pi_low_val <= gate_yellow_thr:
                    return "YELLOW"
                return "GREEN"
            return point_status

        df["pred_status_safe"] = [
            safe_gate(ps, pv) for ps, pv in zip(df["pred_status"].tolist(), df[pi_low_col].tolist())
        ]

        acc_s = float((df["pred_status_safe"] == df["true_status"]).mean())
        conf_s = confusion_df(df["true_status"], df["pred_status_safe"])
        false_green_s = df[(df["true_status"] != "GREEN") & (df["pred_status_safe"] == "GREEN")]

        print("\n--- SAFE gated status (point + PI-low gate) ---")
        print(f"PI low column: {pi_low_col} | gate_yellow_thr: {gate_yellow_thr}")
        print(f"Status accuracy (safe): {acc_s:.3f}")
        print("Confusion (rows=true, cols=pred_safe):")
        print(conf_s.to_string())
        print(f"False GREEN count (safe): {len(false_green_s)}")


if __name__ == "__main__":
    main()
