import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Status helpers
# =========================

def rul_status(rul_cap_value: float, red_thr: float = 20.0, yellow_thr: float = 50.0) -> str:
    if rul_cap_value <= red_thr:
        return "RED"
    if rul_cap_value <= yellow_thr:
        return "YELLOW"
    return "GREEN"


_STATUS_SEVERITY = {"GREEN": 0, "YELLOW": 1, "RED": 2}


def worst_status(a: str, b: str) -> str:
    """Return more severe status among a,b."""
    return a if _STATUS_SEVERITY.get(a, 0) >= _STATUS_SEVERITY.get(b, 0) else b


def find_pi_low_col(df: pd.DataFrame) -> str | None:
    """
    Find lowest-percentile PI column like pi_p10_cap / pi_p5_cap.
    Returns the column name with smallest p.
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
    cols.sort(key=lambda x: x[0])
    return cols[0][1]


# =========================
# Data loaders
# =========================

def load_gt_rul(gt_path: Path, cap: int) -> pd.DataFrame:
    """
    CMAPSS RUL_FD00X.txt: one RUL per unit in order 1..N
    """
    y = pd.read_csv(gt_path, header=None).iloc[:, 0].astype(float).to_numpy()
    unit_id = np.arange(1, len(y) + 1, dtype=int)
    true_rul_raw = np.clip(y, 0, None)
    true_rul_cap = np.clip(true_rul_raw, 0, cap)
    return pd.DataFrame({"unit_id": unit_id, "true_rul_cap": true_rul_cap})


def confusion_df(y_true: pd.Series, y_pred: pd.Series, pred_name: str = "pred_status") -> pd.DataFrame:
    return pd.crosstab(y_true, y_pred, rownames=["true_status"], colnames=[pred_name])


# =========================
# Metrics
# =========================

def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return mae, rmse


def status_metrics(df: pd.DataFrame, pred_status_col: str) -> dict:
    """
    Returns:
      - status_accuracy
      - false_green_count / rate (dangerous misses)
      - red_recall (RED predicted as RED)
      - red_caught_rate (true RED predicted as RED or YELLOW; i.e., NOT GREEN)
    """
    y_true = df["true_status"]
    y_pred = df[pred_status_col]

    acc = float((y_true == y_pred).mean())

    false_green = df[(df["true_status"] != "GREEN") & (df[pred_status_col] == "GREEN")]
    false_green_count = int(len(false_green))
    false_green_rate = float(false_green_count / max(len(df), 1))

    # True RED rows
    red_rows = df[df["true_status"] == "RED"]
    if len(red_rows) == 0:
        red_recall = 0.0
        red_caught_rate = 0.0
    else:
        red_recall = float((red_rows[pred_status_col] == "RED").mean())
        red_caught_rate = float((red_rows[pred_status_col] != "GREEN").mean())

    return {
        "status_accuracy": acc,
        "false_green_count": false_green_count,
        "false_green_rate": false_green_rate,
        "red_recall": red_recall,
        "red_caught_rate": red_caught_rate,
    }


def print_status_block(title: str, df: pd.DataFrame, pred_col: str, pred_label: str):
    m = status_metrics(df, pred_col)
    print(f"\n--- {title} ---")
    print(f"Status accuracy: {m['status_accuracy']:.3f}")
    print(f"False GREEN count: {m['false_green_count']} (rate={m['false_green_rate']:.3f})")
    print(f"RED recall (pred=RED | true=RED): {m['red_recall']:.3f}")
    print(f"RED caught rate (pred!=GREEN | true=RED): {m['red_caught_rate']:.3f}")
    print(f"Confusion (rows=true, cols={pred_label}):")
    print(confusion_df(df["true_status"], df[pred_col], pred_name=pred_label).to_string())


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Evaluate CMAPSS predictions vs ground truth RUL.")
    parser.add_argument("--fd", type=str, required=True, help="FD001/FD002/FD003/FD004")
    parser.add_argument("--cap", type=int, default=125)
    parser.add_argument("--pred", type=str, required=True, help="Path to preds CSV from src.predict")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data dir (root/data/raw/cmapss/CMAPSSData)")
    parser.add_argument("--red-thr", type=float, default=20.0)
    parser.add_argument("--yellow-thr", type=float, default=50.0)
    parser.add_argument("--use-pi-low", action="store_true", help="Also compute conservative + SAFE-gated status from PI low bound")
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

    # Base statuses from point estimate
    df["true_status"] = df["true_rul_cap"].apply(lambda v: rul_status(v, args.red_thr, args.yellow_thr))
    df["pred_status"] = df["pred_rul_cap"].apply(lambda v: rul_status(v, args.red_thr, args.yellow_thr))

    # Regression metrics
    y_true = df["true_rul_cap"].to_numpy(dtype=float)
    y_pred = df["pred_rul_cap"].to_numpy(dtype=float)
    mae, rmse = mae_rmse(y_true, y_pred)

    print(f"FD: {fd} | CAP: {args.cap}")
    print(f"Pred: {pred_path}")
    print(f"GT  : {gt_path}")
    print()
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")

    # Point status block
    print_status_block("Point status (from pred_rul_cap)", df, "pred_status", "pred_status")

    # Worst errors table
    df["abs_err"] = (df["pred_rul_cap"] - df["true_rul_cap"]).abs()
    worst = df.sort_values("abs_err", ascending=False).head(10)[
        ["unit_id", "true_rul_cap", "pred_rul_cap", "abs_err", "true_status", "pred_status"]
    ]
    print("\nTop-10 worst errors:")
    print(worst.to_string(index=False))

    # PI-low modes
    if args.use_pi_low:
        pi_low_col = find_pi_low_col(df)
        if pi_low_col is None:
            print("\n[info] --use-pi-low set but no PI columns found (pi_pXX_cap). Skipping PI-based statuses.")
            return

        # 1) Conservative status: status only from PI-low
        df["pred_status_conservative"] = df[pi_low_col].apply(lambda v: rul_status(v, args.red_thr, args.yellow_thr))
        print_status_block(
            f"Conservative status (from PI low: {pi_low_col})",
            df,
            "pred_status_conservative",
            "pred_conservative",
        )

        # 2) SAFE gated: final status is "worse" of point and PI-low statuses
        #    => never allow GREEN if PI-low indicates YELLOW/RED risk.
        df["pred_status_safe"] = [
            worst_status(p, c) for p, c in zip(df["pred_status"].tolist(), df["pred_status_conservative"].tolist())
        ]

        print_status_block(
            f"SAFE gated status (worst of point & PI-low={pi_low_col})",
            df,
            "pred_status_safe",
            "pred_safe",
        )


if __name__ == "__main__":
    main()
