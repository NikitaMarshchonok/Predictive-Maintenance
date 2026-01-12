import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rul_status(rul_cap_value: float, red_thr: float = 20.0, yellow_thr: float = 50.0) -> str:
    if rul_cap_value <= red_thr:
        return "RED"
    if rul_cap_value <= yellow_thr:
        return "YELLOW"
    return "GREEN"


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


def red_recall_and_caught_rate(df: pd.DataFrame, pred_col: str) -> tuple[float, float]:
    """
    RED recall: P(pred=RED | true=RED)
    RED caught rate: P(pred!=GREEN | true=RED)
    """
    true_red = df[df["true_status"] == "RED"]
    if len(true_red) == 0:
        return 0.0, 0.0

    pred = true_red[pred_col]
    red_recall = float((pred == "RED").mean())
    red_caught = float((pred != "GREEN").mean())
    return red_recall, red_caught


def false_green_stats(df: pd.DataFrame, pred_col: str) -> tuple[int, float]:
    false_green = df[(df["true_status"] != "GREEN") & (df[pred_col] == "GREEN")]
    cnt = int(len(false_green))
    rate = float(cnt / len(df)) if len(df) else 0.0
    return cnt, rate


def print_block(title: str, df: pd.DataFrame, pred_col: str, pred_name: str):
    acc = float((df[pred_col] == df["true_status"]).mean())
    fg_cnt, fg_rate = false_green_stats(df, pred_col)
    rr, rc = red_recall_and_caught_rate(df, pred_col)

    print(f"\n--- {title} ---")
    print(f"Status accuracy: {acc:.3f}")
    print(f"False GREEN count: {fg_cnt} (rate={fg_rate:.3f})")
    print(f"RED recall (pred=RED | true=RED): {rr:.3f}")
    print(f"RED caught rate (pred!=GREEN | true=RED): {rc:.3f}")

    conf = confusion_df(df["true_status"], df[pred_col], pred_name=pred_name)
    print("Confusion (rows=true, cols=pred_status):")
    print(conf.to_string())


def main():
    parser = argparse.ArgumentParser(description="Evaluate CMAPSS predictions vs ground truth RUL.")
    parser.add_argument("--fd", type=str, required=True, help="FD001/FD002/FD003/FD004")
    parser.add_argument("--cap", type=int, default=125)
    parser.add_argument("--pred", type=str, required=True, help="Path to preds CSV from src.predict")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data dir (root/data/raw/cmapss/CMAPSSData)")

    # Business thresholds for status mapping
    parser.add_argument("--red-thr", type=float, default=20.0)
    parser.add_argument("--yellow-thr", type=float, default=50.0)

    # PI usage
    parser.add_argument("--use-pi-low", action="store_true", help="Compute PI-low based safety statuses")
    parser.add_argument("--pi-col", type=str, default=None, help="Explicit PI-low column, e.g. pi_p10_cap")

    # Gate thresholds (separate from business thresholds!)
    parser.add_argument("--gate-red-thr", type=float, default=None, help="Gate RED threshold for PI-low (default=--red-thr)")
    parser.add_argument("--gate-yellow-thr", type=float, default=None, help="Gate YELLOW threshold for PI-low (default=--yellow-thr)")

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

    # statuses (business thresholds!)
    df["true_status"] = df["true_rul_cap"].apply(lambda v: rul_status(v, args.red_thr, args.yellow_thr))
    df["pred_status"] = df["pred_rul_cap"].apply(lambda v: rul_status(v, args.red_thr, args.yellow_thr))

    # regression metrics (always on cap space)
    mae = float(np.mean(np.abs(df["pred_rul_cap"].to_numpy() - df["true_rul_cap"].to_numpy())))
    rmse = float(np.sqrt(np.mean((df["pred_rul_cap"].to_numpy() - df["true_rul_cap"].to_numpy()) ** 2)))

    print(f"FD: {fd} | CAP: {args.cap}")
    print(f"Pred: {pred_path}")
    print(f"GT  : {gt_path}")
    print()
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")

    # Point block
    print_block("Point status (from pred_rul_cap)", df, "pred_status", "pred_status")

    # worst errors (based on point prediction)
    df["abs_err"] = (df["pred_rul_cap"] - df["true_rul_cap"]).abs()
    worst = df.sort_values("abs_err", ascending=False).head(10)[
        ["unit_id", "true_rul_cap", "pred_rul_cap", "abs_err", "true_status", "pred_status"]
    ]
    print("\nTop-10 worst errors:")
    print(worst.to_string(index=False))

    if not args.use_pi_low:
        return

    # choose PI-low column
    pi_low_col = args.pi_col if args.pi_col else find_pi_low_col(df)
    if pi_low_col is None or pi_low_col not in df.columns:
        print("\n[info] --use-pi-low set but no PI columns found (pi_pXX_cap). Skipping PI-based blocks.")
        return

    # Conservative: status from PI-low using BUSINESS thresholds
    df["pred_conservative"] = df[pi_low_col].apply(lambda v: rul_status(v, args.red_thr, args.yellow_thr))
    print_block(f"Conservative status (from PI low: {pi_low_col})", df, "pred_conservative", "pred_conservative")

    # SAFE worst-of (severity max between point and PI-low status)
    severity = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    inv_sev = {v: k for k, v in severity.items()}

    def worst_of(point_s: str, pi_s: str) -> str:
        return inv_sev[max(severity[point_s], severity[pi_s])]

    df["pred_safe_worst"] = [worst_of(p, c) for p, c in zip(df["pred_status"], df["pred_conservative"])]
    print_block(f"SAFE worst-of (point vs PI-low={pi_low_col})", df, "pred_safe_worst", "pred_safe_worst")

    # SAFE gated with SEPARATE gate thresholds
    gate_red = args.gate_red_thr if args.gate_red_thr is not None else args.red_thr
    gate_yellow = args.gate_yellow_thr if args.gate_yellow_thr is not None else args.yellow_thr
    if gate_red >= gate_yellow:
        raise ValueError("Gate thresholds must satisfy gate_red_thr < gate_yellow_thr")

    def safe_gate(point_status: str, pi_low_val: float) -> str:
        # Only downgrade when PI-low is clearly risky, using gate thresholds (NOT business thresholds)
        if point_status == "GREEN":
            if pi_low_val <= gate_red:
                return "RED"
            if pi_low_val <= gate_yellow:
                return "YELLOW"
            return "GREEN"

        # Optional: escalate YELLOW->RED if PI-low is RED by gate
        if point_status == "YELLOW" and pi_low_val <= gate_red:
            return "RED"

        return point_status

    df["pred_safe_gate"] = [
        safe_gate(ps, pv) for ps, pv in zip(df["pred_status"].tolist(), df[pi_low_col].tolist())
    ]
    print_block(
        f"SAFE gated (gate_red={gate_red:g}, gate_yellow={gate_yellow:g}, PI-low={pi_low_col})",
        df,
        "pred_safe_gate",
        "pred_safe_gate",
    )


if __name__ == "__main__":
    main()
