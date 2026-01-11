import argparse
from pathlib import Path

import numpy as np
import pandas as pd


_STATUS_SEVERITY = {"GREEN": 0, "YELLOW": 1, "RED": 2}


def rul_status(rul_cap_value: float, red_thr: float, yellow_thr: float) -> str:
    if rul_cap_value <= red_thr:
        return "RED"
    if rul_cap_value <= yellow_thr:
        return "YELLOW"
    return "GREEN"


def worst_status(a: str, b: str) -> str:
    return a if _STATUS_SEVERITY.get(a, 0) >= _STATUS_SEVERITY.get(b, 0) else b


def load_gt_rul(gt_path: Path, cap: int) -> pd.DataFrame:
    y = pd.read_csv(gt_path, header=None).iloc[:, 0].astype(float).to_numpy()
    unit_id = np.arange(1, len(y) + 1, dtype=int)
    true_rul_raw = np.clip(y, 0, None)
    true_rul_cap = np.clip(true_rul_raw, 0, cap)
    return pd.DataFrame({"unit_id": unit_id, "true_rul_cap": true_rul_cap})


def find_pi_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c.startswith("pi_p") and c.endswith("_cap"):
            cols.append(c)
    return sorted(cols)


def status_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    acc = float((y_true == y_pred).mean())

    false_green_mask = (y_true != "GREEN") & (y_pred == "GREEN")
    false_green_count = int(false_green_mask.sum())
    false_green_rate = float(false_green_count / max(len(y_true), 1))

    red_mask = (y_true == "RED")
    if int(red_mask.sum()) == 0:
        red_recall = 0.0
        red_caught_rate = 0.0
    else:
        red_recall = float((y_pred[red_mask] == "RED").mean())
        red_caught_rate = float((y_pred[red_mask] != "GREEN").mean())

    return {
        "acc": acc,
        "false_green_count": false_green_count,
        "false_green_rate": false_green_rate,
        "red_recall": red_recall,
        "red_caught_rate": red_caught_rate,
    }


def main():
    p = argparse.ArgumentParser(description="Grid search red/yellow thresholds for SAFE status.")
    p.add_argument("--fd", required=True, type=str, help="FD001/FD002/FD003/FD004")
    p.add_argument("--cap", default=125, type=int)
    p.add_argument("--pred", required=True, type=str, help="Preds CSV path")
    p.add_argument("--data-dir", default=None, type=str)
    p.add_argument("--pi-col", default=None, type=str, help="Which PI low column to use (e.g., pi_p10_cap). If not set, tries to prefer pi_p10_cap then smallest p.")
    p.add_argument("--red-min", default=5, type=int)
    p.add_argument("--red-max", default=35, type=int)
    p.add_argument("--red-step", default=1, type=int)
    p.add_argument("--yellow-min", default=15, type=int)
    p.add_argument("--yellow-max", default=90, type=int)
    p.add_argument("--yellow-step", default=1, type=int)
    p.add_argument("--topk", default=15, type=int)
    args = p.parse_args()

    fd = args.fd.upper()
    if fd not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError("fd must be one of FD001..FD004")

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
        raise ValueError("Pred file must contain unit_id, pred_rul_cap")

    gt_df = load_gt_rul(gt_path, cap=int(args.cap))
    df = gt_df.merge(pred_df, on="unit_id", how="inner")
    if df.empty:
        raise ValueError("No matching unit_id between GT and pred")

    # pick PI column
    pi_cols = find_pi_cols(df)
    if args.pi_col:
        if args.pi_col not in df.columns:
            raise ValueError(f"--pi-col {args.pi_col} not found. Available: {pi_cols}")
        pi_col = args.pi_col
    else:
        # Prefer pi_p10_cap if exists, else pick smallest percentile.
        if "pi_p10_cap" in df.columns:
            pi_col = "pi_p10_cap"
        else:
            # parse p number
            parsed = []
            for c in pi_cols:
                try:
                    pnum = int(c.replace("pi_p", "").replace("_cap", ""))
                    parsed.append((pnum, c))
                except Exception:
                    pass
            if not parsed:
                raise ValueError(f"No PI columns found. Available cols: {list(df.columns)}")
            parsed.sort(key=lambda x: x[0])
            pi_col = parsed[0][1]

    # true statuses are always computed from true_rul_cap with same thresholds as the business policy
    results = []

    red_values = list(range(args.red_min, args.red_max + 1, args.red_step))
    yellow_values = list(range(args.yellow_min, args.yellow_max + 1, args.yellow_step))

    for red_thr in red_values:
        for yellow_thr in yellow_values:
            if yellow_thr <= red_thr:
                continue

            true_status = df["true_rul_cap"].apply(lambda v: rul_status(v, red_thr, yellow_thr))
            point_status = df["pred_rul_cap"].apply(lambda v: rul_status(v, red_thr, yellow_thr))
            pi_status = df[pi_col].apply(lambda v: rul_status(v, red_thr, yellow_thr))

            # SAFE: worst of point & PI-low status
            safe_status = [worst_status(a, b) for a, b in zip(point_status.tolist(), pi_status.tolist())]

            m_safe = status_metrics(true_status, pd.Series(safe_status))
            # keep only safe solutions with zero false GREEN
            if m_safe["false_green_count"] == 0:
                results.append({
                    "red_thr": red_thr,
                    "yellow_thr": yellow_thr,
                    "safe_acc": m_safe["acc"],
                    "safe_red_recall": m_safe["red_recall"],
                    "safe_red_caught": m_safe["red_caught_rate"],
                    "pi_col": pi_col,
                })

    if not results:
        print(f"No thresholds found with False GREEN = 0 using {pi_col}. Try widening ranges.")
        return

    out = pd.DataFrame(results).sort_values(["safe_acc", "safe_red_caught"], ascending=[False, False]).head(args.topk)

    print(f"FD: {fd} | CAP: {args.cap} | PI col: {pi_col}")
    print(f"Pred: {pred_path}")
    print("\nTop candidates (False GREEN = 0):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
