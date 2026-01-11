import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rul_status(rul_cap_value: float, red_thr: float, yellow_thr: float) -> str:
    if rul_cap_value <= red_thr:
        return "RED"
    if rul_cap_value <= yellow_thr:
        return "YELLOW"
    return "GREEN"


def severity(s: str) -> int:
    # worst = max(severity)
    return {"GREEN": 0, "YELLOW": 1, "RED": 2}[s]


def load_gt_rul(gt_path: Path, cap: int) -> pd.DataFrame:
    y = pd.read_csv(gt_path, header=None).iloc[:, 0].astype(float).to_numpy()
    unit_id = np.arange(1, len(y) + 1, dtype=int)
    true_rul_raw = np.clip(y, 0, None)
    true_rul_cap = np.clip(true_rul_raw, 0, cap)
    return pd.DataFrame({"unit_id": unit_id, "true_rul_cap": true_rul_cap})


def main():
    parser = argparse.ArgumentParser(
        description="Tune RED/YELLOW thresholds for SAFE status (worst of point & PI-low), with constraints."
    )
    parser.add_argument("--fd", type=str, required=True, help="FD001/FD002/FD003/FD004")
    parser.add_argument("--cap", type=int, default=125)
    parser.add_argument("--pred", type=str, required=True, help="Path to preds CSV from src.predict")
    parser.add_argument("--data-dir", type=str, default=None, help="Override CMAPSSData dir path")
    parser.add_argument("--pi-col", type=str, required=True, help="PI low column, e.g. pi_p10_cap")
    parser.add_argument("--topk", type=int, default=20)

    # search space
    parser.add_argument("--red-min", type=int, default=10)
    parser.add_argument("--red-max", type=int, default=60)
    parser.add_argument("--yellow-max", type=int, default=125)

    # constraints to avoid degenerate thresholds
    parser.add_argument("--min-gap", type=int, default=10, help="Require yellow_thr - red_thr >= min_gap")
    parser.add_argument(
        "--min-yellow-rate",
        type=float,
        default=0.05,
        help="Require share of TRUE YELLOW >= this value (prevents vanishing yellow zone)",
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
    needed = {"unit_id", "pred_rul_cap", args.pi_col}
    missing = needed - set(pred_df.columns)
    if missing:
        raise ValueError(f"Pred file missing columns: {sorted(missing)}")

    gt_df = load_gt_rul(gt_path, cap=int(args.cap))
    df = gt_df.merge(pred_df, on="unit_id", how="inner")
    if df.empty:
        raise ValueError("No matching unit_id between GT and pred file")

    rows = []

    for red_thr in range(int(args.red_min), int(args.red_max) + 1):
        yellow_start = red_thr + int(args.min_gap)
        for yellow_thr in range(yellow_start, int(args.yellow_max) + 1):
            # TRUE status (these thresholds define the business zones)
            true_status = df["true_rul_cap"].apply(lambda v: rul_status(float(v), red_thr, yellow_thr))

            # constraint: YELLOW must exist in meaningful amount
            yellow_rate = float((true_status == "YELLOW").mean())
            if yellow_rate < float(args.min_yellow_rate):
                continue

            # point status
            point_status = df["pred_rul_cap"].apply(lambda v: rul_status(float(v), red_thr, yellow_thr))

            # PI-low status
            pi_status = df[args.pi_col].apply(lambda v: rul_status(float(v), red_thr, yellow_thr))

            # SAFE = worst(point, pi_low) by severity
            safe_status = [
                "RED" if max(severity(ps), severity(pis)) == 2 else ("YELLOW" if max(severity(ps), severity(pis)) == 1 else "GREEN")
                for ps, pis in zip(point_status.tolist(), pi_status.tolist())
            ]
            safe_status = pd.Series(safe_status, index=df.index)

            # key metrics
            safe_acc = float((safe_status == true_status).mean())

            false_green = int(((true_status != "GREEN") & (safe_status == "GREEN")).sum())
            false_green_rate = false_green / float(len(df))

            # red recall on true RED
            true_red_mask = (true_status == "RED")
            n_true_red = int(true_red_mask.sum())
            if n_true_red > 0:
                safe_red_recall = float(((safe_status == "RED") & true_red_mask).sum()) / n_true_red
                safe_red_caught = float(((safe_status != "GREEN") & true_red_mask).sum()) / n_true_red
            else:
                safe_red_recall = 0.0
                safe_red_caught = 0.0

            # how noisy on true GREEN (unnecessary alarms)
            true_green_mask = (true_status == "GREEN")
            n_true_green = int(true_green_mask.sum())
            if n_true_green > 0:
                green_to_red_rate = float(((safe_status == "RED") & true_green_mask).sum()) / n_true_green
                green_to_non_green_rate = float(((safe_status != "GREEN") & true_green_mask).sum()) / n_true_green
            else:
                green_to_red_rate = 0.0
                green_to_non_green_rate = 0.0

            # keep only safe solutions
            if false_green == 0:
                rows.append(
                    {
                        "red_thr": red_thr,
                        "yellow_thr": yellow_thr,
                        "min_gap": int(args.min_gap),
                        "true_yellow_rate": yellow_rate,
                        "safe_acc": safe_acc,
                        "safe_red_recall": safe_red_recall,
                        "safe_red_caught": safe_red_caught,
                        "green_to_red_rate": green_to_red_rate,
                        "green_to_non_green_rate": green_to_non_green_rate,
                        "false_green_rate": false_green_rate,
                        "pi_col": args.pi_col,
                    }
                )

    print(f"FD: {fd} | CAP: {args.cap} | PI col: {args.pi_col}")
    print(f"Pred: {pred_path}")
    print(f"Constraints: min_gap={args.min_gap}, min_yellow_rate={args.min_yellow_rate}")
    print()

    if not rows:
        print("No candidates found with False GREEN = 0 under given constraints.")
        return

    res = pd.DataFrame(rows)

    # Sort: best accuracy, then best catching RED, then less noise on GREEN
    res = res.sort_values(
        by=["safe_acc", "safe_red_caught", "safe_red_recall", "green_to_non_green_rate", "green_to_red_rate"],
        ascending=[False, False, False, True, True],
    )

    cols = [
        "red_thr",
        "yellow_thr",
        "safe_acc",
        "safe_red_recall",
        "safe_red_caught",
        "true_yellow_rate",
        "green_to_non_green_rate",
        "green_to_red_rate",
        "pi_col",
    ]

    print(f"Top candidates (False GREEN = 0):")
    print(res[cols].head(int(args.topk)).to_string(index=False))


if __name__ == "__main__":
    main()
