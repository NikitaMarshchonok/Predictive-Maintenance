import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", type=str, default="FD002")
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--pred", type=str, default="reports/preds_fd002.csv")
    ap.add_argument("--red-thr", type=float, default=20.0)
    ap.add_argument("--yellow-thr", type=float, default=50.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    fd = args.fd.upper()

    pred_path = (root / args.pred).resolve() if not Path(args.pred).is_absolute() else Path(args.pred)
    gt_path = root / "data" / "raw" / "cmapss" / "CMAPSSData" / f"RUL_{fd}.txt"

    if not pred_path.exists():
        raise FileNotFoundError(f"Pred CSV not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT RUL not found: {gt_path}")

    pred = pd.read_csv(pred_path)
    gt = pd.read_csv(gt_path, sep=r"\s+", header=None, names=["true_rul"])

    # В CMAPSS RUL file идет по unit_id 1..N
    gt["unit_id"] = np.arange(1, len(gt) + 1)

    df = pred.merge(gt, on="unit_id", how="inner")
    if df.empty:
        raise ValueError("Merge is empty. Check unit_id alignment.")

    df["true_rul_raw"] = df["true_rul"].clip(lower=0)
    df["true_rul_cap"] = df["true_rul_raw"].clip(upper=args.cap)

    y_true = df["true_rul_cap"].to_numpy(dtype=float)
    y_pred = df["pred_rul_cap"].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def status(v):
        if v <= args.red_thr:
            return "RED"
        if v <= args.yellow_thr:
            return "YELLOW"
        return "GREEN"

    df["true_status"] = df["true_rul_cap"].apply(status)
    df["pred_status"] = df["pred_rul_cap"].apply(status)

    acc = float((df["true_status"] == df["pred_status"]).mean())

    print("FD:", fd, "| CAP:", args.cap)
    print("Pred:", pred_path)
    print("GT  :", gt_path)
    print()
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"Status accuracy (RED/YELLOW/GREEN): {acc:.3f}")
    print()
    print("Confusion (rows=true, cols=pred):")
    print(pd.crosstab(df["true_status"], df["pred_status"]))

    worst = df.assign(abs_err=(df["true_rul_cap"] - df["pred_rul_cap"]).abs()) \
              .sort_values("abs_err", ascending=False).head(10)[
                  ["unit_id", "true_rul_cap", "pred_rul_cap", "abs_err", "true_status", "pred_status"]
              ]
    print("\nTop-10 worst errors:")
    print(worst.to_string(index=False))

if __name__ == "__main__":
    main()
