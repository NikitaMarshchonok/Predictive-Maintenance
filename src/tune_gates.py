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


def main():
    parser = argparse.ArgumentParser(description="Tune PI-low gate thresholds for SAFE gating (business thresholds fixed).")
    parser.add_argument("--fd", required=True, type=str)
    parser.add_argument("--cap", default=125, type=int)
    parser.add_argument("--pred", required=True, type=str)
    parser.add_argument("--data-dir", type=str, default=None)

    parser.add_argument("--pi-col", required=True, type=str, help="PI-low column, e.g. pi_p10_cap")

    # Business thresholds (fixed semantics)
    parser.add_argument("--base-red-thr", default=20.0, type=float)
    parser.add_argument("--base-yellow-thr", default=50.0, type=float)

    # Search grid
    parser.add_argument("--gate-red-min", default=5.0, type=float)
    parser.add_argument("--gate-red-max", default=25.0, type=float)
    parser.add_argument("--gate-red-step", default=1.0, type=float)

    parser.add_argument("--gate-yellow-min", default=25.0, type=float)
    parser.add_argument("--gate-yellow-max", default=60.0, type=float)
    parser.add_argument("--gate-yellow-step", default=1.0, type=float)

    # Constraints
    parser.add_argument("--max-false-green", default=0, type=int)
    parser.add_argument("--min-red-caught", default=0.95, type=float)

    parser.add_argument("--topk", default=20, type=int)
    args = parser.parse_args()

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

    pred_df = pd.read_csv(pred_path)
    if args.pi_col not in pred_df.columns:
        raise ValueError(f"PI column not found in preds: {args.pi_col}")

    gt_df = load_gt_rul(gt_path, cap=args.cap)
    df = gt_df.merge(pred_df, on="unit_id", how="inner")
    if df.empty:
        raise ValueError("No matching unit_id between GT and pred file")

    # Fixed business statuses
    df["true_status"] = df["true_rul_cap"].apply(lambda v: rul_status(v, args.base_red_thr, args.base_yellow_thr))
    df["point_status"] = df["pred_rul_cap"].apply(lambda v: rul_status(v, args.base_red_thr, args.base_yellow_thr))

    pi = df[args.pi_col].astype(float).to_numpy()

    results = []
    red_vals = np.arange(args.gate_red_min, args.gate_red_max + 1e-9, args.gate_red_step)
    yel_vals = np.arange(args.gate_yellow_min, args.gate_yellow_max + 1e-9, args.gate_yellow_step)

    for gate_red in red_vals:
        for gate_yel in yel_vals:
            if gate_red >= gate_yel:
                continue

            # apply gate
            safe = []
            for ps, pv in zip(df["point_status"].tolist(), pi.tolist()):
                if ps == "GREEN":
                    if pv <= gate_red:
                        safe.append("RED")
                    elif pv <= gate_yel:
                        safe.append("YELLOW")
                    else:
                        safe.append("GREEN")
                elif ps == "YELLOW":
                    if pv <= gate_red:
                        safe.append("RED")
                    else:
                        safe.append("YELLOW")
                else:
                    safe.append("RED")

            df_tmp = df.copy()
            df_tmp["safe_gate"] = safe

            fg = false_green(df_tmp, "safe_gate")
            if fg > args.max_false_green:
                continue

            rc = red_caught(df_tmp, "safe_gate")
            if rc < args.min_red_caught:
                continue

            acc = float((df_tmp["safe_gate"] == df_tmp["true_status"]).mean())
            rr = red_recall(df_tmp, "safe_gate")

            # how many point-GREEN got downgraded (false alarms pressure)
            point_green = df_tmp["point_status"] == "GREEN"
            downgraded = float((point_green & (df_tmp["safe_gate"] != "GREEN")).mean())

            results.append(
                {
                    "gate_red": float(gate_red),
                    "gate_yellow": float(gate_yel),
                    "safe_acc": acc,
                    "false_green": fg,
                    "red_recall": rr,
                    "red_caught": rc,
                    "downgrade_rate_from_point_green": downgraded,
                }
            )

    if not results:
        print("No candidates found under constraints.")
        return

    out = pd.DataFrame(results).sort_values(["safe_acc", "downgrade_rate_from_point_green"], ascending=[False, True])
    print(f"FD: {fd} | CAP: {args.cap} | PI col: {args.pi_col}")
    print(f"Pred: {pred_path}")
    print(
        f"Business thresholds fixed: red={args.base_red_thr:g}, yellow={args.base_yellow_thr:g} | "
        f"Constraints: max_false_green={args.max_false_green}, min_red_caught={args.min_red_caught:g}"
    )
    print("\nTop candidates:")
    print(out.head(args.topk).to_string(index=False))


if __name__ == "__main__":
    main()