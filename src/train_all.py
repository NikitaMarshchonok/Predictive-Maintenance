import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sklearn


BASE_COLS = ["unit_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]


def load_cmapss_txt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=BASE_COLS, engine="python")


def add_rul_to_train(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    out = train_df.merge(max_cycle, on="unit_id", how="left")
    out["RUL"] = out["max_cycle"] - out["cycle"]
    return out.drop(columns=["max_cycle"])


def load_rul_file(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, 0].to_numpy()


def make_split_by_units(unit_ids: np.ndarray, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(unit_ids)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val_units = set(uniq[:n_val].tolist())
    train_mask = np.array([u not in val_units for u in unit_ids])
    val_mask = ~train_mask
    return train_mask, val_mask, sorted(list(val_units))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = (y_pred - y_true).astype(float)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


def _make_slope_func(window: int):
    t = np.arange(window, dtype=float)
    t0 = t - t.mean()
    denom = np.sum(t0 ** 2)

    def slope(x: np.ndarray) -> float:
        return float(np.dot(t0, x) / denom) if denom != 0 else 0.0

    return slope


def build_rolling_features(df: pd.DataFrame, sensors: list, window: int) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("unit_id", group_keys=False)
    slope_fn = _make_slope_func(window)

    for s in sensors:
        r = g[s].rolling(window, min_periods=window)
        out[f"{s}_rm{window}"] = r.mean().reset_index(level=0, drop=True)
        out[f"{s}_rs{window}"] = r.std().reset_index(level=0, drop=True)
        out[f"{s}_sl{window}"] = r.apply(slope_fn, raw=True).reset_index(level=0, drop=True)

    return out


def prepare_train_fd(data_dir: Path, fd: str, window: int, drop_sensors: set):
    train_path = data_dir / f"train_{fd}.txt"
    df = load_cmapss_txt(train_path)
    df = add_rul_to_train(df)

    sensors = [s for s in SENSOR_COLS if s not in drop_sensors]
    df_feat = build_rolling_features(df, sensors=sensors, window=window)

    feat_cols = [c for c in df_feat.columns if c.endswith(f"_rm{window}") or c.endswith(f"_rs{window}") or c.endswith(f"_sl{window}")]
    df_feat = df_feat.dropna(subset=feat_cols).reset_index(drop=True)

    X = df_feat[feat_cols].astype(float).to_numpy()
    y = df_feat["RUL"].astype(float).to_numpy()
    unit_ids = df_feat["unit_id"].to_numpy()
    return X, y, unit_ids, feat_cols


def prepare_test_lastcycle_fd(data_dir: Path, fd: str, window: int, drop_sensors: set):
    test_path = data_dir / f"test_{fd}.txt"
    rul_path = data_dir / f"RUL_{fd}.txt"

    df = load_cmapss_txt(test_path)

    last_cycle = df.groupby("unit_id")["cycle"].max().rename("last_cycle")
    df = df.merge(last_cycle, on="unit_id", how="left")
    df = df[df["cycle"] >= (df["last_cycle"] - window + 1)].copy()

    sensors = [s for s in SENSOR_COLS if s not in drop_sensors]
    df_feat = build_rolling_features(df, sensors=sensors, window=window)

    feat_cols = [c for c in df_feat.columns if c.endswith(f"_rm{window}") or c.endswith(f"_rs{window}") or c.endswith(f"_sl{window}")]
    df_feat = df_feat.dropna(subset=feat_cols).copy()
    df_last = df_feat.sort_values(["unit_id", "cycle"]).groupby("unit_id", as_index=False).tail(1)

    X_last = df_last[feat_cols].astype(float).to_numpy()
    unit_ids_last = df_last["unit_id"].to_numpy()

    rul_last = load_rul_file(rul_path)
    y_true_raw = np.array([rul_last[u - 1] for u in unit_ids_last], dtype=float)

    return X_last, unit_ids_last, y_true_raw, feat_cols


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate CMAPSS RUL models for FD001–FD004.")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--cap", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--data-dir", type=str, default="data/raw/cmapss/CMAPSSData")
    parser.add_argument("--drop-sensors", type=str, default="s1,s5,s10,s16,s18,s19")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir
    models_dir = root / "models"
    reports_dir = root / "reports"
    fig_dir = reports_dir / "figures"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    window = args.window
    cap = args.cap
    seed = args.seed
    val_frac = args.val_frac
    drop_sensors = set([s.strip() for s in args.drop_sensors.split(",") if s.strip()])

    fds = ["FD001", "FD002", "FD003", "FD004"]
    rows = []
    t0 = time.time()

    for fd in fds:
        print("\n" + "=" * 60)
        print("FD:", fd)

        X, y_raw, unit_ids, feat_cols = prepare_train_fd(data_dir, fd, window, drop_sensors)
        y_cap = np.minimum(y_raw, cap)

        train_mask, val_mask, val_units = make_split_by_units(unit_ids, val_frac=val_frac, seed=seed)

        X_tr, y_tr = X[train_mask], y_cap[train_mask]
        X_val, y_val = X[val_mask], y_cap[val_mask]

        model = RandomForestRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
            max_features="sqrt",
        )

        model.fit(X_tr, y_tr)
        pred_val = np.clip(model.predict(X_val), 0, cap)

        val_mae = float(mean_absolute_error(y_val, pred_val))
        val_rmse = rmse(y_val, pred_val)

        # fit on full train
        model.fit(X, y_cap)

        X_test, unit_ids_test, y_test_raw, feat_cols_test = prepare_test_lastcycle_fd(data_dir, fd, window, drop_sensors)
        if feat_cols_test != feat_cols:
            raise ValueError(f"[{fd}] Feature mismatch between train/test!")

        y_test_cap = np.minimum(y_test_raw, cap)
        pred_test_raw = np.clip(model.predict(X_test), 0, None)
        pred_test_cap = np.clip(pred_test_raw, 0, cap)

        test_mae_cap = float(mean_absolute_error(y_test_cap, pred_test_cap))
        test_rmse_cap = rmse(y_test_cap, pred_test_cap)
        test_mae_raw = float(mean_absolute_error(y_test_raw, pred_test_raw))
        test_rmse_raw = rmse(y_test_raw, pred_test_raw)

        score_cap = nasa_score(y_test_cap, pred_test_cap)
        score_raw = nasa_score(y_test_raw, pred_test_raw)

        print(f"VAL  MAE(cap): {val_mae:.3f} | RMSE(cap): {val_rmse:.3f}")
        print(f"TEST MAE(cap): {test_mae_cap:.3f} | RMSE(cap): {test_rmse_cap:.3f} | NASA(cap): {score_cap:.2f}")

        tag = f"{fd.lower()}_rf_mss_w{window}_cap{cap}"
        model_path = models_dir / f"{tag}.joblib"
        feat_path = models_dir / f"{tag}_features.json"
        meta_path = models_dir / f"{tag}_meta.json"

        joblib.dump(model, model_path)
        feat_path.write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")

        meta = {
            "fd": fd,
            "model": "RandomForestRegressor",
            "window": window,
            "rul_cap": cap,
            "drop_sensors": sorted(list(drop_sensors)),
            "n_features": len(feat_cols),
            "n_train_rows": int(X.shape[0]),
            "n_train_units": int(len(np.unique(unit_ids))),
            "val_units": val_units,
            "metrics": {
                "val": {"mae_cap": val_mae, "rmse_cap": val_rmse},
                "test_lastcycle": {
                    "mae_cap": test_mae_cap, "rmse_cap": test_rmse_cap, "nasa_cap": score_cap,
                    "mae_raw": test_mae_raw, "rmse_raw": test_rmse_raw, "nasa_raw": score_raw,
                }
            },
            "sklearn_version": sklearn.__version__,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        preds_df = pd.DataFrame({
            "unit_id": unit_ids_test.astype(int),
            "true_rul_raw": y_test_raw,
            "true_rul_cap": y_test_cap,
            "pred_rul_raw": pred_test_raw,
            "pred_rul_cap": pred_test_cap,
            "abs_err_cap": np.abs(pred_test_cap - y_test_cap),
            "err_cap": (pred_test_cap - y_test_cap),
        }).sort_values("unit_id")

        preds_csv = reports_dir / f"{fd.lower()}_test_lastcycle_preds_w{window}_cap{cap}.csv"
        metrics_json = reports_dir / f"{fd.lower()}_test_metrics_w{window}_cap{cap}.json"

        preds_df.to_csv(preds_csv, index=False)
        metrics_json.write_text(json.dumps(meta["metrics"]["test_lastcycle"], indent=2), encoding="utf-8")

        rows.append({
            "fd": fd,
            "window": window,
            "cap": cap,
            "val_mae_cap": val_mae,
            "val_rmse_cap": val_rmse,
            "test_mae_cap": test_mae_cap,
            "test_rmse_cap": test_rmse_cap,
            "nasa_cap": score_cap,
            "model_path": str(model_path.relative_to(root)),
        })

    summary = pd.DataFrame(rows).sort_values("fd").reset_index(drop=True)
    summary_csv = reports_dir / f"cmapss_summary_rf_mss_w{window}_cap{cap}.csv"
    summary_md = reports_dir / f"cmapss_summary_rf_mss_w{window}_cap{cap}.md"
    summary.to_csv(summary_csv, index=False)

    # markdown without tabulate
    def md_table(df):
        cols = df.columns.tolist()
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    summary_md.write_text(
        f"# CMAPSS summary (RF mean/std/slope, window={window}, cap={cap})\n\n"
        + md_table(summary[["fd","val_mae_cap","val_rmse_cap","test_mae_cap","test_rmse_cap","nasa_cap"]])
        + "\n",
        encoding="utf-8"
    )

    # plot
    plt.figure()
    plt.bar(summary["fd"], summary["test_mae_cap"])
    plt.title(f"TEST MAE (cap={cap}) by FD (RF m/s/s, w={window})")
    plt.xlabel("FD")
    plt.ylabel("MAE (cap)")
    mae_fig = fig_dir / f"cmapss_test_mae_cap_w{window}_cap{cap}.png"
    plt.savefig(mae_fig, dpi=150, bbox_inches="tight")

    print("\nSaved:")
    print(" -", summary_csv.relative_to(root))
    print(" -", summary_md.relative_to(root))
    print(" -", mae_fig.relative_to(root))
    print(f"\nDone in {(time.time()-t0):.1f}s")


if __name__ == "__main__":
    main()
