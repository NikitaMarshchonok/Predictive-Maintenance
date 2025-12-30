# FD001 — RUL prediction report
## Model
- Dataset: **FD001**
- Window: **30**
- RUL cap: **125**
- Features: **mean/std/slope (rolling)**
- Model file: ``
- sklearn (train): `unknown`
- sklearn (runtime): `1.7.2`
- python: `3.11.9`

## Metrics (computed from last-cycle test)
- MAE (cap): **13.036**
- RMSE (cap): **17.106**
- MAE (raw): **14.106**
- RMSE (raw): **18.224**
- NASA score (cap): **659.05**

## Metrics (saved json)
```json
{
  "dataset": "FD001",
  "split": "TEST_last_cycle",
  "window": 30,
  "rul_cap": 125,
  "mae_cap": 13.036241502164502,
  "rmse_cap": 17.10566577312899,
  "mae_raw": 14.106241502164503,
  "rmse_raw": 18.22385995700342,
  "model_path": "/Users/nikitamarshchonok/Desktop/Predictive Maintenance/models/fd001_rf_mss_w30_cap125.joblib"
}
```

## Figures
![True vs Pred](reports/figures/fd001_true_vs_pred_cap125_w30.png)
![Residuals scatter](reports/figures/fd001_residuals_cap125_w30.png)
![Residuals hist](reports/figures/fd001_residuals_hist_cap125_w30.png)
