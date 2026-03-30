# HDB resale price predictor (supervised learning)

Data is fetched from **data.gov.sg** — dataset *Resale flat prices based on registration date from Jan-2017 onwards* (`d_8b84c4ee58e3cfc0ece0d773c8ca6abc`).

## Setup

```bash
cd "hdb price predictor"
pip install -r requirements.txt
python scripts/fetch_hdb_resale.py
```

Use `python scripts/fetch_hdb_resale.py --force` to re-download when the portal has a new `lastUpdatedAt`.

## Web UI (HTML form)

Train and save the production model bundle, then start the Flask app:

```bash
pip install -r requirements.txt
python -m hdb_ml.export_bundle          # full data; add --sample 50000 for a quicker export
python app.py                           # http://127.0.0.1:5000
```

`--port` and `--host` are supported (see `python app.py --help`). The page loads town, flat type, model, and storey options from the bundle; users enter floor area (sqm), lease commencement year, and valuation year (optional advanced: remaining lease override).

## Run the pipeline

```bash
python run_pipeline.py
```

- **`--sample N`**: random subset for quicker runs (omit for full ~228k rows).
- **`--no-plots`**: metrics only, no figures.
- **`--csv PATH`**: use another CSV with the same columns.

Outputs go to `outputs/` (predicted vs actual, Random Forest importances, SHAP for RF and XGBoost, mean price by town and flat type).

## Workflow (implemented)

1. **Preprocessing** — drop incomplete rows; **StandardScaler** on numeric features; **one-hot** encoding for town, flat type, flat model, storey range.
2. **Feature engineering** — remaining lease (years), flat age (sale year − lease commence year), storey midpoint; **`price_per_sqm`** is computed for analysis only (not used as a model input — it would leak the target).
3. **Models** — Linear Regression, Random Forest, XGBoost (`tree_method=hist`), LightGBM; metrics **RMSE**, **MAE**, **R²** on a random hold-out set.
4. **Visualisation** — prediction scatter plots, feature importance / SHAP (tree models), bar charts by town and flat type.

**Optional geolocation** (MRT / schools): the official CSV has no lat/lon. If you merge external coordinates, add columns such as `lat`, `lng`, `nearest_mrt_km` and extend `hdb_ml.config.GEO_COLUMNS`; then use `build_xy(..., include_geo=True)` (wire this in `run_pipeline.py` if needed).

## Risks and ethical use

- **Data drift** — HDB rules and the market change; retrain and validate on recent periods. Consider a **time-based** train/test split instead of random for more realistic error.
- **Outliers** — rare flat types or areas can skew errors; inspect residuals and segment metrics.
- **Use** — treat predictions as indicative; disclose limitations and uncertainty; do not mislead buyers or sellers.
