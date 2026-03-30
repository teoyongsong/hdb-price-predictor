#!/usr/bin/env python3
"""
End-to-end workflow: load data.gov.sg HDB resale CSV, preprocess, train models,
print RMSE / MAE / R², save plots (predicted vs actual, importances, SHAP, town/flat trends).

Ethical note: use predictions as indicative only; disclose limitations and data drift risk.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Project root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdb_ml.config import DATA_CSV, OUTPUT_DIR, RANDOM_STATE
from hdb_ml.preprocess import build_xy, clean_and_engineer, load_raw_csv
from hdb_ml.train import run_all
from hdb_ml.visualize import (
    plot_feature_importance_rf,
    plot_predicted_vs_actual,
    plot_price_trends,
    plot_shap_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="HDB resale price ML pipeline")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to resale CSV (default: data/resale.csv; fetch via scripts/fetch_hdb_resale.py)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional row limit for faster experimentation (random sample)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else DATA_CSV
    if not csv_path.exists():
        print(
            f"Missing data file: {csv_path}\n"
            "Run: python scripts/fetch_hdb_resale.py",
            file=sys.stderr,
        )
        return 1

    print(f"Loading {csv_path} ...")
    df = load_raw_csv(str(csv_path))
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Using random sample of {args.sample} rows.")

    df = clean_and_engineer(df)
    print(f"Rows after cleaning: {len(df)}", flush=True)

    X, y, num_cols, cat_cols = build_xy(df, include_geo=False)
    print(f"Numeric features: {num_cols}", flush=True)
    print(f"Categorical features: {cat_cols}", flush=True)

    print("Training models (this may take a few minutes on full data)...", flush=True)
    results = run_all(X, y, num_cols, cat_cols)

    print("\n=== Metrics (hold-out test set) ===")
    best = min(results, key=lambda r: r.rmse)
    for r in results:
        mark = " *" if r is best else ""
        print(f"{r.name:30} RMSE: {r.rmse:,.0f}  MAE: {r.mae:,.0f}  R²: {r.r2:.4f}{mark}")

    if args.no_plots:
        return 0

    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    for r in results:
        safe = re.sub(r"[^\w\-]+", "_", r.name).strip("_").lower()
        plot_predicted_vs_actual(r, out / f"pred_vs_actual_{safe}.png")

    # RF importances + SHAP using RF pipeline (interpretable tree)
    rf = next((r for r in results if r.name.startswith("random_forest")), None)
    if rf:
        plot_feature_importance_rf(rf.model, out / "feature_importance_random_forest.png")
        plot_shap_summary(rf.model, X, out / "shap_summary_random_forest.png")

    xgb = next((r for r in results if r.name == "xgboost"), None)
    if xgb:
        plot_shap_summary(xgb.model, X, out / "shap_summary_xgboost.png")

    plot_price_trends(df, out)

    print(f"\nFigures written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
