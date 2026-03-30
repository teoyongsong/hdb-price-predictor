"""Train on full cleaned data and save a joblib bundle for the web app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from hdb_ml.config import DATA_CSV, MODEL_BUNDLE_PATH
from hdb_ml.preprocess import build_xy, clean_and_engineer, load_raw_csv
from hdb_ml.train import fit_xgboost_pipeline


def build_option_lists(df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in categorical_cols:
        vals = sorted(df[c].astype(str).unique().tolist())
        out[c] = vals
    return out


def export_xgboost_bundle(
    csv_path: Path | str | None = None,
    out_path: Path | str | None = None,
    sample: int | None = None,
) -> Path:
    """
    Fit XGBoost on (optionally sampled) cleaned data and save pipeline + column lists + dropdown values.
    """
    csv_path = Path(csv_path or DATA_CSV)
    out_path = Path(out_path or MODEL_BUNDLE_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_raw_csv(str(csv_path))
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    df = clean_and_engineer(df)
    X, y, num_cols, cat_cols = build_xy(df, include_geo=False)
    option_lists = build_option_lists(df, cat_cols)

    pipe = fit_xgboost_pipeline(X, y, num_cols, cat_cols)

    bundle: dict[str, Any] = {
        "pipeline": pipe,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "option_lists": option_lists,
        "model_name": "xgboost",
        "n_train_rows": len(X),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return out_path


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Export XGBoost pipeline bundle for the web app")
    parser.add_argument("--csv", type=str, default=None, help="Path to resale CSV")
    parser.add_argument("--out", type=str, default=None, help="Output .joblib path")
    parser.add_argument("--sample", type=int, default=None, help="Random sample size for faster export")
    args = parser.parse_args()
    try:
        p = export_xgboost_bundle(csv_path=args.csv, out_path=args.out, sample=args.sample)
    except Exception as e:
        print(e, file=sys.stderr)
        return 1
    b = joblib.load(p)
    n = int(b.get("n_train_rows", 0))
    print(f"Saved model bundle to {p.resolve()} (trained on {n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
