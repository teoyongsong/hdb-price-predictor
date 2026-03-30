"""Load, clean, and build train/test matrices."""

from __future__ import annotations

import pandas as pd

from hdb_ml.features import add_engineered_features, model_feature_columns


def load_raw_csv(path: str | None = None) -> pd.DataFrame:
    from hdb_ml.config import DATA_CSV

    p = path or DATA_CSV
    return pd.read_csv(p, low_memory=False)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = add_engineered_features(df)
    # Core columns required for modelling
    req = [
        "floor_area_sqm",
        "remaining_lease_years",
        "age_years",
        "storey_mid",
        "town",
        "flat_type",
        "flat_model",
        "storey_range",
        "resale_price",
    ]
    df = df.dropna(subset=[c for c in req if c in df.columns])
    df = df[df["floor_area_sqm"] > 0]
    df = df[df["age_years"].between(0, 120)]
    df = df[df["remaining_lease_years"].between(0, 99)]
    return df.reset_index(drop=True)


def build_xy(
    df: pd.DataFrame,
    include_geo: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    num_cols, cat_cols = model_feature_columns(df, include_geo=include_geo)
    X = df[num_cols + cat_cols].copy()
    y = df["resale_price"].copy()
    return X, y, num_cols, cat_cols
