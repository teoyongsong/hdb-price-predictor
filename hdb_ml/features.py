"""
Engineered features for HDB resale modelling.

Note: `price_per_sqm` uses the target (`resale_price`) and must not be used as a model input
when predicting price — only for EDA and reporting.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from hdb_ml.config import GEO_COLUMNS


_RE_LEASE = re.compile(
    r"(?P<years>\d+)\s*years?\s*(?:(?P<months>\d+)\s*months?)?",
    re.IGNORECASE,
)
_RE_STOREY = re.compile(r"(\d+)\s*TO\s*(\d+)", re.IGNORECASE)


def parse_remaining_lease_years(s: str | float) -> float:
    """Convert strings like '61 years 04 months' to approximate years (float)."""
    if pd.isna(s):
        return np.nan
    text = str(s).strip()
    m = _RE_LEASE.search(text)
    if not m:
        return np.nan
    y = float(m.group("years"))
    mo = m.group("months")
    if mo is not None:
        y += float(mo) / 12.0
    return y


def parse_storey_midpoint(storey_range: str | float) -> float:
    """Midpoint of e.g. '10 TO 12' for use as numeric feature."""
    if pd.isna(storey_range):
        return np.nan
    text = str(storey_range).strip()
    m = _RE_STOREY.search(text)
    if not m:
        return np.nan
    low, high = float(m.group(1)), float(m.group(2))
    return (low + high) / 2.0


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["month"].astype(str)
    out["sale_month"] = pd.to_datetime(out["month"] + "-01", errors="coerce")
    out["sale_year"] = out["sale_month"].dt.year
    out["lease_commence_year"] = pd.to_numeric(out["lease_commence_date"], errors="coerce")

    out["remaining_lease_years"] = out["remaining_lease"].map(parse_remaining_lease_years)
    out["storey_mid"] = out["storey_range"].map(parse_storey_midpoint)
    out["age_years"] = out["sale_year"] - out["lease_commence_year"]

    out["floor_area_sqm"] = pd.to_numeric(out["floor_area_sqm"], errors="coerce")
    out["resale_price"] = pd.to_numeric(out["resale_price"], errors="coerce")

    # For analysis only — do not pass to model as X (leakage)
    out["price_per_sqm"] = out["resale_price"] / out["floor_area_sqm"].replace(0, np.nan)

    return out


def model_feature_columns(
    df: pd.DataFrame,
    include_geo: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Returns (numeric_cols, categorical_cols) for the modelling matrix.
    Geo columns are included only if present and `include_geo` is True.
    """
    numeric = [
        "floor_area_sqm",
        "remaining_lease_years",
        "age_years",
        "storey_mid",
    ]
    categorical = ["town", "flat_type", "flat_model", "storey_range"]

    if include_geo:
        for c in GEO_COLUMNS:
            if c in df.columns and df[c].notna().any():
                if c in ("lat", "lng", "nearest_mrt_km", "nearest_school_km"):
                    numeric.append(c)

    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    return numeric, categorical
