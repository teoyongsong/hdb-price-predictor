"""Build feature rows for inference and run saved pipelines."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hdb_ml.features import parse_storey_midpoint


def load_model_bundle(path: Path | str | None = None) -> dict[str, Any] | None:
    """Load `model_bundle.joblib` produced by `hdb_ml.export_bundle`."""
    import joblib

    from hdb_ml.config import MODEL_BUNDLE_PATH, OUTPUT_DIR

    if path is not None:
        candidates = [Path(path)]
    else:
        # Prefer committed path (Streamlit Cloud), then legacy local outputs/
        candidates = [MODEL_BUNDLE_PATH, OUTPUT_DIR / "model_bundle.joblib"]

    for p in candidates:
        if p.exists():
            return joblib.load(p)
    return None


def remaining_lease_from_99_year_lease(lease_commence_year: int, valuation_year: int) -> float:
    """Approximate remaining lease (years) for a 99-year lease from commencement."""
    age = float(valuation_year - lease_commence_year)
    return float(max(0.0, min(99.0, 99.0 - age)))


def build_prediction_row(
    floor_area_sqm: float,
    town: str,
    flat_type: str,
    flat_model: str,
    storey_range: str,
    lease_commence_year: int,
    valuation_year: int,
    remaining_lease_years: float | None = None,
) -> pd.DataFrame:
    """
    Single-row frame with the same columns as training `X`.
    If `remaining_lease_years` is None, uses 99-year lease approximation from lease year and valuation year.
    """
    age_years = float(valuation_year - lease_commence_year)
    if remaining_lease_years is None:
        rem = remaining_lease_from_99_year_lease(lease_commence_year, valuation_year)
    else:
        rem = float(remaining_lease_years)

    storey_mid = parse_storey_midpoint(storey_range)
    if np.isnan(storey_mid):
        raise ValueError(f"Unrecognised storey range: {storey_range!r}")

    row = {
        "floor_area_sqm": float(floor_area_sqm),
        "remaining_lease_years": rem,
        "age_years": age_years,
        "storey_mid": float(storey_mid),
        "town": str(town).strip(),
        "flat_type": str(flat_type).strip(),
        "flat_model": str(flat_model).strip(),
        "storey_range": str(storey_range).strip(),
    }
    return pd.DataFrame([row])


def predict_from_bundle(bundle: dict[str, Any], row: pd.DataFrame) -> float:
    pipe = bundle["pipeline"]
    num_cols: list[str] = bundle["numeric_cols"]
    cat_cols: list[str] = bundle["categorical_cols"]
    X = row[num_cols + cat_cols]
    out = pipe.predict(X)
    return float(out[0])


def predict_price_from_inputs(
    bundle: dict[str, Any],
    floor_area_sqm: float,
    town: str,
    flat_type: str,
    flat_model: str,
    storey_range: str,
    lease_commence_year: int,
    valuation_year: int,
    remaining_lease_years: float | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Validate inputs, build one row, predict. Returns (payload, error_message).
    payload keys: predicted_price_sgd, predicted_price_raw, inputs.
    """
    if not (25 <= floor_area_sqm <= 350):
        return None, "Floor area (sqm) should be between 25 and 350."
    town, flat_type, flat_model, storey_range = (
        str(town).strip(),
        str(flat_type).strip(),
        str(flat_model).strip(),
        str(storey_range).strip(),
    )
    if not town or not flat_type or not flat_model or not storey_range:
        return None, "Please select town, flat type, model, and storey range."
    if lease_commence_year < 1966 or lease_commence_year > valuation_year:
        return None, "Lease commence year must be plausible (e.g. 1966–valuation year)."
    if valuation_year < 2017 or valuation_year > date.today().year + 1:
        return None, "Valuation year looks invalid."

    rem_opt = remaining_lease_years
    if rem_opt is not None:
        if not (0 <= rem_opt <= 99):
            return None, "Remaining lease must be between 0 and 99 years."

    try:
        row = build_prediction_row(
            floor_area_sqm=floor_area_sqm,
            town=town,
            flat_type=flat_type,
            flat_model=flat_model,
            storey_range=storey_range,
            lease_commence_year=lease_commence_year,
            valuation_year=valuation_year,
            remaining_lease_years=rem_opt,
        )
    except ValueError as e:
        return None, str(e)

    age_years = float(valuation_year - lease_commence_year)
    if not (0 <= age_years <= 120):
        return None, "Implied flat age is out of range; check lease year and valuation year."

    price = predict_from_bundle(bundle, row)
    if price < 0 or price > 5_000_000:
        return None, "Prediction out of expected range; treat as unreliable."

    payload = {
        "predicted_price_sgd": round(price, -2),
        "predicted_price_raw": price,
        "inputs": {
            "floor_area_sqm": floor_area_sqm,
            "town": town,
            "flat_type": flat_type,
            "flat_model": flat_model,
            "storey_range": storey_range,
            "lease_commence_year": lease_commence_year,
            "valuation_year": valuation_year,
            "remaining_lease_years": float(row["remaining_lease_years"].iloc[0]),
            "age_years": age_years,
        },
    }
    return payload, None
