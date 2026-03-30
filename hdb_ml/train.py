"""Train baseline and tree models; report RMSE, MAE, R²."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hdb_ml.config import RANDOM_STATE, TEST_SIZE


@dataclass
class FitResult:
    name: str
    model: Any
    rmse: float
    mae: float
    r2: float
    y_test: np.ndarray
    y_pred: np.ndarray


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def train_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> FitResult:
    pre = make_preprocessor(numeric_cols, categorical_cols)
    pipe = Pipeline([("prep", pre), ("reg", LinearRegression())])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse, mae, r2 = _metrics(y_test.to_numpy(), y_pred)
    return FitResult("linear_regression", pipe, rmse, mae, r2, y_test.to_numpy(), y_pred)


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_estimators: int = 80,
    max_depth: int | None = 18,
) -> FitResult:
    pre = make_preprocessor(numeric_cols, categorical_cols)
    pipe = Pipeline(
        [
            ("prep", pre),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse, mae, r2 = _metrics(y_test.to_numpy(), y_pred)
    return FitResult(f"random_forest(n={n_estimators})", pipe, rmse, mae, r2, y_test.to_numpy(), y_pred)


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> FitResult | None:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None

    pre = make_preprocessor(numeric_cols, categorical_cols)
    pipe = Pipeline(
        [
            ("prep", pre),
            (
                "reg",
                XGBRegressor(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse, mae, r2 = _metrics(y_test.to_numpy(), y_pred)
    return FitResult("xgboost", pipe, rmse, mae, r2, y_test.to_numpy(), y_pred)


def fit_xgboost_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """Train XGBoost on all rows (for production export)."""
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError("xgboost is required to export the production model") from e

    pre = make_preprocessor(numeric_cols, categorical_cols)
    pipe = Pipeline(
        [
            ("prep", pre),
            (
                "reg",
                XGBRegressor(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> FitResult | None:
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    pre = make_preprocessor(numeric_cols, categorical_cols)
    pipe = Pipeline(
        [
            ("prep", pre),
            (
                "reg",
                lgb.LGBMRegressor(
                    n_estimators=120,
                    max_depth=-1,
                    learning_rate=0.08,
                    num_leaves=48,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                    n_jobs=1,
                ),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pipe.fit(X_train, y_train)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*valid feature names.*",
            category=UserWarning,
        )
        y_pred = pipe.predict(X_test)
    rmse, mae, r2 = _metrics(y_test.to_numpy(), y_pred)
    return FitResult("lightgbm", pipe, rmse, mae, r2, y_test.to_numpy(), y_pred)


def run_all(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[FitResult]:
    results: list[FitResult] = []
    print("  linear_regression ...", flush=True)
    results.append(train_linear_regression(X, y, numeric_cols, categorical_cols))
    print("  random_forest ...", flush=True)
    results.append(train_random_forest(X, y, numeric_cols, categorical_cols))
    print("  xgboost ...", flush=True)
    xgb = train_xgboost(X, y, numeric_cols, categorical_cols)
    if xgb:
        results.append(xgb)
    print("  lightgbm ...", flush=True)
    lgb = train_lightgbm(X, y, numeric_cols, categorical_cols)
    if lgb:
        results.append(lgb)
    return results
