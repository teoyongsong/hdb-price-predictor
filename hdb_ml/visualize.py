"""Plots: predicted vs actual, importances, SHAP (tree models), trends by town/flat type."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from hdb_ml.train import FitResult


def plot_predicted_vs_actual(result: "FitResult", out_path: Path, title: str | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    y_t, y_p = result.y_test, result.y_pred
    lim = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())
    plt.scatter(y_t, y_p, alpha=0.15, s=8)
    plt.plot(lim, lim, "r--", lw=1, label="Perfect fit")
    plt.xlabel("Actual resale price (SGD)")
    plt.ylabel("Predicted resale price (SGD)")
    plt.title(title or f"{result.name}: predicted vs actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance_rf(pipe, out_path: Path, top_n: int = 25) -> None:
    """RandomForest feature importances after ColumnTransformer."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prep = pipe.named_steps["prep"]
    reg = pipe.named_steps["reg"]
    names = prep.get_feature_names_out()
    imp = reg.feature_importances_
    order = np.argsort(imp)[::-1][:top_n]
    plt.figure(figsize=(10, max(6, top_n * 0.25)))
    y_labels = [names[i] for i in order]
    plt.barh(range(len(y_labels)), imp[order], color="steelblue")
    plt.yticks(range(len(y_labels)), y_labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Random Forest — feature importance (top {})".format(top_n))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_shap_summary(
    pipe,
    X_sample: pd.DataFrame,
    out_path: Path,
    max_samples: int = 500,
) -> bool:
    """SHAP summary for tree-based step (RF, XGB, LGBM). Returns False if skipped."""
    try:
        import shap
    except ImportError:
        return False

    reg = pipe.named_steps.get("reg")
    prep = pipe.named_steps.get("prep")
    if reg is None or prep is None:
        return False
    if not hasattr(reg, "feature_importances_") and type(reg).__name__ not in (
        "XGBRegressor",
        "LGBMRegressor",
        "RandomForestRegressor",
    ):
        # XGB/LGBM still work with TreeExplainer
        pass

    Xs = X_sample.iloc[:max_samples]
    try:
        X_trans = prep.transform(Xs)
    except Exception:
        return False

    try:
        explainer = shap.TreeExplainer(reg)
        sv = explainer.shap_values(X_trans)
    except Exception:
        return False

    names = prep.get_feature_names_out()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_trans, feature_names=names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def plot_price_trends(
    df: pd.DataFrame,
    out_dir: Path,
    price_col: str = "resale_price",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "town" in df.columns:
        g = df.groupby("town", observed=True)[price_col].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        g.plot(kind="bar", color="steelblue")
        plt.ylabel("Mean resale price (SGD)")
        plt.xlabel("Town")
        plt.title("Mean resale price by town")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "mean_price_by_town.png", dpi=150)
        plt.close()

    if "flat_type" in df.columns:
        g = df.groupby("flat_type", observed=True)[price_col].mean().sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        g.plot(kind="bar", color="darkseagreen")
        plt.ylabel("Mean resale price (SGD)")
        plt.xlabel("Flat type")
        plt.title("Mean resale price by flat type")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "mean_price_by_flat_type.png", dpi=150)
        plt.close()
