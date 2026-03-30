#!/usr/bin/env python3
"""
Flask app: HTML form for HDB resale price prediction (loads `outputs/model_bundle.joblib`).
Run `python -m hdb_ml.export_bundle` first to train and save the bundle.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdb_ml.config import MODEL_BUNDLE_PATH
from hdb_ml.inference import load_model_bundle, predict_price_from_inputs

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))

_bundle: dict | None = None


def get_bundle() -> dict | None:
    global _bundle
    if _bundle is not None:
        return _bundle
    _bundle = load_model_bundle()
    return _bundle


@app.route("/")
def index():
    ready = get_bundle() is not None
    return render_template("index.html", model_ready=ready, model_path=str(MODEL_BUNDLE_PATH))


@app.route("/api/health")
def health():
    b = get_bundle()
    return jsonify(
        {
            "model_loaded": b is not None,
            "bundle_path": str(MODEL_BUNDLE_PATH),
        }
    )


@app.route("/api/options")
def options():
    b = get_bundle()
    if not b:
        return (
            jsonify(
                {
                    "error": "Model bundle not found. Run: python -m hdb_ml.export_bundle",
                    "bundle_path": str(MODEL_BUNDLE_PATH),
                }
            ),
            503,
        )
    return jsonify(
        {
            "option_lists": b["option_lists"],
            "model_name": b.get("model_name", "xgboost"),
            "n_train_rows": b.get("n_train_rows"),
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    b = get_bundle()
    if not b:
        return (
            jsonify({"error": "Model bundle not found. Run: python -m hdb_ml.export_bundle"}),
            503,
        )

    data = request.get_json(silent=True) or {}
    try:
        floor_area = float(data.get("floor_area_sqm", 0))
        town = str(data.get("town", "")).strip()
        flat_type = str(data.get("flat_type", "")).strip()
        flat_model = str(data.get("flat_model", "")).strip()
        storey_range = str(data.get("storey_range", "")).strip()
        lease_commence_year = int(data.get("lease_commence_year", 0))
        valuation_year = int(data.get("valuation_year", date.today().year))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    rem_override = data.get("remaining_lease_years")
    rem_opt: float | None
    if rem_override is not None and rem_override != "":
        try:
            rem_opt = float(rem_override)
        except (TypeError, ValueError):
            return jsonify({"error": "Remaining lease (years) must be a number."}), 400
    else:
        rem_opt = None

    payload, err = predict_price_from_inputs(
        b,
        floor_area_sqm=floor_area,
        town=town,
        flat_type=flat_type,
        flat_model=flat_model,
        storey_range=storey_range,
        lease_commence_year=lease_commence_year,
        valuation_year=valuation_year,
        remaining_lease_years=rem_opt,
    )
    if err:
        err_l = err.lower()
        status = 422 if "unreliable" in err_l or "expected range" in err_l else 400
        return jsonify({"error": err}), status
    return jsonify(payload)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="HDB price predictor web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
