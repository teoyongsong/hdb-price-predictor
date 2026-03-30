"""
Streamlit UI for HDB resale price estimates.

Run locally:
  streamlit run streamlit_app.py

Requires `outputs/model_bundle.joblib` from:
  python -m hdb_ml.export_bundle
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from hdb_ml.config import MODEL_BUNDLE_PATH
from hdb_ml.inference import load_model_bundle, predict_price_from_inputs

st.set_page_config(
    page_title="HDB resale estimate",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_bundle():
    return load_model_bundle()


def main() -> None:
    st.title("HDB resale estimate")
    st.caption(
        "Indicative price from a gradient-boosted model trained on open resale data "
        "(data.gov.sg). Not a valuation for finance or legal use."
    )

    bundle = get_bundle()
    if bundle is None:
        st.error(f"No model file at `{MODEL_BUNDLE_PATH}`.")
        st.code("python -m hdb_ml.export_bundle", language="bash")
        st.stop()

    opts = bundle["option_lists"]
    n_train = bundle.get("n_train_rows")
    if n_train:
        st.sidebar.metric("Training rows (bundle)", f"{n_train:,}")

    default_val_year = min(2026, date.today().year + 1)

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            town = st.selectbox("Town", opts["town"], index=0)
            flat_type = st.selectbox("Flat type", opts["flat_type"], index=0)
        with c2:
            flat_model = st.selectbox("Flat model", opts["flat_model"], index=0)
            storey_range = st.selectbox("Storey range", opts["storey_range"], index=0)

        floor_area = st.number_input(
            "Floor area (sqm)",
            min_value=25.0,
            max_value=350.0,
            value=90.0,
            step=1.0,
        )

        y1, y2 = st.columns(2)
        with y1:
            lease_year = st.number_input(
                "Lease commence year",
                min_value=1966,
                max_value=2035,
                value=1995,
                step=1,
            )
        with y2:
            val_year = st.number_input(
                'Valuation "as of" year',
                min_value=2017,
                max_value=date.today().year + 1,
                value=default_val_year,
                step=1,
            )

        with st.expander("Advanced"):
            override_rem = st.checkbox("Override remaining lease (years)")
            rem_opt: float | None = None
            if override_rem:
                rem_opt = float(
                    st.number_input(
                        "Remaining lease (years)",
                        min_value=0.0,
                        max_value=99.0,
                        value=70.0,
                        step=0.5,
                    )
                )

        submitted = st.form_submit_button("Estimate price", type="primary")

    if submitted:
        payload, err = predict_price_from_inputs(
            bundle,
            floor_area_sqm=float(floor_area),
            town=town,
            flat_type=flat_type,
            flat_model=flat_model,
            storey_range=storey_range,
            lease_commence_year=int(lease_year),
            valuation_year=int(val_year),
            remaining_lease_years=rem_opt if override_rem else None,
        )
        if err:
            st.error(err)
        elif payload:
            p = payload["predicted_price_sgd"]
            inp = payload["inputs"]
            st.success(f"**Estimated resale price: SGD {p:,.0f}**")
            st.info(
                f"Approx. age **{inp['age_years']:.0f}** y · Remaining lease ~**{inp['remaining_lease_years']:.1f}** y"
            )

    st.divider()
    st.markdown(
        "**Limitations:** market drift, rare flat types, and policy changes are not fully captured. "
        "Round figures are indicative; always disclose uncertainty."
    )


if __name__ == "__main__":
    main()
