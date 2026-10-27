
import json
import urllib.error
import urllib.request

import numpy as np
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="QuantumFlix | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"
DATA_PATH = "artifacts/sequence_data.npz"

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1020;
        color: #f1f5f9;
    }
    [data-testid="stMetric"] {
        background-color: #151d32;
        border: 1px solid #293550;
        padding: 18px;
        border-radius: 12px;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 750;
        color: #f8fafc;
    }
    .subtitle {
        color: #a5b4cc;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load saved validation data
# --------------------------------------------------
@st.cache_data
def load_validation_data():
    with np.load(DATA_PATH) as data:
        return data["X_val"], data["y_val"]


st.markdown(
    '<div class="main-title">⚙️ QuantumFlix</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Remaining Useful Life (RUL) '
    'Estimation & Predictive Maintenance</div>',
    unsafe_allow_html=True,
)

try:
    X_val, y_val = load_validation_data()
except (FileNotFoundError, KeyError, OSError) as exc:
    st.error(f"Could not load validation data: {exc}")
    st.stop()

# --------------------------------------------------
# API health
# --------------------------------------------------
api_healthy = False

try:
    request = urllib.request.Request(f"{API_URL}/health")
    with urllib.request.urlopen(request, timeout=3) as response:
        health = json.loads(response.read().decode())
        api_healthy = (
            health.get("status") == "healthy"
            and health.get("model_loaded") is True
        )
except (urllib.error.URLError, TimeoutError, ValueError):
    health = {}

# --------------------------------------------------
# Overview metrics
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Validation Sequences", f"{len(X_val):,}")

with col2:
    st.metric("Window Size", "50 cycles")

with col3:
    st.metric("Input Features", "18")

with col4:
    st.metric(
        "Prediction API",
        "Online" if api_healthy else "Offline",
    )

st.divider()

# --------------------------------------------------
# Engine sequence selection
# --------------------------------------------------
st.subheader("Engine RUL Prediction")

st.caption(
    "Choose a held-out validation sequence. These are simulated "
    "engine sequences from the NASA C-MAPSS FD001 training dataset, "
    "not live aircraft telemetry."
)

engine_index = st.selectbox(
    "Select validation sequence",
    options=list(range(len(X_val))),
    format_func=lambda i: (
        f"Validation sequence {i + 1} "
        f"(actual RUL: {y_val[i]:.0f} cycles)"
    ),
)

sequence = X_val[engine_index]

left, right = st.columns([1, 1])

with left:
    st.markdown("#### Selected sequence")
    st.write(f"Sequence ID: **{engine_index + 1}**")
    st.write(f"Time steps: **{sequence.shape[0]}**")
    st.write(f"Features per step: **{sequence.shape[1]}**")
    st.write(f"Actual RUL: **{y_val[engine_index]:.0f} cycles**")

with right:
    st.markdown("#### Model prediction")

    if not api_healthy:
        st.warning(
            "The prediction API is offline. Start it in a separate "
            "terminal with: `uvicorn src.api.main:app --reload`"
        )
    elif st.button("Predict Remaining Useful Life", type="primary"):
        payload = json.dumps(
            {"sequence": sequence.tolist()}
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{API_URL}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode())

            predicted_rul = float(result["predicted_rul_cycles"])
            actual_rul = float(y_val[engine_index])
            absolute_error = abs(predicted_rul - actual_rul)

            if not np.isfinite(predicted_rul):
                st.error("The model returned a non-finite prediction.")
            else:
                st.metric(
                    "Predicted RUL",
                    f"{predicted_rul:.2f} cycles",
                    delta=f"{predicted_rul - actual_rul:+.2f} vs actual",
                )
                st.metric("Actual RUL", f"{actual_rul:.2f} cycles")
                st.metric("Absolute Error", f"{absolute_error:.2f} cycles")

                if predicted_rul <= 20:
                    st.error("High maintenance priority")
                elif predicted_rul <= 50:
                    st.warning("Maintenance planning recommended")
                else:
                    st.success("Lower immediate maintenance priority")

                st.caption(
                    "Risk bands are illustrative demo thresholds, not "
                    "validated operational or aircraft-safety limits."
                )

        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            st.error(f"Prediction request failed: {exc}")

st.divider()
st.caption(
    "Demo only · NASA C-MAPSS FD001 · PyTorch LSTM · FastAPI · Streamlit"
)
