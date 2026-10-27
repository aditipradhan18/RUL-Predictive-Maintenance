
import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

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
# Data and API helpers
# --------------------------------------------------
@st.cache_data
def load_validation_data():
    with np.load(DATA_PATH) as data:
        return data["X_val"], data["y_val"]


def check_api():
    try:
        request = urllib.request.Request(f"{API_URL}/health")
        with urllib.request.urlopen(request, timeout=3) as response:
            health = json.loads(response.read().decode())

        return (
            health.get("status") == "healthy"
            and health.get("model_loaded") is True
        )

    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def predict_rul(sequence):
    payload = json.dumps(
        {"sequence": sequence.tolist()}
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{API_URL}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode())

    prediction = float(result["predicted_rul_cycles"])

    if not np.isfinite(prediction):
        raise ValueError("The model returned a non-finite prediction.")

    return max(0.0, prediction)


def classify_priority(rul):
    # Illustrative thresholds for this demonstration only.
    if rul <= 20:
        return "High"
    if rul <= 50:
        return "Medium"
    return "Low"


def predict_fleet(X_val, y_val, indices):
    results = []

    progress = st.progress(0, text="Analyzing simulated engines...")

    for position, index in enumerate(indices, start=1):
        predicted = predict_rul(X_val[index])
        actual = float(y_val[index])

        results.append(
            {
                "Engine": f"SIM-{index + 1:04d}",
                "Sequence Index": int(index),
                "Predicted RUL (cycles)": predicted,
                "Actual RUL (cycles)": actual,
                "Signed Error (cycles)": predicted - actual,
                "Absolute Error (cycles)": abs(predicted - actual),
                "Priority": classify_priority(predicted),
            }
        )

        progress.progress(
            position / len(indices),
            text=f"Analyzed {position} of {len(indices)} engines",
        )

    progress.empty()
    return pd.DataFrame(results)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">⚙️ QuantumFlix</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Remaining Useful Life (RUL) '
    'Estimation & Predictive Maintenance</div>',
    unsafe_allow_html=True,
)

st.info(
    "Simulation demo: each engine represents a validation sequence "
    "from NASA C-MAPSS FD001. These are not real engines or live telemetry."
)

try:
    X_val, y_val = load_validation_data()
except (FileNotFoundError, KeyError, OSError) as exc:
    st.error(f"Could not load validation data: {exc}")
    st.stop()

api_healthy = check_api()


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
# Simulated fleet overview
# --------------------------------------------------
st.subheader("Fleet Overview")

st.caption(
    "Select a small sample of validation sequences and analyze them "
    "using the trained LSTM model through FastAPI."
)

max_fleet_size = min(20, len(X_val))

fleet_size = st.slider(
    "Number of simulated engines",
    min_value=min(3, max_fleet_size),
    max_value=max_fleet_size,
    value=min(5, max_fleet_size),
)

fleet_indices = np.linspace(
    0, len(X_val) - 1, num=fleet_size, dtype=int
)

if "fleet_results" not in st.session_state:
    st.session_state.fleet_results = None

if st.button(
    "Analyze Fleet",
    type="primary",
    disabled=not api_healthy,
):
    try:
        with st.spinner("Generating fleet predictions..."):
            st.session_state.fleet_results = predict_fleet(
                X_val, y_val, fleet_indices
            )
    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        KeyError,
    ) as exc:
        st.error(f"Fleet analysis failed: {exc}")

fleet_df = st.session_state.fleet_results

if fleet_df is not None and not fleet_df.empty:
    # Recompute if the user changes fleet size.
    expected_indices = set(map(int, fleet_indices))
    existing_indices = set(fleet_df["Sequence Index"].astype(int))

    if expected_indices != existing_indices:
        st.info("Fleet size changed. Click Analyze Fleet to refresh predictions.")
    else:
        fleet_df = fleet_df.sort_values(
            "Predicted RUL (cycles)"
        ).reset_index(drop=True)

        high_count = int((fleet_df["Priority"] == "High").sum())
        medium_count = int((fleet_df["Priority"] == "Medium").sum())
        low_count = int((fleet_df["Priority"] == "Low").sum())

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Analyzed Engines", len(fleet_df))
        m2.metric("High Priority", high_count)
        m3.metric("Medium Priority", medium_count)
        m4.metric("Low Priority", low_count)

        st.markdown("#### Engines ranked by predicted RUL")

        display_columns = [
            "Engine",
            "Predicted RUL (cycles)",
            "Actual RUL (cycles)",
            "Absolute Error (cycles)",
            "Priority",
        ]

        st.dataframe(
            fleet_df[display_columns].round(2),
            use_container_width=True,
            hide_index=True,
        )

        most_urgent = fleet_df.iloc[0]

        st.warning(
            f"Lowest predicted RUL: {most_urgent['Engine']} — "
            f"{most_urgent['Predicted RUL (cycles)']:.2f} cycles. "
            "This is a model-based demonstration ranking, not an "
            "operational maintenance instruction."
        )

elif not api_healthy:
    st.warning(
        "The prediction API is offline. Start it in another terminal "
        "with: `uvicorn src.api.main:app --reload`"
    )

else:
    st.info("Click Analyze Fleet to generate predictions.")

st.caption(
    "Priority thresholds are illustrative and are not validated "
    "aircraft maintenance limits."
)

st.divider()


# --------------------------------------------------
# Prediction error analysis
# --------------------------------------------------
st.subheader("Prediction Error Analysis")

st.caption(
    "Charts below use the fleet predictions generated above. "
    "They update when you analyze a new fleet."
)

if fleet_df is not None and not fleet_df.empty:
    fleet_plot_df = fleet_df.copy()

    mae = fleet_plot_df["Absolute Error (cycles)"].mean()
    rmse = np.sqrt(
        np.mean(fleet_plot_df["Signed Error (cycles)"] ** 2)
    )
    mean_error = fleet_plot_df["Signed Error (cycles)"].mean()

    e1, e2, e3 = st.columns(3)

    e1.metric("Fleet MAE", f"{mae:.2f} cycles")
    e2.metric("Fleet RMSE", f"{rmse:.2f} cycles")
    e3.metric("Mean Signed Error", f"{mean_error:+.2f} cycles")

    chart_left, chart_right = st.columns(2)

    # Predicted versus actual RUL
    with chart_left:
        st.markdown("#### Predicted vs. Actual RUL")

        fig, ax = plt.subplots(figsize=(7, 5))

        actual = fleet_plot_df["Actual RUL (cycles)"]
        predicted = fleet_plot_df["Predicted RUL (cycles)"]

        ax.scatter(actual, predicted)

        lower = min(actual.min(), predicted.min(), 0)
        upper = max(actual.max(), predicted.max(), 1)

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
        )

        ax.set_xlabel("Actual RUL (cycles)")
        ax.set_ylabel("Predicted RUL (cycles)")
        ax.set_title("Predictions vs. Ground Truth")
        ax.grid(True, alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

    # Distribution of absolute prediction errors
    with chart_right:
        st.markdown("#### Absolute Error Distribution")

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.hist(
            fleet_plot_df["Absolute Error (cycles)"],
            bins=min(10, max(3, len(fleet_plot_df))),
        )

        ax.set_xlabel("Absolute Error (cycles)")
        ax.set_ylabel("Number of Engines")
        ax.set_title("How Large Are the Errors?")
        ax.grid(True, axis="y", alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Largest Prediction Errors")

    largest_errors = fleet_plot_df.sort_values(
        "Absolute Error (cycles)",
        ascending=False,
    )

    st.dataframe(
        largest_errors[
            [
                "Engine",
                "Actual RUL (cycles)",
                "Predicted RUL (cycles)",
                "Signed Error (cycles)",
                "Absolute Error (cycles)",
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Fleet metrics describe only this selected sample. They are not "
        "the model's full validation metrics."
    )

else:
    st.info(
        "Analyze a fleet first to generate predictions and display "
        "prediction error charts."
    )

st.divider()


# --------------------------------------------------
# Individual sequence prediction
# --------------------------------------------------
st.subheader("Individual Engine RUL Prediction")

engine_index = st.selectbox(
    "Select a validation sequence",
    options=list(range(len(X_val))),
    format_func=lambda i: (
        f"SIM-{i + 1:04d} — actual RUL: {y_val[i]:.0f} cycles"
    ),
)

sequence = X_val[engine_index]

left, right = st.columns(2)

with left:
    st.markdown("#### Selected sequence")
    st.write(f"Sequence ID: **{engine_index + 1}**")
    st.write(f"Time steps: **{sequence.shape[0]}**")
    st.write(f"Features per step: **{sequence.shape[1]}**")
    st.write(f"Actual RUL: **{y_val[engine_index]:.0f} cycles**")

with right:
    st.markdown("#### Model prediction")

    if not api_healthy:
        st.warning("Prediction API is offline.")

    elif st.button("Predict Remaining Useful Life"):
        try:
            predicted_rul = predict_rul(sequence)
            actual_rul = float(y_val[engine_index])

            st.metric(
                "Predicted RUL",
                f"{predicted_rul:.2f} cycles",
                delta=f"{predicted_rul - actual_rul:+.2f} vs actual",
            )

            st.metric("Actual RUL", f"{actual_rul:.2f} cycles")

            st.metric(
                "Absolute Error",
                f"{abs(predicted_rul - actual_rul):.2f} cycles",
            )

            priority = classify_priority(predicted_rul)

            if priority == "High":
                st.error("High demo maintenance priority")
            elif priority == "Medium":
                st.warning("Medium demo maintenance priority")
            else:
                st.success("Low demo maintenance priority")

        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            KeyError,
        ) as exc:
            st.error(f"Prediction request failed: {exc}")

st.divider()

st.caption(
    "QuantumFlix demo · NASA C-MAPSS FD001 · "
    "PyTorch LSTM · FastAPI · Streamlit"
)
