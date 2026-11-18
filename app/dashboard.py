
import json
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Wedge, Rectangle, FancyArrowPatch
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RUL Command Centre",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = "http://127.0.0.1:8000"
DATA_PATH = PROJECT_ROOT / "artifacts" / "sequence_data.npz"

FEATURE_COLUMNS = [
    "cycle", "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

C = {
    "bg": "#080E18",
    "panel": "#101B2A",
    "border": "#263A50",
    "text": "#EAF2FA",
    "muted": "#91A5BA",
    "cyan": "#42D9E8",
    "green": "#42E0AE",
    "amber": "#F5B942",
    "red": "#F05D78",
    "blue": "#579BFF",
    "purple": "#A78BFA",
}


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(circle at 78% 0%, rgba(34,90,120,.17), transparent 27%),
        linear-gradient(135deg,#080E18 0%,#0A1421 58%,#0B1724 100%);
    color:#EAF2FA;
    font-family:'Manrope',sans-serif;
}
.block-container {max-width:1720px;padding-top:1.2rem;padding-bottom:2rem;}
[data-testid="stSidebar"] {background:#0C1624;border-right:1px solid #263A50;}
[data-testid="stSidebar"] * {color:#DCE8F4;}
h1,h2,h3,h4 {color:#F2F7FC !important;letter-spacing:-.035em;}
p,label {color:#DCE8F4;}
.mono,.eyebrow,.tiny,.metric-label,.panel-title {
    font-family:'DM Mono',monospace;
}
.eyebrow {
    color:#42D9E8;font-size:10px;letter-spacing:.16em;
    text-transform:uppercase;
}
.topbar {
    display:flex;align-items:center;justify-content:space-between;gap:18px;
    border:1px solid #263A50;border-left:3px solid #42D9E8;
    background:linear-gradient(100deg,#122233,#0C1826);
    padding:16px 20px;margin-bottom:15px;
}
.brand {font-size:19px;font-weight:800;color:#EAF7FC;letter-spacing:-.03em;}
.topmeta {color:#91A5BA;font:11px 'DM Mono',monospace;text-align:right;}
.section-kicker {
    color:#42D9E8;font:10px 'DM Mono',monospace;
    letter-spacing:.16em;text-transform:uppercase;margin:12px 0 5px;
}
.section-heading {color:#F0F6FC;font-size:21px;font-weight:800;}
.section-note {color:#91A5BA;font-size:12px;margin:3px 0 12px;}
.panel {
    background:linear-gradient(145deg,#111E2E,#0D1826);
    border:1px solid #263A50;padding:15px 17px;
}
.panel-title {
    color:#DDEAF5;font-size:11px;text-transform:uppercase;
    letter-spacing:.12em;margin-bottom:11px;
}
.metric-label {color:#91A5BA;font-size:10px;letter-spacing:.1em;text-transform:uppercase;}
.metric-value {color:#F4F8FC;font-size:27px;font-weight:800;line-height:1.15;margin-top:5px;}
.metric-unit {color:#91A5BA;font-size:11px;margin-top:3px;}
.status-pill {
    display:inline-block;padding:5px 9px;border:1px solid #31516A;
    color:#42E0AE;background:rgba(66,224,174,.08);
    font:10px 'DM Mono',monospace;letter-spacing:.08em;
}
.status-off {color:#F05D78;border-color:#71394A;background:rgba(240,93,120,.08);}
.callout {
    border-left:3px solid #42D9E8;background:rgba(66,217,232,.07);
    padding:10px 13px;color:#BFD0DF;font-size:12px;
}
.tiny {color:#8298AE;font-size:10px;}
.stButton>button {
    background:linear-gradient(100deg,#1C7991,#216A9C);
    color:white;border:1px solid #3AAFC2;border-radius:3px;
    font-weight:800;min-height:42px;
}
.stButton>button:hover {background:#2697AD;color:white;border-color:#5BE4EF;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div {
    background:#101E2E;border-color:#344B62;border-radius:3px;
}
[data-testid="stMetric"] {
    background:#101B2A;border:1px solid #263A50;padding:12px 14px;
}
[data-testid="stMetricLabel"] {color:#91A5BA !important;font-family:'DM Mono',monospace !important;}
[data-testid="stMetricValue"] {color:#F2F7FC !important;font-weight:800;}
.stTabs [data-baseweb="tab-list"] {gap:3px;background:#0D1927;border:1px solid #263A50;padding:4px;}
.stTabs [data-baseweb="tab"] {
    color:#91A5BA;font:11px 'DM Mono',monospace;
    text-transform:uppercase;letter-spacing:.06em;padding:10px 14px;
}
.stTabs [aria-selected="true"] {color:#42D9E8 !important;background:#172B3E !important;}
hr {border-color:#263A50;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA AND API
# ============================================================

@st.cache_data(show_spinner=False)
def load_validation_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Validation data not found: {DATA_PATH}")

    with np.load(DATA_PATH) as data:
        X_val = data["X_val"]
        y_val = data["y_val"]

    if X_val.ndim != 3 or X_val.shape[1:] != (50, 18):
        raise ValueError(f"Unexpected X_val shape: {X_val.shape}")

    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val lengths do not match.")

    if not np.isfinite(X_val).all() or not np.isfinite(y_val).all():
        raise ValueError("Validation data contains non-finite values.")

    return X_val, y_val


def check_api():
    try:
        request = urllib.request.Request(f"{API_URL}/health")
        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.loads(response.read().decode())

        return (
            result.get("status") == "healthy"
            and result.get("model_loaded") is True
        )
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False


def predict_rul(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.shape != (50, 18):
        raise ValueError(f"Expected (50, 18), got {sequence.shape}")

    if not np.isfinite(sequence).all():
        raise ValueError("Sequence contains non-finite values.")

    payload = json.dumps({"sequence": sequence.tolist()}).encode()

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
        raise ValueError("Model returned a non-finite prediction.")

    return max(0.0, prediction)


def classify_priority(rul):
    # Illustrative thresholds only.
    if rul <= 20:
        return "High"
    if rul <= 50:
        return "Medium"
    return "Low"


def explain_sequence_features(sequence, X_val):
    baseline = np.median(
        X_val.reshape(-1, X_val.shape[-1]), axis=0
    )

    original = predict_rul(sequence)
    rows = []

    for index, feature in enumerate(FEATURE_COLUMNS):
        if feature == "cycle":
            continue

        occluded = sequence.copy()
        occluded[:, index] = baseline[index]

        changed = predict_rul(occluded)
        signed = changed - original

        rows.append({
            "Feature": feature,
            "Original Prediction": original,
            "Replacement Prediction": changed,
            "Signed Change": signed,
            "Absolute Change": abs(signed),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Absolute Change", ascending=False)
        .reset_index(drop=True),
        original,
    )


def predict_fleet(X_val, y_val, indices):
    rows = []
    progress = st.progress(0, text="Analyzing validation sequences...")

    try:
        for position, index in enumerate(indices, start=1):
            predicted = predict_rul(X_val[index])
            actual = float(y_val[index])

            rows.append({
                "Engine": f"SIM-{index+1:04d}",
                "Sequence Index": int(index),
                "Predicted RUL": predicted,
                "Actual RUL": actual,
                "Signed Error": predicted - actual,
                "Absolute Error": abs(predicted - actual),
                "Priority": classify_priority(predicted),
            })

            progress.progress(
                position / len(indices),
                text=f"Analyzed {position} of {len(indices)} sequences",
            )
    finally:
        progress.empty()

    return pd.DataFrame(rows)


# ============================================================
# DIAGRAMS AND VISUAL HELPERS
# ============================================================

def panel_header(kicker, title, note=None):
    st.markdown(
        f'<div class="section-kicker">{kicker}</div>'
        f'<div class="section-heading">{title}</div>'
        + (f'<div class="section-note">{note}</div>' if note else ""),
        unsafe_allow_html=True,
    )


def metric_panel(label, value, unit="", accent="#42D9E8"):
    st.markdown(f"""
    <div class="panel" style="border-top:2px solid {accent};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>
    """, unsafe_allow_html=True)


def make_gauge(value, maximum, title, suffix="", color=None):
    color = color or C["cyan"]
    ratio = 0 if maximum <= 0 else float(np.clip(value / maximum, 0, 1))

    fig, ax = plt.subplots(figsize=(3.1, 2.15), dpi=120)
    fig.patch.set_facecolor(C["panel"])
    ax.set_facecolor(C["panel"])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-.25, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Wedge((0, 0), 1, 0, 180, width=.16, color="#26394C"))
    ax.add_patch(Wedge((0, 0), 1, 0, 180*ratio, width=.16, color=color))

    ax.text(0, .18, f"{value:.1f}{suffix}", ha="center",
            color="#F2F7FC", fontsize=18, fontweight="bold")
    ax.text(0, -.08, title.upper(), ha="center",
            color="#91A5BA", fontsize=8, family="monospace")

    fig.tight_layout(pad=0)
    return fig


def draw_engine_schematic():
    """Conceptual turbofan illustration, not a physical sensor map."""

    fig, ax = plt.subplots(figsize=(11, 4.3), dpi=130)
    fig.patch.set_facecolor(C["panel"])
    ax.set_facecolor(C["panel"])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (1.05, 1.45), 9.7, 2.05,
        boxstyle="round,pad=.12,rounding_size=.35",
        linewidth=1.8, edgecolor="#4D7895", facecolor="#172B3E"
    ))

    ax.add_patch(FancyBboxPatch(
        (2.1, 1.78), 7.7, 1.38,
        boxstyle="round,pad=.08,rounding_size=.28",
        linewidth=1.1, edgecolor="#31536C", facecolor="#0E1B29"
    ))

    # Fan
    ax.add_patch(Circle((2, 2.48), .78, edgecolor=C["cyan"],
                        facecolor="#122B3B", linewidth=2))

    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x2 = 2 + .68*np.cos(angle)
        y2 = 2.48 + .68*np.sin(angle)
        ax.plot([2, x2], [2.48, y2], color="#6CBACB", lw=2)

    ax.add_patch(Circle((2, 2.48), .17, color=C["cyan"]))

    # Compressor
    for x, radius in [(3.35, .52), (4.15, .43), (4.85, .35)]:
        ax.add_patch(Circle((x, 2.48), radius, fill=False,
                            edgecolor="#5D91B1", linewidth=1.5))
        ax.plot([x, x], [2.48-radius, 2.48+radius],
                color="#315A75", lw=1)

    # Combustor
    ax.add_patch(FancyBboxPatch(
        (5.55, 1.95), 1.65, 1.05,
        boxstyle="round,pad=.08",
        edgecolor="#C78D45", facecolor="#3A2B20", linewidth=1.5
    ))

    # Turbine
    for x in [7.65, 8.35, 9.05]:
        ax.add_patch(Circle((x, 2.48), .48, fill=False,
                            edgecolor="#B58B62", linewidth=1.5))
        ax.plot([x-.28, x+.28], [2.2, 2.76],
                color="#B58B62", lw=1.2)

    ax.add_patch(FancyArrowPatch(
        (10.1, 2.48), (11.2, 2.48),
        arrowstyle="-|>", mutation_scale=18,
        color=C["green"], linewidth=2
    ))

    ax.text(2, 3.52, "FAN", color=C["cyan"], ha="center", fontsize=8)
    ax.text(4.1, 3.1, "COMPRESSOR", color="#83B7D4", ha="center", fontsize=8)
    ax.text(6.35, 3.25, "COMBUSTOR", color=C["amber"], ha="center", fontsize=8)
    ax.text(8.35, 3.15, "TURBINE", color="#D7AE7C", ha="center", fontsize=8)
    ax.text(10.7, 2.05, "EXHAUST", color=C["green"], ha="center", fontsize=8)

    callouts = [
        (3.0, 4.45, "sensor_2", C["cyan"]),
        (5.35, 4.45, "sensor_4", C["amber"]),
        (8.0, 4.45, "sensor_11", C["red"]),
        (9.7, .55, "sensor_17", C["purple"]),
        (5.5, .55, "sensor_12", C["green"]),
        (2.7, .55, "sensor_7", C["blue"]),
    ]

    for x, y, label, color in callouts:
        anchor_y = 3.25 if y > 3 else 1.7
        end_y = y + .18 if y > 3 else y + .3
        ax.plot([x, x], [anchor_y, end_y],
                color=color, alpha=.75, lw=1, linestyle="--")
        ax.scatter([x], [anchor_y], s=24, color=color, zorder=5)
        ax.text(
            x, y, label, color=color, ha="center", va="center",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=.35", fc="#0C1927", ec=color, lw=.8)
        )

    ax.text(
        6, .08,
        "CONCEPTUAL ENGINE SCHEMATIC · FEATURE CALLOUTS ARE NOT PHYSICAL SENSOR LOCATIONS",
        color="#7890A6", ha="center", fontsize=7, family="monospace"
    )

    fig.tight_layout(pad=.5)
    return fig


def draw_workflow():
    fig, ax = plt.subplots(figsize=(12, 2.15), dpi=130)
    fig.patch.set_facecolor(C["panel"])
    ax.set_facecolor(C["panel"])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2)
    ax.axis("off")

    labels = [
        ("NASA C-MAPSS", "FD001 DATA"),
        ("PREPROCESS", "SCALE + SELECT"),
        ("SEQUENCES", "50 × 18 INPUT"),
        ("LSTM MODEL", "PYTORCH"),
        ("FASTAPI", "POST /predict"),
        ("RUL OUTPUT", "CYCLES"),
    ]

    xs = np.linspace(.95, 11.05, len(labels))

    for i, (x, (title, sub)) in enumerate(zip(xs, labels)):
        color = C["cyan"] if i in (0, 3, 5) else C["blue"]

        ax.add_patch(FancyBboxPatch(
            (x-.78, .55), 1.56, .92,
            boxstyle="round,pad=.08",
            fc="#14283B", ec=color, lw=1.2
        ))

        ax.text(x, 1.12, title, ha="center",
                color="#F0F6FC", fontsize=8, fontweight="bold")
        ax.text(x, .82, sub, ha="center",
                color="#91A5BA", fontsize=6.5, family="monospace")

        if i < len(xs)-1:
            ax.add_patch(FancyArrowPatch(
                (x+.82, 1.01), (xs[i+1]-.84, 1.01),
                arrowstyle="-|>", mutation_scale=12,
                color="#4E809E", lw=1.2
            ))

    fig.tight_layout(pad=.2)
    return fig


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors="#A8BACB", labelsize=8)
    ax.grid(True, color="#7890A6", alpha=.16, linewidth=.6)

    for spine in ax.spines.values():
        spine.set_color("#30465B")

    if title:
        ax.set_title(title, color="#EAF2FA", fontsize=10, pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color="#A8BACB", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color="#A8BACB", fontsize=8)


# ============================================================
# LOAD DATA
# ============================================================

try:
    X_val, y_val = load_validation_data()
except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
    st.error(f"Could not load validation data: {exc}")
    st.stop()

api_healthy = check_api()

for key in ("fleet_results", "individual_result", "feature_explanation"):
    if key not in st.session_state:
        st.session_state[key] = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ◉ RUL COMMAND CENTRE")
    st.markdown('<div class="eyebrow">Predictive maintenance analytics</div>',
                unsafe_allow_html=True)

    st.divider()

    if api_healthy:
        st.markdown('<span class="status-pill">● API ONLINE</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-off">● API OFFLINE</span>',
                    unsafe_allow_html=True)

    st.markdown("### MODEL CONFIGURATION")
    st.write("**Architecture** · 2-layer LSTM")
    st.write("**Hidden size** · 64")
    st.write("**Dropout** · 0.20")
    st.write("**Inference device** · CPU")

    st.divider()

    st.markdown("### DATASET")
    st.write("**Source** · NASA C-MAPSS FD001")
    st.write(f"**Validation sequences** · {len(X_val):,}")
    st.write("**Window** · 50 cycles")
    st.write("**Features** · 18")

    st.divider()
    st.caption("Academic demonstration using held-out simulated validation sequences. Not live aircraft telemetry.")


# ============================================================
# HEADER AND TABS
# ============================================================

api_label = "API ONLINE" if api_healthy else "API OFFLINE"
api_color = C["green"] if api_healthy else C["red"]

st.markdown(f"""
<div class="topbar">
    <div>
        <div class="eyebrow">MACHINE LEARNING / ENGINE HEALTH ANALYTICS</div>
        <div class="brand">REMAINING USEFUL LIFE <span style="color:#42D9E8">/</span> COMMAND CENTRE</div>
    </div>
    <div class="topmeta">
        <span style="color:{api_color}">● {api_label}</span><br>
        NASA C-MAPSS FD001 | LSTM REGRESSOR | 50 × 18
    </div>
</div>
""", unsafe_allow_html=True)

overview_tab, fleet_tab, engine_tab, explain_tab, performance_tab = st.tabs([
    "01 / COMMAND OVERVIEW",
    "02 / FLEET ANALYTICS",
    "03 / ENGINE INSPECTOR",
    "04 / EXPLAINABILITY",
    "05 / MODEL PERFORMANCE",
])


# ============================================================
# 01 — COMMAND OVERVIEW
# ============================================================

with overview_tab:
    panel_header(
        "LIVE SYSTEM VIEW",
        "Engine health monitoring",
        "Validation-set command view · simulated sequences · model-backed analysis"
    )

    left, right = st.columns([1.65, 1], gap="medium")

    with left:
        st.markdown(
            '<div class="panel"><div class="panel-title">ENGINE CORE / CONCEPTUAL SENSOR MAP</div>',
            unsafe_allow_html=True
        )
        st.pyplot(draw_engine_schematic(), use_container_width=True)
        st.markdown(
            '<div class="tiny">Sensor labels represent model inputs only; positions are conceptual, not physical mappings.</div></div>',
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            '<div class="panel"><div class="panel-title">SYSTEM READINESS</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="metric-label">PREDICTION SERVICE</div>'
            f'<div style="font-size:20px;font-weight:800;color:{api_color};margin:5px 0 15px">{api_label}</div>',
            unsafe_allow_html=True
        )

        items = [
            ("VALIDATION DATA", "LOADED", C["green"]),
            ("LSTM CHECKPOINT", "API-REPORTED LOADED" if api_healthy else "NOT VERIFIED",
             C["green"] if api_healthy else C["amber"]),
            ("INPUT CONTRACT", "50 × 18 FEATURES", C["cyan"]),
            ("INFERENCE DEVICE", "CPU", C["muted"]),
        ]

        for label, value, color in items:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;border-top:1px solid #263A50;padding:10px 0">'
                f'<span class="tiny">{label}</span>'
                f'<span style="color:{color};font:10px DM Mono,monospace">{value}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="panel" style="margin-top:12px"><div class="panel-title">DATASET SNAPSHOT</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="metric-value">{len(X_val):,}</div>'
            '<div class="metric-unit">HELD-OUT VALIDATION SEQUENCES</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div style="display:flex;gap:8px;margin-top:15px">
            <div style="flex:1;background:#172B3E;padding:9px">
                <div class="metric-label">WINDOW</div><b>50</b>
                <div class="tiny">cycles</div>
            </div>
            <div style="flex:1;background:#172B3E;padding:9px">
                <div class="metric-label">INPUTS</div><b>18</b>
                <div class="tiny">features</div>
            </div>
        </div></div>
        """, unsafe_allow_html=True)

    panel_header(
        "INFERENCE PIPELINE",
        "From sensor sequence to RUL estimate",
        "Actual project workflow represented as a connected processing chain"
    )

    st.pyplot(draw_workflow(), use_container_width=True)

    st.markdown(
        '<div class="panel"><div class="panel-title">50-CYCLE INPUT WINDOW / SAMPLE SEQUENCE</div>',
        unsafe_allow_html=True
    )

    sample_idx = st.selectbox(
        "Preview validation sequence",
        options=list(range(min(len(X_val), 100))),
        format_func=lambda i: f"SIM-{i+1:04d}",
        key="overview_sequence",
    )

    seq = X_val[sample_idx]

    fig, ax = plt.subplots(figsize=(11, 2.3), dpi=120)
    fig.patch.set_facecolor(C["panel"])
    ax.set_facecolor(C["panel"])

    selected_features = [4, 6, 10, 11, 15]

    for row, feature_index in enumerate(selected_features):
        values = seq[:, feature_index]
        low, high = values.min(), values.max()

        for col, value in enumerate(values):
            intensity = (value-low)/(high-low+1e-8)
            ax.add_patch(Rectangle(
                (col, row), .88, .72,
                color=plt.cm.viridis(.18+.72*intensity)
            ))

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 5)
    ax.set_yticks(np.arange(5)+.36)
    ax.set_yticklabels(
        ["sensor_2", "sensor_4", "sensor_11", "sensor_12", "sensor_17"],
        color="#B9CBD9", fontsize=8
    )
    ax.set_xticks([0, 10, 20, 30, 40, 49])
    ax.set_xticklabels(
        ["t-49", "t-40", "t-30", "t-20", "t-10", "latest"],
        color="#91A5BA", fontsize=8
    )
    ax.tick_params(axis="x", length=0)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="callout">All records shown are held-out NASA C-MAPSS FD001 validation sequences. They are simulated examples, not a live aircraft fleet.</div>',
        unsafe_allow_html=True
    )


# ============================================================
# 02 — FLEET ANALYTICS
# ============================================================

with fleet_tab:
    panel_header(
        "MULTI-SEQUENCE ANALYSIS",
        "Validation fleet analytics",
        "Run model inference across a selectable sample of held-out validation sequences"
    )

    max_fleet_size = min(20, len(X_val))
    min_fleet_size = min(3, max_fleet_size)

    fleet_size = st.slider(
        "Sequences to analyze",
        min_value=min_fleet_size,
        max_value=max_fleet_size,
        value=min(5, max_fleet_size),
        key="fleet_size"
    )

    fleet_indices = np.linspace(
        0, len(X_val)-1, num=fleet_size, dtype=int
    )

    run_col, note_col = st.columns([1, 3])

    with run_col:
        run_fleet = st.button(
            "▶ RUN FLEET INFERENCE",
            type="primary",
            disabled=not api_healthy,
            key="fleet_button"
        )

    with note_col:
        st.markdown(
            '<div class="tiny" style="padding-top:12px">Each selected sequence is sent to the FastAPI prediction endpoint.</div>',
            unsafe_allow_html=True
        )

    if run_fleet:
        try:
            with st.spinner("Running LSTM predictions..."):
                st.session_state.fleet_results = predict_fleet(
                    X_val, y_val, fleet_indices
                )
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError) as exc:
            st.error(f"Fleet analysis failed: {exc}")

    fleet_df = st.session_state.fleet_results

    if fleet_df is not None and not fleet_df.empty:
        expected = set(map(int, fleet_indices))
        actual_indices = set(fleet_df["Sequence Index"].astype(int))

        if expected != actual_indices:
            st.info("The selected sample size changed. Run Fleet Inference again.")
        else:
            fleet_df = fleet_df.sort_values(
                "Predicted RUL"
            ).reset_index(drop=True)

            high = int((fleet_df["Priority"] == "High").sum())
            medium = int((fleet_df["Priority"] == "Medium").sum())
            low = int((fleet_df["Priority"] == "Low").sum())

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                metric_panel("ANALYZED SEQUENCES", f"{len(fleet_df):02d}",
                             "selected validation sample", C["cyan"])
            with m2:
                metric_panel("HIGH PRIORITY", f"{high:02d}",
                             "illustrative threshold ≤ 20 cycles", C["red"])
            with m3:
                metric_panel("MEDIUM PRIORITY", f"{medium:02d}",
                             "illustrative threshold ≤ 50 cycles", C["amber"])
            with m4:
                metric_panel("LOW PRIORITY", f"{low:02d}",
                             "illustrative threshold > 50 cycles", C["green"])

            lcol, rcol = st.columns([1.45, 1], gap="medium")

            with lcol:
                st.markdown(
                    '<div class="panel"><div class="panel-title">RUL RANKING / LOWEST PREDICTED FIRST</div>',
                    unsafe_allow_html=True
                )

                plot_df = fleet_df.sort_values("Predicted RUL", ascending=True)

                fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
                fig.patch.set_facecolor(C["panel"])

                bar_colors = [
                    C["red"] if p == "High"
                    else C["amber"] if p == "Medium"
                    else C["green"]
                    for p in plot_df["Priority"]
                ]

                ax.barh(
                    plot_df["Engine"],
                    plot_df["Predicted RUL"],
                    color=bar_colors,
                    alpha=.9
                )
                ax.invert_yaxis()
                style_axes(ax, "Predicted remaining useful life",
                           "Predicted RUL (cycles)")

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.markdown('</div>', unsafe_allow_html=True)

            with rcol:
                st.markdown(
                    '<div class="panel"><div class="panel-title">SAMPLE ERROR SUMMARY</div>',
                    unsafe_allow_html=True
                )

                mae = float(fleet_df["Absolute Error"].mean())
                rmse = float(np.sqrt(np.mean(fleet_df["Signed Error"]**2)))
                mean_error = float(fleet_df["Signed Error"].mean())

                st.pyplot(
                    make_gauge(mae, max(25, mae*1.5),
                               "Sample MAE", " cyc", C["cyan"]),
                    use_container_width=True
                )

                st.metric("Sample RMSE", f"{rmse:.2f} cycles")
                st.metric("Mean signed error", f"{mean_error:+.2f} cycles")

                st.markdown('</div>', unsafe_allow_html=True)

            a, b = st.columns(2)

            with a:
                st.markdown(
                    '<div class="panel"><div class="panel-title">PREDICTED VS ACTUAL</div>',
                    unsafe_allow_html=True
                )

                fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
                fig.patch.set_facecolor(C["panel"])

                ax.scatter(
                    fleet_df["Actual RUL"],
                    fleet_df["Predicted RUL"],
                    s=65, color=C["cyan"],
                    edgecolors="#D8F8FC", linewidths=.5
                )

                lo = min(0, fleet_df["Actual RUL"].min(),
                         fleet_df["Predicted RUL"].min())
                hi = max(fleet_df["Actual RUL"].max(),
                         fleet_df["Predicted RUL"].max(), 1)

                ax.plot([lo, hi], [lo, hi], "--",
                        color=C["amber"], lw=1)

                style_axes(ax, "Prediction agreement",
                           "Actual RUL (cycles)", "Predicted RUL (cycles)")

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.markdown('</div>', unsafe_allow_html=True)

            with b:
                st.markdown(
                    '<div class="panel"><div class="panel-title">ABSOLUTE ERROR DISTRIBUTION</div>',
                    unsafe_allow_html=True
                )

                fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
                fig.patch.set_facecolor(C["panel"])

                ax.hist(
                    fleet_df["Absolute Error"],
                    bins=min(10, max(3, len(fleet_df))),
                    color=C["purple"],
                    edgecolor="#D9D1FF",
                    alpha=.85
                )

                style_axes(ax, "Prediction error frequency",
                           "Absolute error (cycles)", "Sequence count")

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="panel"><div class="panel-title">SEQUENCE REGISTER / ANALYSIS OUTPUT</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                fleet_df[[
                    "Engine", "Predicted RUL", "Actual RUL",
                    "Absolute Error", "Priority"
                ]].round(2),
                use_container_width=True,
                hide_index=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

            st.caption(
                "Metrics describe only the selected validation sample. Priority thresholds are illustrative, not validated maintenance limits."
            )

    elif not api_healthy:
        st.warning(
            "Prediction API is offline. Start it in another terminal with: "
            "uvicorn src.api.main:app --reload"
        )

    else:
        st.markdown(
            '<div class="callout">No fleet inference has been run yet. Select a sample size and run inference to populate the analytics panels.</div>',
            unsafe_allow_html=True
        )


# ============================================================
# 03 — ENGINE INSPECTOR
# ============================================================

with engine_tab:
    panel_header(
        "SINGLE-SEQUENCE DIAGNOSTICS",
        "Engine sequence inspector",
        "Inspect one simulated validation sequence, its RUL estimate, and sensor behaviour"
    )

    engine_index = st.selectbox(
        "Select validation sequence",
        options=list(range(len(X_val))),
        format_func=lambda i:
            f"SIM-{i+1:04d} · actual RUL {float(y_val[i]):.0f} cycles",
        key="engine_select"
    )

    sequence = X_val[engine_index]

    left, right = st.columns([1.1, 1], gap="medium")

    with left:
        st.markdown(
            '<div class="panel"><div class="panel-title">SEQUENCE PROFILE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="metric-value">SIM-{engine_index+1:04d}</div>'
            '<div class="metric-unit">VALIDATION SEQUENCE IDENTIFIER</div>',
            unsafe_allow_html=True
        )

        x1, x2, x3 = st.columns(3)
        x1.metric("Time steps", sequence.shape[0])
        x2.metric("Features", sequence.shape[1])
        x3.metric("Actual RUL", f"{float(y_val[engine_index]):.0f} cyc")

        st.markdown(
            '<div class="tiny">INPUT TENSOR · 50 TIME STEPS × 18 FEATURES</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="panel"><div class="panel-title">LSTM INFERENCE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-label">PREDICTION ENDPOINT</div>'
            '<div style="font:18px DM Mono,monospace;color:#42D9E8;margin:5px 0 12px">POST /predict</div>',
            unsafe_allow_html=True
        )

        if not api_healthy:
            st.warning("Prediction API is offline.")

        if st.button(
            "▶ ESTIMATE REMAINING USEFUL LIFE",
            type="primary",
            disabled=not api_healthy,
            key="individual_predict"
        ):
            try:
                with st.spinner("Running LSTM inference..."):
                    predicted = predict_rul(sequence)

                st.session_state.individual_result = {
                    "index": int(engine_index),
                    "predicted": predicted,
                    "actual": float(y_val[engine_index])
                }

            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError) as exc:
                st.error(f"Prediction failed: {exc}")

        st.markdown('</div>', unsafe_allow_html=True)

    result = st.session_state.individual_result

    if result is not None and result["index"] == engine_index:
        predicted = float(result["predicted"])
        actual = float(result["actual"])
        error = abs(predicted-actual)
        priority = classify_priority(predicted)

        accent = (
            C["red"] if priority == "High"
            else C["amber"] if priority == "Medium"
            else C["green"]
        )

        st.markdown(
            '<div class="section-kicker">MODEL OUTPUT</div>'
            '<div class="section-heading">Prediction result</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_panel("PREDICTED RUL", f"{predicted:.2f}",
                         "cycles", C["cyan"])
        with c2:
            metric_panel("ACTUAL RUL", f"{actual:.2f}",
                         "cycles · validation label", C["blue"])
        with c3:
            metric_panel("ABSOLUTE ERROR", f"{error:.2f}",
                         "cycles", C["amber"])

        st.markdown(
            f'<div class="callout" style="border-left-color:{accent}">'
            f'ILLUSTRATIVE PRIORITY · <b style="color:{accent}">{priority.upper()}</b>. '
            'Thresholds are demonstration-only and are not operational maintenance guidance.</div>',
            unsafe_allow_html=True
        )

    elif result is not None:
        st.info("Run the prediction for the currently selected sequence.")

    st.markdown(
        '<div class="section-kicker">SENSOR TELEMETRY VIEW</div>'
        '<div class="section-heading">50-cycle feature trajectories</div>',
        unsafe_allow_html=True
    )

    sensor_features = [f for f in FEATURE_COLUMNS if f != "cycle"]

    chosen_sensors = st.multiselect(
        "Select features to plot",
        options=sensor_features,
        default=["sensor_2", "sensor_4", "sensor_11"],
        key="sensor_select"
    )

    if chosen_sensors:
        fig, ax = plt.subplots(figsize=(11, 4.2), dpi=120)
        fig.patch.set_facecolor(C["panel"])

        for feature in chosen_sensors:
            idx = FEATURE_COLUMNS.index(feature)
            ax.plot(
                np.arange(1, 51),
                sequence[:, idx],
                label=feature,
                linewidth=1.8
            )

        style_axes(
            ax,
            "Selected feature values across input window",
            "Time step in 50-cycle window",
            "Stored feature value"
        )

        ax.legend(
            facecolor="#142235",
            edgecolor="#263A50",
            labelcolor="#EAF2FA",
            fontsize=8,
            ncol=min(4, len(chosen_sensors))
        )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    else:
        st.info("Choose at least one feature to display its trajectory.")


# ============================================================
# 04 — EXPLAINABILITY
# ============================================================

with explain_tab:
    panel_header(
        "LOCAL MODEL SENSITIVITY",
        "Feature occlusion analysis",
        "Replace one feature at a time with its validation-set median and observe the prediction change"
    )

    explain_index = st.selectbox(
        "Sequence for sensitivity analysis",
        options=list(range(len(X_val))),
        format_func=lambda i:
            f"SIM-{i+1:04d} · actual RUL {float(y_val[i]):.0f} cycles",
        key="explain_engine"
    )

    st.markdown(
        '<div class="callout">Each non-cycle feature is replaced across all 50 time steps with its validation-set median. The original cycle trajectory is preserved. This measures model sensitivity, not causality.</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "◎ RUN FEATURE OCCLUSION",
        type="primary",
        disabled=not api_healthy,
        key="explain_button"
    ):
        try:
            with st.spinner("Running feature-occlusion predictions..."):
                explanation_df, original_prediction = explain_sequence_features(
                    X_val[explain_index], X_val
                )

            st.session_state.feature_explanation = {
                "index": int(explain_index),
                "prediction": original_prediction,
                "results": explanation_df
            }

        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError) as exc:
            st.error(f"Explainability failed: {exc}")

    explanation = st.session_state.feature_explanation

    if explanation is not None and explanation["index"] == explain_index:
        df = explanation["results"]

        st.metric(
            "Original LSTM prediction",
            f"{explanation['prediction']:.2f} cycles"
        )

        chart_df = df.head(10).sort_values("Absolute Change")

        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        fig.patch.set_facecolor(C["panel"])

        bars = ax.barh(
            chart_df["Feature"],
            chart_df["Absolute Change"],
            color=[
                C["cyan"] if v >= chart_df["Absolute Change"].median()
                else C["blue"]
                for v in chart_df["Absolute Change"]
            ]
        )

        style_axes(ax, "Top feature sensitivity",
                   "Absolute prediction change (cycles)")

        for bar, value in zip(bars, chart_df["Absolute Change"]):
            ax.text(
                value,
                bar.get_y()+bar.get_height()/2,
                f"  {value:.2f}",
                color="#EAF2FA",
                va="center",
                fontsize=7
            )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(
            '<div class="panel"><div class="panel-title">OCCLUSION RESULTS REGISTER</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            df.round(3),
            use_container_width=True,
            hide_index=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.caption(
            "Positive signed change means the prediction increased after replacement; negative means it decreased. Magnitude is local sensitivity for this sequence and baseline."
        )

    elif explanation is not None:
        st.info("Run feature occlusion for the currently selected sequence.")

    elif not api_healthy:
        st.warning("The prediction API is offline. Start it with: uvicorn src.api.main:app --reload")


# ============================================================
# 05 — MODEL PERFORMANCE
# ============================================================

with performance_tab:
    panel_header(
        "VALIDATION METRICS",
        "Model performance analysis",
        "Evaluation of the held-out validation set using recorded project results"
    )

    st.markdown(
        '<div class="callout">Fresh sample-level metrics are calculated from fleet inference. Previously recorded full-validation metrics are shown separately.</div>',
        unsafe_allow_html=True
    )

    fleet_df = st.session_state.fleet_results

    if fleet_df is not None and not fleet_df.empty:
        mae = float(fleet_df["Absolute Error"].mean())
        rmse = float(np.sqrt(np.mean(fleet_df["Signed Error"]**2)))
        bias = float(fleet_df["Signed Error"].mean())
        p90 = float(np.percentile(fleet_df["Absolute Error"], 90))

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_panel("SAMPLE MAE", f"{mae:.2f}", "cycles", C["cyan"])
        with c2:
            metric_panel("SAMPLE RMSE", f"{rmse:.2f}", "cycles", C["blue"])
        with c3:
            metric_panel("MEAN SIGNED ERROR", f"{bias:+.2f}", "cycles", C["amber"])
        with c4:
            metric_panel("90TH PERCENTILE ABS ERROR", f"{p90:.2f}", "cycles", C["purple"])

        a, b = st.columns(2)

        with a:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
            fig.patch.set_facecolor(C["panel"])

            ax.scatter(
                fleet_df["Actual RUL"],
                fleet_df["Predicted RUL"],
                color=C["cyan"],
                s=60
            )

            lo = min(0, fleet_df["Actual RUL"].min(),
                     fleet_df["Predicted RUL"].min())
            hi = max(fleet_df["Actual RUL"].max(),
                     fleet_df["Predicted RUL"].max(), 1)

            ax.plot([lo, hi], [lo, hi], "--", color=C["amber"])

            style_axes(ax, "Actual vs predicted",
                       "Actual RUL (cycles)", "Predicted RUL (cycles)")

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with b:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
            fig.patch.set_facecolor(C["panel"])

            ax.hist(
                fleet_df["Absolute Error"],
                bins=min(10, max(3, len(fleet_df))),
                color=C["purple"],
                alpha=.85
            )

            style_axes(ax, "Absolute error distribution",
                       "Absolute error (cycles)", "Count")

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.caption("These metrics represent the selected fleet sample, not the complete validation set.")

    else:
        st.info("Run Fleet Inference in the Fleet Analytics tab to populate sample-level performance charts.")

    st.markdown(
        '<div class="panel"><div class="panel-title">RECORDED PROJECT EVALUATION / CONTEXT</div>',
        unsafe_allow_html=True
    )

    st.write("Previously recorded LSTM validation performance:")

    st.markdown(
        "**MAE:** approximately 13.50 cycles &nbsp;&nbsp; "
        "**RMSE:** approximately 19.78 cycles",
        unsafe_allow_html=True
    )

    st.write("Previously recorded baseline results:")

    baseline_df = pd.DataFrame([
        {"Model": "Linear Regression", "MAE (cycles)": 21.385, "RMSE (cycles)": 27.366},
        {"Model": "Random Forest", "MAE (cycles)": 20.553, "RMSE (cycles)": 28.040},
        {"Model": "XGBoost", "MAE (cycles)": 20.296, "RMSE (cycles)": 27.659},
    ])

    st.dataframe(
        baseline_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Baseline models use final-timestep features, whereas the LSTM uses the full 50-step sequence. This is not a perfectly controlled apples-to-apples comparison."
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:12px;color:#70869B;font:10px DM Mono,monospace;letter-spacing:.09em">
    REMAINING USEFUL LIFE PREDICTION AND PREDICTIVE MAINTENANCE<br>
    NASA C-MAPSS FD001 · PYTORCH LSTM · FASTAPI · STREAMLIT<br><br>
    ACADEMIC DEMONSTRATION · SIMULATED VALIDATION DATA · NOT A CERTIFIED AIRCRAFT MAINTENANCE SYSTEM
</div>
""", unsafe_allow_html=True)
