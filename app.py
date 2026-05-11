"""
app.py — John Deere AI Interpretability Demo
Streamlit app demonstrating model transparency for equipment failure prediction.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import shap

from pathlib import Path

from model import (
    load_data,
    train_model,
    get_shap_explainer,
    get_feature_ranges,
    FEATURES,
    FEATURE_LABELS,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Interpretability · John Deere",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* John Deere green accent */
    :root { --jd-green: #367C2B; --jd-yellow: #FFDE00; }

    [data-testid="stSidebar"] {
        background: #1a1a1a;
    }
    [data-testid="stSidebar"] * { color: #f0f0f0 !important; }
    [data-testid="stSidebar"] .stSlider label { color: #FFDE00 !important; }

    .metric-card {
        background: #f8f9fa;
        border-left: 5px solid #367C2B;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .metric-card h1 { font-size: 2.4rem; margin: 0; }
    .metric-card p  { margin: 0; color: #555; font-size: 0.85rem; }

    .risk-high  { border-left-color: #d62728 !important; }
    .risk-med   { border-left-color: #ff7f0e !important; }
    .risk-low   { border-left-color: #2ca02c !important; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #367C2B;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #FFDE00;
        padding-bottom: 4px;
    }
    .counterfactual-box {
        background: #fffbea;
        border: 1px solid #FFDE00;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load & cache model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on AI4I 2020 dataset…")
def load_everything():
    X, y, df = load_data()
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    explainer = get_shap_explainer(model, X_train)
    # Pre-compute global SHAP on a sample for speed
    sample = X_train.sample(min(300, len(X_train)), random_state=0)
    shap_global = explainer(sample)
    ranges = get_feature_ranges(X)
    return model, explainer, shap_global, sample, ranges, X_train, df


model, explainer, shap_global, shap_sample, ranges, X_train, df = load_everything()

# ── Sidebar — Input sliders ───────────────────────────────────────────────────
with st.sidebar:
    image_path = Path(__file__).with_name("john_deere_logo.png")
    st.image(
        str(image_path),
        width=160,
    )
    st.markdown("## ⚙️ Equipment Parameters")
    st.markdown("Adjust sensor readings to simulate a machine state:")

    air_temp = st.slider(
        "Air Temperature (K)",
        min_value=float(ranges["Air temperature [K]"]["min"]),
        max_value=float(ranges["Air temperature [K]"]["max"]),
        value=float(ranges["Air temperature [K]"]["mean"]),
        step=0.1,
        format="%.1f K",
    )

    proc_temp = st.slider(
        "Process Temperature (K)",
        min_value=float(ranges["Process temperature [K]"]["min"]),
        max_value=float(ranges["Process temperature [K]"]["max"]),
        value=float(ranges["Process temperature [K]"]["mean"]),
        step=0.1,
        format="%.1f K",
    )

    rot_speed = st.slider(
        "Rotational Speed (rpm)",
        min_value=int(ranges["Rotational speed [rpm]"]["min"]),
        max_value=int(ranges["Rotational speed [rpm]"]["max"]),
        value=int(ranges["Rotational speed [rpm]"]["mean"]),
        step=10,
        format="%d rpm",
    )

    torque = st.slider(
        "Torque (Nm)",
        min_value=float(ranges["Torque [Nm]"]["min"]),
        max_value=float(ranges["Torque [Nm]"]["max"]),
        value=float(ranges["Torque [Nm]"]["mean"]),
        step=0.5,
        format="%.1f Nm",
    )

    tool_wear = st.slider(
        "Tool Wear (min)",
        min_value=int(ranges["Tool wear [min]"]["min"]),
        max_value=int(ranges["Tool wear [min]"]["max"]),
        value=int(ranges["Tool wear [min]"]["mean"]),
        step=1,
        format="%d min",
    )

    st.markdown("---")
    st.markdown("**Demo Presets**")
    col_a, col_b = st.columns(2)
    preset_high = col_a.button("🔴 High Risk", use_container_width=True)
    preset_low  = col_b.button("🟢 Low Risk",  use_container_width=True)

# Handle presets via session state
if preset_high:
    st.session_state["preset"] = "high"
if preset_low:
    st.session_state["preset"] = "low"

# Override sliders with preset values when a preset was just clicked
if st.session_state.get("preset") == "high":
    air_temp   = 304.0
    proc_temp  = 312.0
    rot_speed  = 1200
    torque     = 65.0
    tool_wear  = 220
elif st.session_state.get("preset") == "low":
    air_temp   = 298.0
    proc_temp  = 307.0
    rot_speed  = 1550
    torque     = 38.0
    tool_wear  = 30

# ── Build input row ───────────────────────────────────────────────────────────
input_data = pd.DataFrame(
    [[air_temp, proc_temp, rot_speed, torque, tool_wear]],
    columns=FEATURES,
)

# ── Prediction ────────────────────────────────────────────────────────────────
prob_failure = model.predict_proba(input_data)[0][1]
risk_pct = prob_failure * 100

if risk_pct >= 60:
    risk_label, risk_emoji, risk_css = "HIGH RISK", "🔴", "risk-high"
elif risk_pct >= 30:
    risk_label, risk_emoji, risk_css = "MODERATE RISK", "🟡", "risk-med"
else:
    risk_label, risk_emoji, risk_css = "LOW RISK", "🟢", "risk-low"

# ── Local SHAP values ─────────────────────────────────────────────────────────
shap_local = explainer(input_data)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🚜 AI Interpretability Demo")
st.markdown(
    "**Workshop: Explainable AI for Industrial Systems** &nbsp;·&nbsp; "
    "Equipment Failure Prediction with SHAP explanations"
)
st.markdown("---")

# ── Part 1 — Black Box Prediction ────────────────────────────────────────────
st.markdown('<div class="section-header">Part 1 — Prediction</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.markdown(
        f"""
        <div class="metric-card {risk_css}">
            <p>Predicted Failure Probability</p>
            <h1>{risk_pct:.1f}%</h1>
            <p>{risk_emoji} {risk_label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <p>Air Temp</p>
            <h1 style="font-size:1.6rem">{air_temp:.1f} K</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <p>Torque</p>
            <h1 style="font-size:1.6rem">{torque:.1f} Nm</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <p>Tool Wear</p>
            <h1 style="font-size:1.6rem">{tool_wear} min</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <p>Rot. Speed</p>
            <h1 style="font-size:1.6rem">{rot_speed} rpm</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "> **Question for the audience:** The model says "
    f"**{risk_pct:.1f}% failure probability** — but *why?* "
    "This is the core problem interpretability solves."
)

# ── Part 2 — Local Explanation ───────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Part 2 — Local Explanation (This Prediction)</div>', unsafe_allow_html=True)
st.markdown(
    "SHAP (SHapley Additive exPlanations) shows how much each feature "
    "**pushed the prediction up or down** for *this specific machine state*."
)

col_wf, col_bar = st.columns(2)

with col_wf:
    st.markdown("**Waterfall Plot** — contribution of each feature")
    fig_wf, ax_wf = plt.subplots(figsize=(6, 4))
    shap.plots.waterfall(shap_local[0, :, 1], max_display=5, show=False)
    fig_wf = plt.gcf()
    fig_wf.patch.set_facecolor("white")
    st.pyplot(fig_wf, use_container_width=True)
    plt.close("all")

with col_bar:
    st.markdown("**Feature Contributions** — magnitude & direction")
    sv = shap_local.values[0, :, 1]
    labels = [FEATURE_LABELS[f] for f in FEATURES]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in sv]
    fig_b, ax_b = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(labels))
    ax_b.barh(y_pos, sv, color=colors, edgecolor="white", height=0.55)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(labels, fontsize=9)
    ax_b.axvline(0, color="black", linewidth=0.8)
    ax_b.set_xlabel("SHAP value (impact on failure probability)")
    ax_b.set_title("Local Feature Impact", fontsize=11, fontweight="bold")
    ax_b.set_facecolor("#fafafa")
    fig_b.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig_b, use_container_width=True)
    plt.close("all")

    # Narrative
    top_idx = int(np.argmax(np.abs(sv)))
    top_feat = FEATURE_LABELS[FEATURES[top_idx]]
    top_val  = sv[top_idx]
    direction = "increased" if top_val > 0 else "decreased"
    st.info(
        f"**Key driver:** *{top_feat}* {direction} failure risk the most "
        f"(SHAP = {top_val:+.4f})"
    )

# ── Part 3 — Global Explanation ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Part 3 — Global Explanation (Overall Model Behavior)</div>', unsafe_allow_html=True)
st.markdown(
    "Global SHAP shows **what the model relies on across all predictions** — "
    "not just this one machine."
)

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**Mean |SHAP| — Overall Feature Importance**")
    mean_abs = np.abs(shap_global.values).mean(axis=(0, 2))
    feat_imp = sorted(zip(FEATURES, mean_abs), key=lambda x: x[1], reverse=True)
    imp_labels = [FEATURE_LABELS[f] for f, _ in feat_imp]
    imp_vals   = [v for _, v in feat_imp]
    jd_greens  = ["#367C2B", "#4a9c3b", "#5db84e", "#85cc6f", "#b3e09a"]
    fig_gi, ax_gi = plt.subplots(figsize=(6, 4))
    bars = ax_gi.barh(
        range(len(imp_labels))[::-1],
        imp_vals,
        color=jd_greens,
        edgecolor="white",
        height=0.55,
    )
    ax_gi.set_yticks(range(len(imp_labels))[::-1])
    ax_gi.set_yticklabels(imp_labels, fontsize=9)
    ax_gi.set_xlabel("Mean |SHAP value|")
    ax_gi.set_title("Global Feature Importance", fontsize=11, fontweight="bold")
    ax_gi.set_facecolor("#fafafa")
    fig_gi.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig_gi, use_container_width=True)
    plt.close("all")

with col_g2:
    st.markdown("**Beeswarm Plot — Distribution of SHAP values**")
    fig_bee, ax_bee = plt.subplots(figsize=(6, 4))
    shap.plots.beeswarm(shap_global[:,:,0], max_display=5, show=False)
    fig_bee = plt.gcf()
    fig_bee.patch.set_facecolor("white")
    st.pyplot(fig_bee, use_container_width=True)
    plt.close("all")

# ── Part 4 — Counterfactual ───────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Part 4 — Counterfactual Reasoning ("What-If")</div>', unsafe_allow_html=True)
st.markdown(
    "Counterfactuals answer: **\"What would I need to change to reduce failure risk?\"** "
    "This turns AI explanations into *actionable maintenance decisions*."
)

st.markdown("#### Simulate an Intervention")
cf_col1, cf_col2 = st.columns(2)

with cf_col1:
    cf_wear_reduction = st.slider(
        "Reduce Tool Wear by (min)", 0, min(tool_wear, 200), 50, 5
    )
    cf_torque_reduction = st.slider(
        "Reduce Torque by (Nm)", 0.0, min(torque, 30.0), 5.0, 0.5
    )

cf_input = input_data.copy()
cf_input["Tool wear [min]"]  = max(0, tool_wear - cf_wear_reduction)
cf_input["Torque [Nm]"]      = max(0, torque - cf_torque_reduction)
cf_prob = model.predict_proba(cf_input)[0][1] * 100
delta   = risk_pct - cf_prob

with cf_col2:
    st.markdown(
        f"""
        <div class="counterfactual-box">
            <b>Original State</b><br>
            Tool Wear: {tool_wear} min &nbsp;|&nbsp; Torque: {torque:.1f} Nm<br>
            Failure Risk: <b>{risk_pct:.1f}%</b>
            <hr style="margin: 8px 0; border-color:#ddd">
            <b>After Intervention</b><br>
            Tool Wear: {cf_input['Tool wear [min]'].values[0]:.0f} min
            &nbsp;|&nbsp; Torque: {cf_input['Torque [Nm]'].values[0]:.1f} Nm<br>
            Failure Risk: <b>{cf_prob:.1f}%</b>
            <hr style="margin: 8px 0; border-color:#ddd">
            {"✅" if delta > 0 else "⚠️"} Risk change:
            <b style="color:{'#2ca02c' if delta > 0 else '#d62728'}">
                {'-' if delta > 0 else '+'}{abs(delta):.1f} percentage points
            </b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if delta > 5:
        st.success(
            f"Replacing the tool and reducing torque by {cf_torque_reduction:.1f} Nm "
            f"would cut predicted failure risk from **{risk_pct:.1f}%** to **{cf_prob:.1f}%** "
            f"— a **{delta:.1f} pp reduction**. This is how AI enables proactive maintenance."
        )
    elif delta <= 0:
        st.warning("Try increasing the intervention values to see a meaningful risk reduction.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Dataset: AI4I 2020 Predictive Maintenance · "
    "Model: Random Forest (scikit-learn) · "
    "Explanations: SHAP TreeExplainer · "
    "Built with Streamlit</small>",
    unsafe_allow_html=True,
)
