import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import re
import io
from io import BytesIO
from datetime import datetime

# === Streamlit Config ===
st.set_page_config(page_title="Response Curve Generator", layout="wide")
st.title("📈 Response Curve Generator")

# === MODE TOGGLE (TOP RIGHT) ===
title_col, mode_col = st.columns([3, 2])
with mode_col:
    mode = st.radio(
        "",
        ["Current Response Curves", "Simulate one response curve"],
        index=0,
        horizontal=True
    )

# === Instruction Message ===
st.markdown("""
**⚠️ Important:**  
Upload the **COE file exactly as received**, containing:
- `Decomps Vol`
- `Decomps Value`
- `Media KeyMetrics`
- `Media Spends`
- `Model Result`
- `Predictors Summary`
""")

status = st.empty()
uploaded_file = st.file_uploader("Upload MMM Results Excel file", type=["xlsx"])


# --- Guard clause
if not uploaded_file:
    st.info("⬆️ Upload a MMM results file to get started.")
    st.stop()


# ================================
# === LOAD + PREP DATA
# ================================
if uploaded_file:
    status.success("File uploaded successfully! Processing...")
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")

    required_sheets = [
        "Decomps Vol", "Decomps Value", "Media KeyMetrics",
        "Media Spends", "Model Result", "Predictors Summary"
    ]
    if not all(s in xls.sheet_names for s in required_sheets):
        status.error("❌ Missing required sheets.")
        st.stop()

    decomps_vol = xls.parse("Decomps Vol", usecols=["EndPeriod", "final_0_vol"])
    decomps_val = xls.parse("Decomps Value", usecols=["EndPeriod", "final_0_val"])
    media_metrics = xls.parse("Media KeyMetrics")
    media_spends = xls.parse("Media Spends")

    decomps_vol["EndPeriod"] = pd.to_datetime(decomps_vol["EndPeriod"])
    min_date = decomps_vol["EndPeriod"].min().date()
    max_date = decomps_vol["EndPeriod"].max().date()

    base_data = pd.merge(decomps_vol, decomps_val, on="EndPeriod")

    model_result = xls.parse("Model Result", header=None, skiprows=1).iloc[:, [2, 3]]
    model_result.columns = ["Variable Name", "Raw Name"]
    model_coeffs = xls.parse("Predictors Summary", header=None, skiprows=1).iloc[:, [1, 3]]
    model_coeffs.columns = ["Raw Name", "Coefficient"]

    media_channels = [c for c in media_metrics.columns if c not in ["EndPeriod", "custom_period", "Unnamed: 0"]]

    def extract_params(raw):
        if pd.isna(raw): return None, None, None
        raw = str(raw)
        hl = re.search(r"hlfl([\d\.]+)", raw)
        stp = re.search(r"step([\d\.]+)", raw)
        sat = re.search(r"sat([\d\.]+)", raw)
        return (
            float(hl.group(1)) if hl else None,
            float(stp.group(1)) if stp else None,
            float(sat.group(1)) if sat else None,
        )

    model_result[["Half-life", "Steepness", "Saturation"]] = model_result["Raw Name"].apply(
        lambda x: pd.Series(extract_params(x))
    )

    params_df = model_result.merge(model_coeffs, on="Raw Name", how="left")
    params_df = params_df[params_df["Variable Name"].isin(media_channels)]

    for df in [media_metrics, media_spends, base_data]:
        df["EndPeriod"] = pd.to_datetime(df["EndPeriod"])

    status.info("✅ File processed successfully!")

# ================================
# === SHARED FUNCTIONS
# ================================
def geometric_adstock(x, half_life):
    decay = 0.5 ** (1 / half_life)
    ad = np.zeros_like(x)
    ad[0] = x[0]
    for t in range(1, len(x)):
        ad[t] = x[t] + decay * ad[t - 1]
    return ad

def sigmoid_saturation(adstocked, steepness, saturation, max_adstock):
    return (
        max_adstock
        / (1 + np.exp(-10 * steepness / max_adstock * (adstocked - saturation * max_adstock)))
    ) - (
        max_adstock
        / (1 + np.exp(-10 * steepness / max_adstock * (0 - saturation * max_adstock)))
    )

def calculate_incremental(exec_w, base_vol, price, coef, hl, stp, sat, max_ad):
    adstocked = geometric_adstock(exec_w, hl)
    sat_exec = sigmoid_saturation(adstocked, stp, sat, max_ad)
    return sum((np.exp(coef * sat_exec[i]) - 1) * base_vol[i] * price[i]
               for i in range(len(exec_w)))

# ================================
# === OPTIONS UI
# ================================

if uploaded_file:
    st.subheader("Options")

    if mode == "Current Response Curves":
        selected_channels = st.multiselect(
            "Select channels",
            options=params_df["Variable Name"].tolist(),
            default=params_df["Variable Name"].tolist()
        )
    else:
        sim_channel = st.selectbox(
            "Media vehicle",
            params_df["Variable Name"].tolist()
        )


start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# Filter
mm_f = media_metrics[(media_metrics["EndPeriod"] >= pd.to_datetime(start_date)) &
                     (media_metrics["EndPeriod"] <= pd.to_datetime(end_date))]
ms_f = media_spends[(media_spends["EndPeriod"] >= pd.to_datetime(start_date)) &
                    (media_spends["EndPeriod"] <= pd.to_datetime(end_date))]
bd_f = base_data[(base_data["EndPeriod"] >= pd.to_datetime(start_date)) &
                 (base_data["EndPeriod"] <= pd.to_datetime(end_date))]

base_vol = bd_f["final_0_vol"].values
base_val = bd_f["final_0_val"].values
weekly_price = base_val / base_vol

# ================================
# === SIMULATION CONTROLS
# ================================
if mode == "Simulate one response curve":
    st.subheader("Simulation parameters")

    c1, c2, c3 = st.columns(3)
    with c1:
        half_life = st.slider("Half-life", 0.1, 5.0, 1.0, step=0.01)
    with c2:
        steepness = st.slider("Steepness", 0.1, 3.0, 1.0, step=0.01)
    with c3:
        saturation = st.slider("Saturation", 0.1, 0.9, 0.5, step=0.01)

# ================================
# === CURVE GENERATION
# ================================
st.subheader("Preview")

channels_to_plot = (
    selected_channels if mode == "Current Response Curves" else [sim_channel]
)

for ch in channels_to_plot:
    row = params_df[params_df["Variable Name"] == ch].iloc[0]

    coef = row["Coefficient"]
    hl = row["Half-life"] if mode == "Current Response Curves" else half_life
    stp = row["Steepness"] if mode == "Current Response Curves" else steepness
    sat = row["Saturation"] if mode == "Current Response Curves" else saturation
    
    if (
        pd.isna(hl) or pd.isna(stp) or pd.isna(sat)
        or hl <= 0 or stp <= 0 or sat <= 0
    ):
        continue

    exec_w = mm_f[ch].fillna(0).values
    curr_exec = np.sum(exec_w)
    curr_spend = ms_f[ch].sum()

    ad_real = geometric_adstock(exec_w, hl)
    max_ad = np.max(ad_real)

    curr_inc = calculate_incremental(
        exec_w, base_vol, weekly_price, coef, hl, stp, sat, max_ad
    )

    x_exec = np.linspace(0, 2 * curr_exec, 200)
    inc_vals, spend_vals = [], []

    for e in x_exec:
        r = e / curr_exec if curr_exec > 0 else 0
        inc_vals.append(
            calculate_incremental(exec_w * r, base_vol, weekly_price,
                                  coef, hl, stp, sat, max_ad)
        )
        spend_vals.append(curr_spend * r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spend_vals, y=inc_vals, mode="lines"))
    fig.add_trace(go.Scatter(x=[curr_spend], y=[curr_inc],
                             mode="markers", marker=dict(size=10, color="red")))
    fig.update_layout(
        title=f"{'SIMULATION' if mode!='Current Response Curves' else 'CURRENT'} – {ch}",
        xaxis_title="Spend",
        yaxis_title="Incremental Sales (Value)"
    )
    st.plotly_chart(fig, use_container_width=True)
