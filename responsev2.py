import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import re
import io
from datetime import datetime

# === Streamlit Config ===
st.set_page_config(page_title="Response Curve Generator", layout="wide")
st.title("📈 Response Curve Generator")

# === MODE TOGGLE (PRO UX) ===

st.markdown("## Switch Analysis Mode 🔁")
st.caption("Change between model‑fitted response curves and manual simulations of adstocked-saturated executions")

simulate_mode = st.toggle(
    "Simulate Curve with Custom Half-Life, Steepness, and Saturation Parameters",
    value=False
)

mode = (
    "Simulate Curve with Custom Half-Life, Steepness, and Saturation Parameters"
    if simulate_mode
    else "Current Response Curves"
)

# === Instruction Message ===
st.markdown("""
**⚠️ Important:**  
Make sure to upload the **COE file exactly as you received it**, and confirm that the following sheets exist with these exact names:  
- `Decomps Vol`  
- `Decomps Value`  
- `Media KeyMetrics`  
- `Media Spends`  
- `Model Result`  
- `Predictors Summary`  
""")

uploaded_file = st.file_uploader("Upload your MMM Results Excel file", type=["xlsx"])

if not uploaded_file:
    st.info("⬆️ Upload a MMM results file to get started.")
    st.stop()

# === Load Excel ===
xls = pd.ExcelFile(uploaded_file, engine="openpyxl")

required_sheets = [
    "Decomps Vol", "Decomps Value", "Media KeyMetrics",
    "Media Spends", "Model Result", "Predictors Summary"
]
if not all(s in xls.sheet_names for s in required_sheets):
    st.error("❌ Missing required sheets.")
    st.stop()

decomps_vol = xls.parse("Decomps Vol", usecols=["EndPeriod", "final_0_vol"])
decomps_val = xls.parse("Decomps Value", usecols=["EndPeriod", "final_0_val"])
media_metrics = xls.parse("Media KeyMetrics")
media_spends = xls.parse("Media Spends")
model_result = xls.parse("Model Result", header=None, skiprows=1).iloc[:, [2, 3]]
model_result.columns = ["Variable Name", "Raw Name"]
model_coeffs = xls.parse("Predictors Summary", header=None, skiprows=1).iloc[:, [1, 3]]
model_coeffs.columns = ["Raw Name", "Coefficient"]

# Dates
decomps_vol["EndPeriod"] = pd.to_datetime(decomps_vol["EndPeriod"])
media_metrics["EndPeriod"] = pd.to_datetime(media_metrics["EndPeriod"])
media_spends["EndPeriod"] = pd.to_datetime(media_spends["EndPeriod"])

min_date = decomps_vol["EndPeriod"].min().date()
max_date = decomps_vol["EndPeriod"].max().date()

base_data = decomps_vol.merge(decomps_val, on="EndPeriod")

media_channels = [
    c for c in media_metrics.columns
    if c not in ["EndPeriod", "custom_period", "Unnamed: 0"]
]

# === Extract params (UNCHANGED) ===
def extract_params(raw):
    if pd.isna(raw):
        return None, None, None
    raw = str(raw)
    hl = re.search(r"hlfl([\d\.]+)", raw)
    stp = re.search(r"step([\d\.]+)", raw)
    sat = re.search(r"sat([\d\.]+)", raw)
    return (
        float(hl.group(1)) if hl else None,
        float(stp.group(1)) if stp else None,
        float(sat.group(1)) if sat else None
    )

model_result[["Half-life", "Steepness", "Saturation"]] = (
    model_result["Raw Name"].apply(lambda x: pd.Series(extract_params(x)))
)

params_df = (
    model_result.merge(model_coeffs, on="Raw Name", how="left")
    .query("`Variable Name` in @media_channels")
)

# === FUNCTIONS (UNCHANGED) ===
def geometric_adstock(x, half_life):
    decay = 0.5 ** (1 / half_life)
    ad = np.zeros_like(x)
    ad[0] = x[0]
    for t in range(1, len(x)):
        ad[t] = x[t] + decay * ad[t - 1]
    return ad

def sigmoid_saturation(ad, stp, sat, max_ad):
    return (
        max_ad / (1 + np.exp(-10 * stp / max_ad * (ad - sat * max_ad)))
    ) - (
        max_ad / (1 + np.exp(-10 * stp / max_ad * (0 - sat * max_ad)))
    )

def calculate_incremental(exec_w, base_vol, price, coef, hl, stp, sat, max_ad):
    ad = geometric_adstock(exec_w, hl)
    sat_ad = sigmoid_saturation(ad, stp, sat, max_ad)
    return sum(
        (np.exp(coef * sat_ad[i]) - 1) * base_vol[i] * price[i]
        for i in range(len(exec_w))
    )

# === OPTIONS ===
st.subheader("Options")

if mode == "Current Response Curves":
    selected_channels = st.multiselect(
        "Select channels to plot",
        options=params_df["Variable Name"].tolist(),
        default=params_df["Variable Name"].tolist()
    )
else:
    sim_channel = st.selectbox(
        "Media vehicle",
        params_df["Variable Name"].tolist()
    )

start_date = st.date_input("Start Date", min_date)
end_date = st.date_input("End Date", max_date)

# === Filter data ===
mm_f = media_metrics[
    (media_metrics["EndPeriod"] >= pd.to_datetime(start_date)) &
    (media_metrics["EndPeriod"] <= pd.to_datetime(end_date))
]
ms_f = media_spends[
    (media_spends["EndPeriod"] >= pd.to_datetime(start_date)) &
    (media_spends["EndPeriod"] <= pd.to_datetime(end_date))
]
bd_f = base_data[
    (base_data["EndPeriod"] >= pd.to_datetime(start_date)) &
    (base_data["EndPeriod"] <= pd.to_datetime(end_date))
]

base_vol_weeks = bd_f["final_0_vol"].values
base_val_weeks = bd_f["final_0_val"].values
weekly_price = base_val_weeks / base_vol_weeks

# === Simulation sliders ===
if mode == "Simulate Curve with Custom Half-Life, Steepness, and Saturation Parameters":
    st.subheader("Simulation parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_hl = st.slider("Half-life", 0.1, 5.0, 1.0, step=0.01)
    with c2:
        sim_stp = st.slider("Steepness", 0.1, 3.0, 1.0, step=0.01)
    with c3:
        sim_sat = st.slider("Saturation", 0.1, 0.9, 0.5, step=0.01)

# === PREVIEW ===
st.subheader("Preview")

# === SHARED PLOTTING LOGIC WITH ORIGINAL STYLE ===
def plot_response_curves(
    media_vehicle,
    execution_weeks,
    current_execution,
    current_spend,
    coef,
    half_life,
    steepness,
    saturation
):
    ad_real = geometric_adstock(execution_weeks, half_life)
    max_ad = np.max(ad_real)

    current_inc = calculate_incremental(
        execution_weeks, base_vol_weeks, weekly_price,
        coef, half_life, steepness, saturation, max_ad
    )
    current_roas = current_inc / current_spend if current_spend > 0 else 0

    executions_curve = np.linspace(0, 2 * current_execution, 200)

    inc_vals, spend_vals, roas_vals, mroas_vals = [], [], [], []
    pv, ps = 0, 0
    for ex in executions_curve:
        r = ex / current_execution if current_execution > 0 else 0
        inc = calculate_incremental(
            execution_weeks * r,
            base_vol_weeks,
            weekly_price,
            coef,
            half_life,
            steepness,
            saturation,
            max_ad
        )
        sp = current_spend * r

        inc_vals.append(inc)
        spend_vals.append(sp)
        roas_vals.append(inc / sp if sp > 0 else 0)
        mroas_vals.append((inc - pv) / (sp - ps) if sp > ps else 0)
        pv, ps = inc, sp

    # === Plot 1 (IDENTICAL STYLE) ===
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=executions_curve,
        y=inc_vals,
        mode='lines',
        name='Incremental Sales Value'
    ))
    fig1.add_trace(go.Scatter(
        x=[current_execution], y=[current_inc],
        mode='markers+text',
        text=["Current"],
        textposition="top center",
        marker=dict(color='red', size=10)
    ))
    fig1.update_layout(
        title=f"{media_vehicle}: Execution vs Incremental Sales Value",
        xaxis=dict(
            title="Execution",
            range=[0.1 * current_execution, 1.8 * current_execution]
        ),
        yaxis=dict(title="Incremental Sales (Value)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=80, b=100)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # === Plot 2 (IDENTICAL STYLE) ===
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=spend_vals, y=inc_vals,
        mode='lines',
        name='Incremental Sales Value'
    ))
    fig2.add_trace(go.Scatter(
        x=spend_vals, y=roas_vals,
        mode='lines',
        yaxis='y2',
        line=dict(color='green'),
        name='ROAS'
    ))
    fig2.add_trace(go.Scatter(
        x=spend_vals, y=mroas_vals,
        mode='lines',
        yaxis='y2',
        line=dict(color='orange', dash='dot'),
        name='Marginal ROAS'
    ))
    fig2.add_trace(go.Scatter(
        x=[current_spend], y=[current_inc],
        mode='markers+text',
        text=["Current Sales"],
        textposition="top center",
        marker=dict(color='blue', size=10)
    ))
    fig2.add_trace(go.Scatter(
        x=[current_spend], y=[current_roas],
        mode='markers+text',
        text=["Current ROAS"],
        textposition="bottom center",
        marker=dict(color='green', size=10),
        yaxis='y2'
    ))
    fig2.update_layout(
        title=f"{media_vehicle}: Spend vs Incremental Sales with ROAS and Marginal ROAS",
        xaxis=dict(
            title="Spend",
            range=[0.1 * current_spend, 1.8 * current_spend]
        ),
        yaxis=dict(title="Incremental Sales (Value)"),
        yaxis2=dict(
            title="ROAS / Marginal ROAS",
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=80, b=100)
    )
    st.plotly_chart(fig2, use_container_width=True)

# === EXECUTION ===
if mode == "Current Response Curves":
    for _, row in params_df.iterrows():
        if row["Variable Name"] not in selected_channels:
            continue
        if any(pd.isna(row[c]) for c in ["Half-life", "Steepness", "Saturation", "Coefficient"]):
            continue

        exec_w = mm_f[row["Variable Name"]].fillna(0).values
        plot_response_curves(
            row["Variable Name"],
            exec_w,
            exec_w.sum(),
            ms_f[row["Variable Name"]].sum(),
            row["Coefficient"],
            row["Half-life"],
            row["Steepness"],
            row["Saturation"]
        )
else:
    row = params_df[params_df["Variable Name"] == sim_channel].iloc[0]
    exec_w = mm_f[sim_channel].fillna(0).values
    plot_response_curves(
        sim_channel,
        exec_w,
        exec_w.sum(),
        ms_f[sim_channel].sum(),
        row["Coefficient"],
        sim_hl,
        sim_stp,
        sim_sat
    )
