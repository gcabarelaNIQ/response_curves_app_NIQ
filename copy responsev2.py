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

# === MODE TOGGLE ===
st.markdown("## 🔁 Switch Analysis Mode")
st.caption(
    "Change between model‑fitted response curves and manual simulations "
    "of adstocked–saturated executions"
)

simulate_mode = st.toggle(
    "Simulate Curve with Custom Half‑Life, Steepness, and Saturation Parameters",
    value=False
)

MODE_SIM = "Simulate"
MODE_CURRENT = "Current"

mode = MODE_SIM if simulate_mode else MODE_CURRENT

# === Instructions ===
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

# === Dates & base ===
for df in (decomps_vol, media_metrics, media_spends):
    df["EndPeriod"] = pd.to_datetime(df["EndPeriod"])

min_date = decomps_vol["EndPeriod"].min().date()
max_date = decomps_vol["EndPeriod"].max().date()

base_data = decomps_vol.merge(decomps_val, on="EndPeriod")

media_channels = [
    c for c in media_metrics.columns
    if c not in ["EndPeriod", "custom_period", "Unnamed: 0"]
]

# === Extract parameters (UNCHANGED) ===
def extract_params(raw_name):
    if pd.isna(raw_name):
        return (None, None, None)
    raw_name = str(raw_name)
    hlfl_match = re.search(r"hlfl(\d+(\.\d+)?)", raw_name)
    step_match = re.search(r"step(\d+(\.\d+)?)", raw_name)
    sat_match = re.search(r"sat(\d+(\.\d+)?)", raw_name)
    return (
        float(hlfl_match.group(1)) if hlfl_match else None,
        float(step_match.group(1)) if step_match else None,
        float(sat_match.group(1)) if sat_match else None
    )

model_result[["Half-life", "Steepness", "Saturation"]] = (
    model_result["Raw Name"].apply(lambda x: pd.Series(extract_params(x)))
)

params_df = (
    model_result.merge(model_coeffs, on="Raw Name", how="left")
    .query("`Variable Name` in @media_channels")
)

# === Maths (UNCHANGED) ===
def geometric_adstock(x, half_life):
    decay = 0.5 ** (1 / half_life)
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]
    for t in range(1, len(x)):
        adstocked[t] = x[t] + decay * adstocked[t - 1]
    return adstocked

def sigmoid_saturation(adstocked, steepness, saturation, max_adstock):
    return (
        max_adstock / (1 + np.exp(-10 * steepness / max_adstock *
                                  (adstocked - saturation * max_adstock)))
    ) - (
        max_adstock / (1 + np.exp(-10 * steepness / max_adstock *
                                  (0 - saturation * max_adstock)))
    )

def calculate_incremental(executions, base_vol_weeks, weekly_price,
                          coef, half_life, steepness, saturation, max_adstock):
    adstocked = geometric_adstock(executions, half_life)
    saturated = sigmoid_saturation(adstocked, steepness, saturation, max_adstock)
    incremental_val_total = 0
    for i in range(len(executions)):
        inc_vol_week = (np.exp(coef * saturated[i]) - 1) * base_vol_weeks[i]
        inc_val_week = inc_vol_week * weekly_price[i]
        incremental_val_total += inc_val_week
    return incremental_val_total

# === Options ===
st.subheader("Options")

if mode == MODE_CURRENT:
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

media_metrics_filtered = media_metrics[
    (media_metrics["EndPeriod"] >= pd.to_datetime(start_date)) &
    (media_metrics["EndPeriod"] <= pd.to_datetime(end_date))
]
media_spends_filtered = media_spends[
    (media_spends["EndPeriod"] >= pd.to_datetime(start_date)) &
    (media_spends["EndPeriod"] <= pd.to_datetime(end_date))
]
base_data_filtered = base_data[
    (base_data["EndPeriod"] >= pd.to_datetime(start_date)) &
    (base_data["EndPeriod"] <= pd.to_datetime(end_date))
]

base_vol_weeks = base_data_filtered["final_0_vol"].values
base_val_weeks = base_data_filtered["final_0_val"].values
weekly_price = base_val_weeks / base_vol_weeks

# === Simulation sliders ===
if mode == MODE_SIM:
    st.subheader("Simulation parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_hl = st.slider("Half-life", 0.1, 5.0, 1.0, step=0.01)
    with c2:
        sim_stp = st.slider("Steepness", 0.1, 3.0, 1.0, step=0.01)
    with c3:
        sim_sat = st.slider("Saturation", 0.1, 0.9, 0.5, step=0.01)

st.subheader("Preview")

# ======================================================================
# ✅ ORIGINAL PLOTTING CODE — VERBATIM, STYLE PRESERVED
# ======================================================================
def plot_response_curves(
    media_vehicle,
    execution_weeks,
    current_execution,
    current_spend,
    coef,
    half_life,
    steepness,
    saturation,
    return_figs=False
):
    adstocked_real = geometric_adstock(execution_weeks, half_life)
    max_adstock = np.max(adstocked_real)

    current_inc_val = calculate_incremental(
        execution_weeks, base_vol_weeks, weekly_price,
        coef, half_life, steepness, saturation, max_adstock
    )
    current_roas = current_inc_val / current_spend if current_spend > 0 else 0

    curve_points = 200
    executions_curve = np.linspace(0, 2 * current_execution, curve_points)
    incremental_values, spend_values, roas_values, marginal_roas_values = [], [], [], []

    prev_val, prev_spend = 0, 0
    for ex in executions_curve:
        ratio = ex / current_execution if current_execution > 0 else 0
        scaled_execution = execution_weeks * ratio
        inc_val = calculate_incremental(
            scaled_execution, base_vol_weeks, weekly_price,
            coef, half_life, steepness, saturation, max_adstock
        )
        incremental_values.append(inc_val)
        spend = current_spend * ratio
        spend_values.append(spend)
        roas = inc_val / spend if spend > 0 else 0
        roas_values.append(roas)
        marginal_roas = ((inc_val - prev_val) / (spend - prev_spend)
                         if spend > prev_spend else 0)
        marginal_roas_values.append(marginal_roas)
        prev_val, prev_spend = inc_val, spend

    # --- Plot 1 (EXECUTION) ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=executions_curve, y=incremental_values,
        mode='lines', name='Incremental Sales Value'
    ))
    fig1.add_trace(go.Scatter(
        x=[current_execution], y=[current_inc_val],
        mode='markers+text', text=["Current"],
        textposition="top center",
        marker=dict(color='red', size=10), name='Current Point'
    ))
    fig1.update_layout(
        title=f"{media_vehicle}: Execution vs Incremental Sales Value",
        xaxis=dict(
            title="Execution",
            range=[0.1 * current_execution, 1.8 * current_execution]
        ),
        yaxis=dict(title="Incremental Sales (Value)"),
        legend=dict(
            orientation="h", yanchor="bottom",
            y=-0.3, xanchor="center", x=0.5
        ),
        margin=dict(t=80, b=100)
    )

    # --- Plot 2 (SPEND) ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=spend_values, y=incremental_values,
        mode='lines', name='Incremental Sales Value'
    ))
    fig2.add_trace(go.Scatter(
        x=spend_values, y=roas_values,
        mode='lines', name='ROAS',
        yaxis='y2', line=dict(color='green')
    ))
    fig2.add_trace(go.Scatter(
        x=spend_values, y=marginal_roas_values,
        mode='lines', name='Marginal ROAS',
        yaxis='y2', line=dict(color='orange', dash='dot')
    ))
    fig2.add_trace(go.Scatter(
        x=[current_spend], y=[current_inc_val],
        mode='markers+text', text=["Current Sales"],
        textposition="top center",
        marker=dict(color='blue', size=10)
    ))
    fig2.add_trace(go.Scatter(
        x=[current_spend], y=[current_roas],
        mode='markers+text', text=["Current ROAS"],
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
            overlaying='y', side='right'
        ),
        legend=dict(
            orientation="h", yanchor="bottom",
            y=-0.3, xanchor="center", x=0.5
        ),
        margin=dict(t=80, b=100)
    )

    if return_figs:
        return fig1, fig2

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# === Execute ===
if mode == MODE_CURRENT:
    for _, row in params_df.iterrows():
        if row["Variable Name"] not in selected_channels:
            continue
        if any(pd.isna(row[col]) for col in
               ["Half-life", "Steepness", "Saturation", "Coefficient"]):
            continue

        exec_w = media_metrics_filtered[row["Variable Name"]].fillna(0).values
        plot_response_curves(
            row["Variable Name"],
            exec_w,
            exec_w.sum(),
            media_spends_filtered[row["Variable Name"]].sum(),
            row["Coefficient"],
            row["Half-life"],
            row["Steepness"],
            row["Saturation"]
        )

else:
    row = params_df[params_df["Variable Name"] == sim_channel].iloc[0]
    exec_w = media_metrics_filtered[sim_channel].fillna(0).values

    fig_exec, fig_spend = plot_response_curves(
        sim_channel,
        exec_w,
        exec_w.sum(),
        media_spends_filtered[sim_channel].sum(),
        row["Coefficient"],
        sim_hl,
        sim_stp,
        sim_sat,
        return_figs=True
    )

    # ✅ Simulation: Spend first, Execution second
    st.plotly_chart(fig_spend, use_container_width=True)
    st.plotly_chart(fig_exec, use_container_width=True)
