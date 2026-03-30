import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import re
from datetime import datetime

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="Response Curve Generator", layout="wide")
st.title("📈 Response Curve Generator")

# =====================================================
# MODE TOGGLE (TOP RIGHT)
# =====================================================
_, mode_col = st.columns([3, 2])
with mode_col:
    mode = st.radio(
        "",
        ["Current Response Curves", "Simulate one response curve"],
        index=0,
        horizontal=True
    )

# =====================================================
# INSTRUCTIONS
# =====================================================
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

uploaded_file = st.file_uploader("Upload MMM Results Excel file", type=["xlsx"])

if not uploaded_file:
    st.info("⬆️ Upload a MMM results file to get started.")
    st.stop()

# =====================================================
# LOAD DATA
# =====================================================
xls = pd.ExcelFile(uploaded_file, engine="openpyxl")

required_sheets = [
    "Decomps Vol",
    "Decomps Value",
    "Media KeyMetrics",
    "Media Spends",
    "Model Result",
    "Predictors Summary"
]
if not all(s in xls.sheet_names for s in required_sheets):
    st.error("❌ Missing required sheets.")
    st.stop()

decomps_vol = xls.parse("Decomps Vol", usecols=["EndPeriod", "final_0_vol"])
decomps_val = xls.parse("Decomps Value", usecols=["EndPeriod", "final_0_val"])
media_metrics = xls.parse("Media KeyMetrics")
media_spends = xls.parse("Media Spends")

decomps_vol["EndPeriod"] = pd.to_datetime(decomps_vol["EndPeriod"])
media_metrics["EndPeriod"] = pd.to_datetime(media_metrics["EndPeriod"])
media_spends["EndPeriod"] = pd.to_datetime(media_spends["EndPeriod"])

min_date = decomps_vol["EndPeriod"].min().date()
max_date = decomps_vol["EndPeriod"].max().date()

base_data = decomps_vol.merge(decomps_val, on="EndPeriod")

model_result = xls.parse("Model Result", header=None, skiprows=1).iloc[:, [2, 3]]
model_result.columns = ["Variable Name", "Raw Name"]

model_coeffs = xls.parse("Predictors Summary", header=None, skiprows=1).iloc[:, [1, 3]]
model_coeffs.columns = ["Raw Name", "Coefficient"]

media_channels = [
    c for c in media_metrics.columns
    if c not in ["EndPeriod", "custom_period", "Unnamed: 0"]
]

# =====================================================
# PARAM EXTRACTION
# =====================================================
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

# =====================================================
# SHARED FUNCTIONS (UNCHANGED)
# =====================================================
def geometric_adstock(x, half_life):
    decay = 0.5 ** (1 / half_life)
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]
    for t in range(1, len(x)):
        adstocked[t] = x[t] + decay * adstocked[t - 1]
    return adstocked

def sigmoid_saturation(adstocked, steepness, saturation, max_adstock):
    return (
        max_adstock /
        (1 + np.exp(-10 * steepness / max_adstock *
                    (adstocked - saturation * max_adstock)))
    ) - (
        max_adstock /
        (1 + np.exp(-10 * steepness / max_adstock *
                    (0 - saturation * max_adstock)))
    )

def calculate_incremental(executions, base_vol_weeks, weekly_price,
                          coef, half_life, steepness, saturation, max_adstock):
    adstocked = geometric_adstock(executions, half_life)
    saturated = sigmoid_saturation(adstocked, steepness, saturation, max_adstock)
    total = 0
    for i in range(len(executions)):
        vol = (np.exp(coef * saturated[i]) - 1) * base_vol_weeks[i]
        total += vol * weekly_price[i]
    return total

# =====================================================
# OPTIONS UI (IDENTICAL)
# =====================================================
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

start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# =====================================================
# DATE FILTERING (UNCHANGED)
# =====================================================
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

# =====================================================
# SIMULATION CONTROLS
# =====================================================
if mode == "Simulate one response curve":
    st.subheader("Simulation parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_hl = st.slider("Half-life", 0.1, 5.0, 1.0, step=0.01)
    with c2:
        sim_stp = st.slider("Steepness", 0.1, 3.0, 1.0, step=0.01)
    with c3:
        sim_sat = st.slider("Saturation", 0.1, 0.9, 0.5, step=0.01)

# =====================================================
# PREVIEW
# =====================================================
st.subheader("Preview")

# =====================================================
# CURRENT RESPONSE CURVES (100% ORIGINAL)
# =====================================================
if mode == "Current Response Curves":

    for _, row in params_df.iterrows():
        if row["Variable Name"] not in selected_channels:
            continue
        if any(pd.isna(row[c]) for c in ["Half-life", "Steepness", "Saturation", "Coefficient"]):
            continue

        media_vehicle = row["Variable Name"]
        half_life = row["Half-life"]
        steepness = row["Steepness"]
        saturation = row["Saturation"]
        coef = round(row["Coefficient"], 15)

        execution_weeks = media_metrics_filtered[media_vehicle].fillna(0).values
        current_execution = np.sum(execution_weeks)
        current_spend = media_spends_filtered[media_vehicle].sum()

        adstocked_real = geometric_adstock(execution_weeks, half_life)
        max_adstock = np.max(adstocked_real)

        current_inc_val = calculate_incremental(
            execution_weeks, base_vol_weeks, weekly_price,
            coef, half_life, steepness, saturation, max_adstock
        )
        current_roas = current_inc_val / current_spend if current_spend > 0 else 0

        executions_curve = np.linspace(0, 2 * current_execution, 200)
        inc_vals, spend_vals, roas_vals, mroas_vals = [], [], [], []

        prev_val, prev_spend = 0, 0
        for ex in executions_curve:
            ratio = ex / current_execution if current_execution > 0 else 0
            inc = calculate_incremental(
                execution_weeks * ratio, base_vol_weeks,
                weekly_price, coef, half_life,
                steepness, saturation, max_adstock
            )
            spend = current_spend * ratio

            inc_vals.append(inc)
            spend_vals.append(spend)
            roas_vals.append(inc / spend if spend > 0 else 0)
            mroas_vals.append((inc - prev_val) / (spend - prev_spend) if spend > prev_spend else 0)

            prev_val, prev_spend = inc, spend

        # === Plot 1 (IDENTICAL)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=executions_curve,
            y=inc_vals,
            mode='lines',
            name='Incremental Sales Value'
        ))
        fig1.add_trace(go.Scatter(
            x=[current_execution],
            y=[current_inc_val],
            mode='markers+text',
            text=["Current"],
            textposition="top center",
            marker=dict(color='red', size=10),
            name='Current Point'
        ))
        fig1.update_layout(
            title=f"{media_vehicle}: Execution vs Incremental Sales Value",
            xaxis=dict(title="Execution",
                       range=[0.1 * current_execution, 1.8 * current_execution]),
            yaxis=dict(title="Incremental Sales (Value)"),
            legend=dict(orientation="h", yanchor="bottom",
                        y=-0.3, xanchor="center", x=0.5),
            margin=dict(t=80, b=100)
        )
        st.plotly_chart(fig1, use_container_width=True)

        # === Plot 2 (IDENTICAL)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=spend_vals, y=inc_vals,
            mode='lines', name='Incremental Sales Value'
        ))
        fig2.add_trace(go.Scatter(
            x=spend_vals, y=roas_vals,
            mode='lines', name='ROAS',
            yaxis='y2', line=dict(color='green')
        ))
        fig2.add_trace(go.Scatter(
            x=spend_vals, y=mroas_vals,
            mode='lines', name='Marginal ROAS',
            yaxis='y2', line=dict(color='orange', dash='dot')
        ))
        fig2.add_trace(go.Scatter(
            x=[current_spend], y=[current_inc_val],
            mode='markers+text', text=["Current Sales"],
            textposition="top center",
            marker=dict(color='blue', size=10),
            name='Current Sales'
        ))
        fig2.add_trace(go.Scatter(
            x=[current_spend], y=[current_roas],
            mode='markers+text', text=["Current ROAS"],
            textposition="bottom center",
            marker=dict(color='green', size=10),
            name='Current ROAS', yaxis='y2'
        ))
        fig2.update_layout(
            title=f"{media_vehicle}: Spend vs Incremental Sales with ROAS and Marginal ROAS",
            xaxis=dict(title="Spend",
                       range=[0.1 * current_spend, 1.8 * current_spend]),
            yaxis=dict(title="Incremental Sales (Value)"),
            yaxis2=dict(title="ROAS / Marginal ROAS",
                        overlaying='y', side='right'),
            legend=dict(orientation="h", yanchor="bottom",
                        y=-0.3, xanchor="center", x=0.5),
            margin=dict(t=80, b=100)
        )
        st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# SIMULATION MODE (ISOLATED)
# =====================================================
else:
    row = params_df[params_df["Variable Name"] == sim_channel].iloc[0]
    coef = row["Coefficient"]

    execution_weeks = media_metrics_filtered[sim_channel].fillna(0).values
    current_execution = np.sum(execution_weeks)
    current_spend = media_spends_filtered[sim_channel].sum()

    adstocked_real = geometric_adstock(execution_weeks, sim_hl)
    max_adstock = np.max(adstocked_real)

    current_inc_val = calculate_incremental(
        execution_weeks, base_vol_weeks, weekly_price,
        coef, sim_hl, sim_stp, sim_sat, max_adstock
    )

    executions_curve = np.linspace(0, 2 * current_execution, 200)
    inc_vals, spend_vals = [], []

    for ex in executions_curve:
        ratio = ex / current_execution if current_execution > 0 else 0
        inc_vals.append(
            calculate_incremental(
                execution_weeks * ratio, base_vol_weeks,
                weekly_price, coef, sim_hl, sim_stp, sim_sat, max_adstock
            )
        )
        spend_vals.append(current_spend * ratio)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spend_vals, y=inc_vals, mode='lines',
                             name='Incremental Sales Value'))
    fig.add_trace(go.Scatter(x=[current_spend], y=[current_inc_val],
                             mode='markers+text', text=["Current"],
                             textposition="top center",
                             marker=dict(color='red', size=10)))
    fig.update_layout(
        title=f"SIMULATION – {sim_channel}",
        xaxis_title="Spend",
        yaxis_title="Incremental Sales (Value)",
        margin=dict(t=80, b=80)
    )
    st.plotly_chart(fig, use_container_width=True)
