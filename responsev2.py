import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="Response Curve Generator", layout="wide")
st.title("📈 Response Curve Generator")

# =====================================================
# MODE TOGGLE (PRO UX)
# =====================================================
_, mode_col = st.columns([3, 2])
with mode_col:
    simulate_mode = st.toggle(
        "Simulate Curve with Custom Parameters",
        value=False
    )

mode = (
    "Simulate Curve with Custom Parameters"
    if simulate_mode
    else "Current Response Curves"
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

# =====================================================
# FILE UPLOAD + GUARD
# =====================================================
uploaded_file = st.file_uploader("Upload MMM Results Excel file", type=["xlsx"])

if not uploaded_file:
    st.info("⬆️ Upload a MMM results file to get started.")
    st.stop()

# =====================================================
# LOAD DATA
# =====================================================
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
# PARAMETER EXTRACTION
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
    model_result
    .merge(model_coeffs, on="Raw Name", how="left")
    .query("`Variable Name` in @media_channels")
)

# =====================================================
# MATH FUNCTIONS (UNCHANGED)
# =====================================================
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
    total = 0
    for i in range(len(exec_w)):
        total += (np.exp(coef * sat_ad[i]) - 1) * base_vol[i] * price[i]
    return total

# =====================================================
# OPTIONS (SHARED UI ONLY)
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

start_date = st.date_input("Start Date", min_date)
end_date = st.date_input("End Date", max_date)

# =====================================================
# DATE FILTERING
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
# SIMULATION CONTROLS (ONLY HERE)
# =====================================================
if mode == "Simulate Curve with Custom Parameters":
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
# CURRENT RESPONSE CURVES (PERMISSIVE, ORIGINAL)
# =====================================================
if mode == "Current Response Curves":

    for _, row in params_df.iterrows():
        if row["Variable Name"] not in selected_channels:
            continue
        if any(pd.isna(row[c]) for c in ["Half-life", "Steepness", "Saturation", "Coefficient"]):
            continue

        media_vehicle = row["Variable Name"]
        hl = row["Half-life"]
        stp = row["Steepness"]
        sat = row["Saturation"]
        coef = row["Coefficient"]

        exec_w = media_metrics_filtered[media_vehicle].fillna(0).values
        curr_exec = exec_w.sum()
        curr_spend = media_spends_filtered[media_vehicle].sum()

        ad_real = geometric_adstock(exec_w, hl)
        max_ad = ad_real.max()

        curr_inc = calculate_incremental(
            exec_w, base_vol_weeks, weekly_price,
            coef, hl, stp, sat, max_ad
        )
        curr_roas = curr_inc / curr_spend if curr_spend > 0 else 0

        x_exec = np.linspace(0, 2 * curr_exec, 200)

        inc_vals, spend_vals, roas_vals, mroas_vals = [], [], [], []
        pv, ps = 0, 0
        for ex in x_exec:
            r = ex / curr_exec if curr_exec > 0 else 0
            inc = calculate_incremental(exec_w * r, base_vol_weeks,
                                        weekly_price, coef, hl, stp, sat, max_ad)
            spend = curr_spend * r

            inc_vals.append(inc)
            spend_vals.append(spend)
            roas_vals.append(inc / spend if spend > 0 else 0)
            mroas_vals.append((inc - pv) / (spend - ps) if spend > ps else 0)
            pv, ps = inc, spend

        # Execution curve
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x_exec, y=inc_vals, mode="lines"))
        fig1.add_trace(go.Scatter(
            x=[curr_exec], y=[curr_inc],
            mode="markers+text", text=["Current"],
            textposition="top center",
            marker=dict(color="red", size=10)
        ))
        fig1.update_layout(
            title=f"{media_vehicle}: Execution vs Incremental Sales Value",
            xaxis_title="Execution",
            yaxis_title="Incremental Sales (Value)"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Spend curve
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=spend_vals, y=inc_vals, mode="lines", name="Incremental Sales"))
        fig2.add_trace(go.Scatter(x=spend_vals, y=roas_vals, yaxis="y2",
                                  mode="lines", name="ROAS"))
        fig2.add_trace(go.Scatter(x=spend_vals, y=mroas_vals, yaxis="y2",
                                  mode="lines", name="Marginal ROAS"))
        fig2.add_trace(go.Scatter(
            x=[curr_spend], y=[curr_inc],
            mode="markers", name="Current Sales"
        ))
        fig2.add_trace(go.Scatter(
            x=[curr_spend], y=[curr_roas],
            yaxis="y2", mode="markers", name="Current ROAS"
        ))
        fig2.update_layout(
            title=f"{media_vehicle}: Spend vs Incremental Sales with ROAS"
        )
        st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# SIMULATION MODE (STRICT, ISOLATED)
# =====================================================
else:
    row = params_df[params_df["Variable Name"] == sim_channel].iloc[0]
    coef = row["Coefficient"]

    if pd.isna(coef):
        st.error("Selected channel has no coefficient in the model.")
        st.stop()

    exec_w = media_metrics_filtered[sim_channel].fillna(0).values
    curr_exec = exec_w.sum()
    curr_spend = media_spends_filtered[sim_channel].sum()

    ad_real = geometric_adstock(exec_w, sim_hl)
    max_ad = ad_real.max()

    curr_inc = calculate_incremental(
        exec_w, base_vol_weeks, weekly_price,
        coef, sim_hl, sim_stp, sim_sat, max_ad
    )
    curr_roas = curr_inc / curr_spend if curr_spend > 0 else 0

    x_exec = np.linspace(0, 2 * curr_exec, 200)
    inc_vals, spend_vals, roas_vals, mroas_vals = [], [], [], []
    pv, ps = 0, 0

    for ex in x_exec:
        r = ex / curr_exec if curr_exec > 0 else 0
        inc = calculate_incremental(
            exec_w * r, base_vol_weeks,
            weekly_price, coef, sim_hl, sim_stp, sim_sat, max_ad
        )
        spend = curr_spend * r

        inc_vals.append(inc)
        spend_vals.append(spend)
        roas_vals.append(inc / spend if spend > 0 else 0)
        mroas_vals.append((inc - pv) / (spend - ps) if spend > ps else 0)
        pv, ps = inc, spend

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x_exec, y=inc_vals, mode="lines"))
    fig1.add_trace(go.Scatter(
        x=[curr_exec], y=[curr_inc],
        mode="markers", marker=dict(color="red", size=10)
    ))
    fig1.update_layout(
        title=f"SIMULATION – {sim_channel}: Execution vs Incremental Sales Value"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=spend_vals, y=inc_vals, mode="lines", name="Incremental Sales"))
    fig2.add_trace(go.Scatter(x=spend_vals, y=roas_vals, yaxis="y2",
                              mode="lines", name="ROAS"))
    fig2.add_trace(go.Scatter(x=spend_vals, y=mroas_vals, yaxis="y2",
                              mode="lines", name="Marginal ROAS"))
    fig2.add_trace(go.Scatter(
        x=[curr_spend], y=[curr_inc],
        mode="markers", name="Current Sales"
    ))
    fig2.add_trace(go.Scatter(
        x=[curr_spend], y=[curr_roas],
        yaxis="y2", mode="markers", name="Current ROAS"
    ))
    fig2.update_layout(
        title=f"SIMULATION – {sim_channel}: Spend vs Incremental Sales with ROAS"
    )
    st.plotly_chart(fig2, use_container_width=True)
