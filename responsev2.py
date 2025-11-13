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
st.set_page_config(page_title="Response Curve Generator 2.0", layout="wide")
st.title("📈 Response Curve Generator 2.0")

# === Instruction Message ===
st.markdown("""
**⚠️ Important:**  
Make sure to upload the **COE file exactly as you received it**, and confirm that the following sheets exist with these exact names:  
- `Decomps Vol`  
- `Decomps Value`  
- `Media KeyMetrics`  
- `Media Spends`  
- `Model Result`  
- `Model Coefficients`  
""")

# Status placeholder
status = st.empty()

uploaded_file = st.file_uploader("Upload your MMM Results Excel file", type=["xlsx"])

if uploaded_file:
    status.success("File uploaded successfully! Processing...")

    # === Load Excel ===
    xls = pd.ExcelFile(uploaded_file, engine='openpyxl')

    # Validate required sheets
    required_sheets = ["Decomps Vol", "Decomps Value", "Media KeyMetrics", "Media Spends", "Model Result", "Model Coefficients"]
    if not all(sheet in xls.sheet_names for sheet in required_sheets):
        status.error("❌ Missing required sheets. Please upload the correct COE file.")
        st.stop()

    # Load sheets
    decomps_vol = xls.parse("Decomps Vol", usecols=["EndPeriod", "final_0_vol"])
    decomps_val = xls.parse("Decomps Value", usecols=["EndPeriod", "final_0_val"])
    media_metrics = xls.parse("Media KeyMetrics")
    media_spends = xls.parse("Media Spends")
    model_result = xls.parse("Model Result", header=None, skiprows=1).iloc[:, [2, 3]]
    model_result.columns = ["Variable Name", "Raw Name"]
    model_coeffs = xls.parse("Model Coefficients", header=None, skiprows=1).iloc[:, [1, 2]]
    model_coeffs.columns = ["Raw Name", "Coefficient"]

    # Merge decompositions
    base_data = pd.merge(decomps_vol, decomps_val, on="EndPeriod")

    # Media channels
    media_channels = media_metrics.columns.drop("EndPeriod").tolist()
    valid_channels = [ch for ch in media_channels if ch not in ["Unnamed: 0", "custom_period"]]

    # Extract parameters
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

    model_result[["Half-life", "Steepness", "Saturation"]] = model_result["Raw Name"].apply(
        lambda x: pd.Series(extract_params(x))
    )

    # Merge with coefficients
    params_df = model_result.merge(model_coeffs, on="Raw Name", how="left")
    params_df = params_df[params_df["Variable Name"].isin(valid_channels)]

    # Validate params_df
    if params_df.empty:
        status.error("❌ No media parameters found. Check your file.")
        st.stop()

    # Update status
    status.info("✅ File processed successfully! Ready to select options.")
    st.write("✅ Found channels:", params_df["Variable Name"].tolist())

    # === UI Options ===
    st.subheader("Options")
    selected_channels = st.multiselect("Select channels to plot", options=params_df["Variable Name"].tolist(), default=params_df["Variable Name"].tolist())
    start_date = st.date_input("Start Date", datetime(2023, 1, 8))
    end_date = st.date_input("End Date", datetime(2024, 12, 29))

    # Filter by date
    for df in [media_metrics, media_spends, base_data]:
        df["EndPeriod"] = pd.to_datetime(df["EndPeriod"])

    media_metrics_filtered = media_metrics[(media_metrics["EndPeriod"] >= pd.to_datetime(start_date)) & (media_metrics["EndPeriod"] <= pd.to_datetime(end_date))]
    media_spends_filtered = media_spends[(media_spends["EndPeriod"] >= pd.to_datetime(start_date)) & (media_spends["EndPeriod"] <= pd.to_datetime(end_date))]
    base_data_filtered = base_data[(base_data["EndPeriod"] >= pd.to_datetime(start_date)) & (base_data["EndPeriod"] <= pd.to_datetime(end_date))]

    base_vol_weeks = base_data_filtered["final_0_vol"].values
    base_val_weeks = base_data_filtered["final_0_val"].values
    weekly_price = base_val_weeks / base_vol_weeks

    # Functions
    def geometric_adstock(x, half_life):
        decay = 0.5 ** (1 / half_life)
        adstocked = np.zeros_like(x)
        adstocked[0] = x[0]
        for t in range(1, len(x)):
            adstocked[t] = x[t] + decay * adstocked[t - 1]
        return adstocked

    def sigmoid_saturation(adstocked, steepness, saturation, max_adstock):
        return max_adstock / (1 + np.exp(-10 * steepness / max_adstock * (adstocked - saturation * max_adstock)))

    def calculate_incremental(executions, base_vol_weeks, weekly_price, coef, half_life, steepness, saturation, max_adstock):
        adstocked = geometric_adstock(executions, half_life)
        saturated = sigmoid_saturation(adstocked, steepness, saturation, max_adstock)
        incremental_val_total = 0
        for i in range(len(executions)):
            inc_vol_week = (np.exp(coef * saturated[i]) - 1) * base_vol_weeks[i]
            inc_val_week = inc_vol_week * weekly_price[i]
            incremental_val_total += inc_val_week
        return incremental_val_total

    # Generate plots and HTML
    html_buffer = io.StringIO()
    st.subheader("Preview")
    for idx, row in params_df.iterrows():
        if row["Variable Name"] not in selected_channels:
            continue
        if any(pd.isna(row[col]) for col in ["Half-life", "Steepness", "Saturation", "Coefficient"]):
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

        current_inc_val = calculate_incremental(execution_weeks, base_vol_weeks, weekly_price,
                                                coef, half_life, steepness, saturation, max_adstock)
        current_roas = current_inc_val / current_spend if current_spend > 0 else 0

        curve_points = 200
        executions_curve = np.linspace(0.2 * current_execution, 2 * current_execution, curve_points)
        incremental_values, spend_values, roas_values, marginal_roas_values = [], [], [], []

        prev_val, prev_spend = 0, 0
        for ex in executions_curve:
            ratio = ex / current_execution if current_execution > 0 else 0
            scaled_execution = execution_weeks * ratio
            inc_val = calculate_incremental(scaled_execution, base_vol_weeks, weekly_price,
                                            coef, half_life, steepness, saturation, max_adstock)
            incremental_values.append(inc_val)
            spend = current_spend * ratio
            spend_values.append(spend)
            roas = inc_val / spend if spend > 0 else 0
            roas_values.append(roas)
            marginal_roas = (inc_val - prev_val) / (spend - prev_spend) if spend > prev_spend else 0
            marginal_roas_values.append(marginal_roas)
            prev_val, prev_spend = inc_val, spend

        # Plot 1
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=executions_curve, y=incremental_values, mode='lines', name='Incremental Sales Value'))
        fig1.add_trace(go.Scatter(x=[current_execution], y=[current_inc_val],
                                  mode='markers+text', text=["Current"], textposition="top center",
                                  marker=dict(color='red', size=10), name='Current Point'))
        fig1.update_layout(
            title=f"{media_vehicle}: Execution vs Incremental Sales Value",
            xaxis=dict(
                title="Execution",
                range=[0.2 * current_execution, 1.8 * current_execution]  # ✅ Align with executions_curve
            ),
            yaxis=dict(title="Incremental Sales (Value)"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(t=80, b=100))
        st.plotly_chart(fig1, use_container_width=True)
        html_buffer.write(pio.to_html(fig1, full_html=False, include_plotlyjs='cdn'))


        # Plot 2
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=spend_values, y=incremental_values, mode='lines', name='Incremental Sales Value'))
        fig2.add_trace(go.Scatter(x=spend_values, y=roas_values, mode='lines', name='ROAS', yaxis='y2', line=dict(color='green')))
        fig2.add_trace(go.Scatter(x=spend_values, y=marginal_roas_values, mode='lines', name='Marginal ROAS', yaxis='y2', line=dict(color='orange', dash='dot')))
        fig2.add_trace(go.Scatter(x=[current_spend], y=[current_inc_val],
                                  mode='markers+text', text=["Current Sales"], textposition="top center",
                                  marker=dict(color='blue', size=10), name='Current Sales'))
        fig2.add_trace(go.Scatter(x=[current_spend], y=[current_roas],
                                  mode='markers+text', text=["Current ROAS"], textposition="bottom center",
                                  marker=dict(color='green', size=10), name='Current ROAS', yaxis='y2'))
        
        # Layout
        fig2.update_layout(
            title=f"{media_vehicle}: Spend vs Incremental Sales with ROAS and Marginal ROAS",
            xaxis=dict(title="Spend", range=[0.2 * current_spend, 1.8 * current_spend]),
            yaxis=dict(title="Incremental Sales (Value)"),
            yaxis2=dict(title="ROAS / Marginal ROAS", overlaying='y', side='right'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(t=80, b=100)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        html_buffer.write(pio.to_html(fig2, full_html=False, include_plotlyjs=False))


    # Create Excel file on disk using openpyxl
    excel_file = "response_curves.xlsx"
    
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        for idx, row in params_df.iterrows():
            if row["Variable Name"] not in selected_channels:
                continue
            if any(pd.isna(row[col]) for col in ["Half-life", "Steepness", "Saturation", "Coefficient"]):
                continue
    
            media_vehicle = row["Variable Name"]  # Sheet name
            
            # ✅ Find closest index to current_spend
            current_index = min(range(len(spend_values)), key=lambda i: abs(spend_values[i] - current_spend))

    
            # Prepare data for this channel
            data = {
                "Spend": spend_values,
                "Execution": executions_curve,
                "Revenue": incremental_values,
                "ROAS": roas_values,
                "Marginal ROAS": marginal_roas_values,
                "Current": [current_inc_val if i == current_index else "" for i in range(len(spend_values))]
            }
    
            df = pd.DataFrame(data)
    
            # Write to Excel sheet
            df.to_excel(writer, sheet_name=media_vehicle, index=False)
    
    # ✅ Streamlit download button
    st.download_button(
        label="📥 Download RCs Data to Excel 📥",
        data=open(excel_file, "rb").read(),
        file_name="response_curves.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
