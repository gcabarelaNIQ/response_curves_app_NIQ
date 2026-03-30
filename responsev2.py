import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime

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
        value=False,
        help="Manually control adstock, steepness and saturation"
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

