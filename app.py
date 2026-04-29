import streamlit as st
import pickle
import numpy as np
from pathlib import Path
import pandas as pd

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
model = pickle.load(open(BASE_DIR / "model.pkl", "rb"))
scaler = pickle.load(open(BASE_DIR / "scaler.pkl", "rb"))

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Stockout Predictor", layout="centered")

st.title("📦 Stockout Loss Prediction System")

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("📥 Enter Product Details")

inventory = st.number_input("Inventory Level", min_value=0, value=100)
units_sold = st.number_input("Units Sold", min_value=0, value=80)
forecast = st.number_input("Demand Forecast", min_value=0, value=120)
price = st.number_input("Price (₹)", min_value=0.0, value=50.0)
discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0)
competitor_price = st.number_input("Competitor Price (₹)", min_value=0.0, value=55.0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Loss"):

    input_data = np.array([[inventory, units_sold, forecast, price, discount, competitor_price]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # -------------------------------
    # RISK LOGIC
    # -------------------------------
    if forecast > inventory:
        st.error(f"⚠️ HIGH RISK of stockout!\n\nEstimated Loss: ₹{prediction:.2f}")
    elif prediction > 500:
        st.warning(f"⚠️ Moderate Risk\n\nEstimated Loss: ₹{prediction:.2f}")
    else:
        st.success(f"✅ Safe Stock Level\n\nEstimated Loss: ₹{prediction:.2f}")

    # -------------------------------
    # SUMMARY
    # -------------------------------
    st.subheader("📋 Summary")

    summary = {
        "Inventory": inventory,
        "Forecast": forecast,
        "Predicted Loss": round(prediction, 2)
    }

    st.table(pd.DataFrame(summary.items(), columns=["Metric", "Value"]))

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("💡 Helps prevent stockouts and reduce business loss")
