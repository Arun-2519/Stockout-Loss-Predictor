import streamlit as st
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# LOAD MODEL & SCALER
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent

model = pickle.load(open(BASE_DIR / "model.pkl", "rb"))
scaler = pickle.load(open(BASE_DIR / "scaler.pkl", "rb"))

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Stockout Loss Predictor", layout="wide")

st.title("📦 AI Stockout Loss Prediction System")
st.markdown("Predict potential stock loss and get smart alerts for better inventory decisions.")

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("📥 Enter Product Details")

inventory = st.sidebar.number_input("Inventory Level", min_value=0, value=100)
units_sold = st.sidebar.number_input("Units Sold", min_value=0, value=80)
forecast = st.sidebar.number_input("Demand Forecast", min_value=0, value=120)
price = st.sidebar.number_input("Price (₹)", min_value=0.0, value=50.0)
discount = st.sidebar.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0)
competitor_price = st.sidebar.number_input("Competitor Price (₹)", min_value=0.0, value=55.0)

# -------------------------------
# PREDICTION
# -------------------------------
input_data = np.array([[inventory, units_sold, forecast, price, discount, competitor_price]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# -------------------------------
# RISK CLASSIFICATION
# -------------------------------
def classify_risk(inv, demand, loss):
    if demand > inv:
        return "High Risk", "🔴"
    elif loss > 500:
        return "Medium Risk", "🟡"
    else:
        return "Safe", "🟢"

risk, icon = classify_risk(inventory, forecast, prediction)

# -------------------------------
# ALERT SYSTEM
# -------------------------------
st.subheader("🚨 Alert System")

if risk == "High Risk":
    st.error(f"⚠️ HIGH RISK of stockout! Estimated Loss: ₹{prediction:.2f}")
elif risk == "Medium Risk":
    st.warning(f"⚠️ Moderate Risk. Estimated Loss: ₹{prediction:.2f}")
else:
    st.success(f"✅ Safe Stock Level. Estimated Loss: ₹{prediction:.2f}")

# -------------------------------
# DASHBOARD
# -------------------------------
col1, col2 = st.columns(2)

# 📊 Loss Trend
with col1:
    st.subheader("📊 Loss Trend")

    trend_data = pd.DataFrame({
        "Day": range(1, 8),
        "Loss": [200, 250, 300, 400, 350, 450, prediction]
    })

    fig, ax = plt.subplots()
    ax.plot(trend_data["Day"], trend_data["Loss"], marker='o')
    ax.set_xlabel("Days")
    ax.set_ylabel("Loss")
    ax.set_title("Weekly Loss Trend")

    st.pyplot(fig)

# 📉 Stock vs Demand
with col2:
    st.subheader("📉 Inventory vs Demand Forecast")

    fig2, ax2 = plt.subplots()
    ax2.bar(["Inventory", "Forecast"], [inventory, forecast])
    ax2.set_title("Stock vs Demand")

    st.pyplot(fig2)

# -------------------------------
# SUMMARY
# -------------------------------
st.subheader("📋 Prediction Summary")

summary = {
    "Inventory Level": inventory,
    "Demand Forecast": forecast,
    "Predicted Loss (₹)": round(prediction, 2),
    "Risk Level": risk
}

st.table(pd.DataFrame(summary.items(), columns=["Metric", "Value"]))

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("🔍 Powered by Machine Learning | Built with Streamlit")
