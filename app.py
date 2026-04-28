import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.title("📊 Stockout Loss Predictor (AI Model)")

uploaded_file = st.file_uploader("Upload Retail Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    # 🔥 STEP 1: AUTO CREATE TARGET IF NOT EXISTS
    if 'Stockout Loss' not in df.columns:

        st.warning("No 'Stockout Loss' column found → Creating automatically")

        # Try to detect useful columns
        cols = df.columns.tolist()

        demand_col = None
        stock_col = None

        for col in cols:
            if "demand" in col.lower():
                demand_col = col
            if "stock" in col.lower() or "inventory" in col.lower():
                stock_col = col

        if demand_col and stock_col:
            df['Stockout Loss'] = np.maximum(df[demand_col] - df[stock_col], 0)
            st.success(f"Created Stockout Loss using {demand_col} & {stock_col}")
        else:
            st.error("❌ Cannot auto-create target. Please ensure dataset has Demand & Stock columns")
            st.stop()

    target_col = "Stockout Loss"

    # 🔥 Train Model
    if st.button("Train Model"):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle non-numeric columns
        X = pd.get_dummies(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.subheader("📈 Model Performance")
        st.write(f"Train Accuracy (R2): {train_acc:.3f}")
        st.write(f"Test Accuracy (R2): {test_acc:.3f}")
        st.write(f"RMSE: {rmse:.2f}")

        # Save model + columns
        joblib.dump(model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(X.columns.tolist(), "columns.pkl")

        st.success("Model trained & saved successfully ✅")

    # 🔮 Prediction
    if st.button("Predict using saved model"):
        try:
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")
            saved_cols = joblib.load("columns.pkl")

            X = df.drop(columns=[target_col])
            X = pd.get_dummies(X)

            # Align columns
            for col in saved_cols:
                if col not in X:
                    X[col] = 0

            X = X[saved_cols]

            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)

            df['Predicted Loss'] = predictions

            st.subheader("🔮 Predictions")
            st.write(df.head(10))

        except Exception as e:
            st.error(f"Error: {e}")
