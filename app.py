import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Stockout Predictor", layout="wide")

st.title("📊 Stockout Loss Predictor (Fast AI Model)")

# Upload dataset
uploaded_file = st.file_uploader("Upload Retail Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # 🔥 AUTO CREATE TARGET
    if 'Stockout Loss' not in df.columns:
        st.warning("No 'Stockout Loss' found → Creating automatically")

        demand_col = None
        stock_col = None

        for col in df.columns:
            if "demand" in col.lower():
                demand_col = col
            if "stock" in col.lower() or "inventory" in col.lower():
                stock_col = col

        if demand_col and stock_col:
            df['Stockout Loss'] = np.maximum(df[demand_col] - df[stock_col], 0)
            st.success(f"Created using: {demand_col} - {stock_col}")
        else:
            st.error("❌ Need Demand & Stock columns")
            st.stop()

    target_col = "Stockout Loss"

    # 🚀 TRAIN MODEL
    if st.button("🚀 Train Model"):

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 🔥 FAST ENCODING
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        cat_cols = X.select_dtypes(include=['object']).columns

        if len(cat_cols) > 0:
            X[cat_cols] = encoder.fit_transform(X[cat_cols])

        # SPLIT
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # SCALE
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ⚡ FAST MODEL
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            n_jobs=-1,
            random_state=42
        )

        model.fit(X_train, y_train)

        # PREDICT
        y_pred = model.predict(X_test)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # SHOW RESULTS
        st.subheader("📈 Model Performance")
        st.write(f"Train Accuracy (R2): {train_acc:.3f}")
        st.write(f"Test Accuracy (R2): {test_acc:.3f}")
        st.write(f"RMSE: {rmse:.2f}")

        # 💾 SAVE MODEL PACKAGE
        model_data = {
            "model": model,
            "scaler": scaler,
            "encoder": encoder,
            "columns": X.columns.tolist()
        }

        with open("model_package.pkl", "wb") as f:
            pickle.dump(model_data, f)

        st.success("✅ Model trained & saved!")

        # 📥 DOWNLOAD BUTTON
        with open("model_package.pkl", "rb") as f:
            st.download_button(
                label="📥 Download Trained Model",
                data=f,
                file_name="stockout_model.pkl",
                mime="application/octet-stream"
            )

    # 🔮 PREDICTION
    if st.button("🔮 Predict using Saved Model"):
        try:
            with open("model_package.pkl", "rb") as f:
                data = pickle.load(f)

            model = data["model"]
            scaler = data["scaler"]
            encoder = data["encoder"]
            saved_cols = data["columns"]

            X = df.drop(columns=[target_col])

            # ENCODE
            cat_cols = X.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                X[cat_cols] = encoder.transform(X[cat_cols])

            # ALIGN COLUMNS
            for col in saved_cols:
                if col not in X:
                    X[col] = 0

            X = X[saved_cols]

            # SCALE
            X_scaled = scaler.transform(X)

            # PREDICT
            predictions = model.predict(X_scaled)

            df['Predicted Loss'] = predictions

            st.subheader("🔮 Predictions")
            st.write(df.head(10))

        except Exception as e:
            st.error(f"❌ Error: {e}")
