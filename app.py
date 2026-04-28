import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.title("📊 Stockout Loss Predictor (AI Model)")

# Upload dataset
uploaded_file = st.file_uploader("Upload Retail Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    # Target column
    target_col = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Model"):
        X = df.drop(columns=[target_col])
        y = df[target_col]

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

        # Predictions
        y_pred = model.predict(X_test)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Show metrics
        st.subheader("📈 Model Performance")
        st.write(f"Train Accuracy (R2): {train_acc:.3f}")
        st.write(f"Test Accuracy (R2): {test_acc:.3f}")
        st.write(f"RMSE: {rmse:.2f}")

        # Save model
        joblib.dump(model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")

        st.success("Model trained & saved successfully ✅")

    # Prediction Section
    if st.button("Predict using saved model"):
        try:
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")

            X = df.drop(columns=[target_col])
            X_scaled = scaler.transform(X)

            predictions = model.predict(X_scaled)

            df['Predicted Loss'] = predictions

            st.subheader("🔮 Predictions")
            st.write(df.head(10))

        except:
            st.error("Train model first!")
