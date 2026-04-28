import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(layout="wide")
st.title("📊 Advanced Stockout Loss Predictor")

uploaded_file = st.file_uploader("Upload Retail Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # 🔥 AUTO TARGET
    if 'Stockout Loss' not in df.columns:
        st.warning("Creating Stockout Loss automatically")

        demand_col = None
        stock_col = None

        for col in df.columns:
            if "demand" in col.lower():
                demand_col = col
            if "stock" in col.lower() or "inventory" in col.lower():
                stock_col = col

        if demand_col and stock_col:
            df['Stockout Loss'] = np.maximum(df[demand_col] - df[stock_col], 0)
        else:
            st.error("Need Demand & Stock columns")
            st.stop()

    target_col = "Stockout Loss"

    # 🔥 MODEL SELECTION
    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest (Best)", "Gradient Boosting", "Linear Regression"]
    )

    if st.button("🚀 Train Model"):

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Save Product ID if exists
        product_ids = df["Product ID"] if "Product ID" in df.columns else pd.Series(range(len(df)))

        # Encoding
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        cat_cols = X.select_dtypes(include=['object']).columns

        if len(cat_cols) > 0:
            X[cat_cols] = encoder.fit_transform(X[cat_cols])

        # Split
        X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
            X, y, product_ids, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 🔥 MODEL LOGIC
        if model_choice == "Random Forest (Best)":
            model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1)
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor()
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # 🔥 EVALUATION METRICS
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        st.subheader("📈 Evaluation Metrics")
        st.write(f"R2 Score: {r2:.3f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAE: {mae:.2f}")

        # 🔥 CUSTOM OUTPUT FORMAT
        st.subheader("📦 Product-wise Predictions")

        result_df = pd.DataFrame({
            "Product ID": pid_test.values,
            "Stock": df.loc[pid_test.index, stock_col].values,
            "Demand": df.loc[pid_test.index, demand_col].values,
            "Predicted Loss": y_pred
        })

        for _, row in result_df.head(10).iterrows():
            st.text(f"""
Product {row['Product ID']}:
Stock = {row['Stock']}
Demand = {round(row['Demand'],2)}
Predicted Loss = {round(row['Predicted Loss'],2)}
----------------------------------------
""")

        # 💾 SAVE MODEL
        model_data = {
            "model": model,
            "scaler": scaler,
            "encoder": encoder,
            "columns": X.columns.tolist()
        }

        with open("model.pkl", "wb") as f:
            pickle.dump(model_data, f)

        st.success("✅ Model Saved!")

        # 📥 DOWNLOAD
        with open("model.pkl", "rb") as f:
            st.download_button("📥 Download Model", f, "model.pkl")

    # 🔮 PREDICTION MODE
    if st.button("🔮 Predict (Full Dataset)"):
        try:
            with open("model.pkl", "rb") as f:
                data = pickle.load(f)

            model = data["model"]
            scaler = data["scaler"]
            encoder = data["encoder"]
            saved_cols = data["columns"]

            X = df.drop(columns=[target_col])

            cat_cols = X.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                X[cat_cols] = encoder.transform(X[cat_cols])

            for col in saved_cols:
                if col not in X:
                    X[col] = 0

            X = X[saved_cols]
            X_scaled = scaler.transform(X)

            preds = model.predict(X_scaled)
            df["Predicted Loss"] = preds

            st.write(df.head(10))

        except Exception as e:
            st.error(e)
