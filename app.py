import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.title("📊 AI Stockout Loss Predictor (Auto Best Model)")

uploaded_file = st.file_uploader("Upload Retail Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # 🔥 CREATE TARGET
    if 'Stockout Loss' not in df.columns:
        demand_col, stock_col = None, None

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

    target = "Stockout Loss"

    if st.button("🚀 Train & Select Best Model"):

        X = df.drop(columns=[target])
        y = df[target]

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

        # 🔥 MODELS
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor()
        }

        results = []

        best_model = None
        best_score = -999

        st.subheader("📊 Model Comparison")

        for name, model in models.items():
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            results.append([name, train_acc, test_acc, r2, rmse, mae])

            # Select best
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name

        results_df = pd.DataFrame(results, columns=[
            "Model", "Train R2", "Test R2", "R2 Score", "RMSE", "MAE"
        ])

        st.write(results_df)

        st.success(f"🏆 Best Model: {best_name}")

        # 🔥 PREDICTIONS (BEST MODEL)
        y_pred_best = best_model.predict(X_test)

        st.subheader("📦 Product-wise Predictions")

        result_df = pd.DataFrame({
            "Product ID": pid_test.values,
            "Stock": df.loc[pid_test.index, stock_col].values,
            "Demand": df.loc[pid_test.index, demand_col].values,
            "Predicted Loss": y_pred_best
        })

        for _, row in result_df.head(10).iterrows():
            st.text(f"""
Product {row['Product ID']}:
Stock = {row['Stock']}
Demand = {round(row['Demand'],2)}
Predicted Loss = {round(row['Predicted Loss'],2)}
----------------------------------------
""")

        # 💾 SAVE BEST MODEL
        model_data = {
            "model": best_model,
            "scaler": scaler,
            "encoder": encoder,
            "columns": X.columns.tolist()
        }

        with open("best_model.pkl", "wb") as f:
            pickle.dump(model_data, f)

        # DOWNLOAD
        with open("best_model.pkl", "rb") as f:
            st.download_button("📥 Download Best Model", f, "best_model.pkl")
