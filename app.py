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

st.set_page_config(layout="wide")
st.title("📊 Smart Stockout & Revenue Loss Predictor")

# ------------------ STEP 1: LOAD DATA ------------------
file = st.file_uploader("Upload Retail Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("Dataset Loaded Successfully ✅")
    st.dataframe(df.head())

    # ------------------ STEP 2: DETECT COLUMNS ------------------
    demand_col, stock_col, price_col = None, None, None

    for col in df.columns:
        if "demand" in col.lower():
            demand_col = col
        if "stock" in col.lower() or "inventory" in col.lower():
            stock_col = col
        if "price" in col.lower() or "revenue" in col.lower():
            price_col = col

    if not (demand_col and stock_col and price_col):
        st.error("❌ Dataset must contain Demand, Stock, and Price columns")
        st.stop()

    st.success(f"Detected → Demand: {demand_col}, Stock: {stock_col}, Price: {price_col}")

    # ------------------ STEP 3: FEATURE ENGINEERING ------------------
    df['Demand_Stock_Ratio'] = df[demand_col] / (df[stock_col] + 1)
    df['Shortage'] = np.maximum(df[demand_col] - df[stock_col], 0)

    # ------------------ STEP 4: MODEL SELECTION ------------------
    st.header("Select Models")

    selected_models = st.multiselect(
        "Choose Models",
        ["Linear Regression", "Ridge", "Random Forest", "Gradient Boosting"],
        default=["Random Forest", "Gradient Boosting"]
    )

    # ------------------ STEP 5: TRAIN ------------------
    if st.button("🚀 Train Models"):

        target = demand_col

        X = df.drop(columns=[target])
        y = df[target]

        product_ids = df["Product ID"] if "Product ID" in df.columns else pd.Series(range(len(df)))

        # Encoding
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        cat_cols = X.select_dtypes(include=['object']).columns

        if len(cat_cols) > 0:
            X[cat_cols] = encoder.fit_transform(X[cat_cols])

        # Split (FIXED)
        X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
            X, y, product_ids, test_size=0.2, random_state=42
        )

        # Reset index
        X_test = pd.DataFrame(X_test).reset_index(drop=True)
        pid_test = pd.Series(pid_test).reset_index(drop=True)

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models
        model_dict = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor()
        }

        results = []
        best_model = None
        best_score = -999
        best_name = ""
        best_pred = None

        for name in selected_models:
            model = model_dict[name]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test_scaled)

            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test_scaled, y_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            results.append([name, train_acc, test_acc, r2, rmse, mae])

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name
                best_pred = y_pred

        # ------------------ STEP 6: RESULTS ------------------
        results_df = pd.DataFrame(results, columns=[
            "Model", "Train R2", "Test R2", "R2", "RMSE", "MAE"
        ])

        st.subheader("📊 Model Comparison")
        st.dataframe(results_df)

        st.success(f"🏆 Best Model: {best_name}")

        # ------------------ STEP 7: LOSS CALCULATION ------------------
        stock_values = df.iloc[pid_test.index][stock_col].values
        price_values = df.iloc[pid_test.index][price_col].values

        predicted_loss = np.maximum(best_pred - stock_values, 0) * price_values

        # ------------------ STEP 8: PRODUCT OUTPUT ------------------
        st.subheader("📦 Product-wise Predictions")

        result_df = pd.DataFrame({
            "Product ID": pid_test,
            "Stock": stock_values,
            "Predicted Demand": best_pred,
            "Predicted Loss": predicted_loss
        })

        for _, row in result_df.head(10).iterrows():
            st.text(f"""
Product {row['Product ID']}:
Stock = {int(row['Stock'])}
Demand = {round(row['Predicted Demand'], 2)}
Predicted Loss = {round(row['Predicted Loss'], 2)}
----------------------------------------
""")

        # ------------------ STEP 9: SAVE MODEL ------------------
        model_data = {
            "model": best_model,
            "scaler": scaler,
            "encoder": encoder,
            "columns": X.columns.tolist(),
            "stock_col": stock_col,
            "price_col": price_col,
            "demand_col": demand_col
        }

        with open("final_model.pkl", "wb") as f:
            pickle.dump(model_data, f)

        st.success("✅ Model Saved Successfully")

        with open("final_model.pkl", "rb") as f:
            st.download_button("📥 Download Model", f, "final_model.pkl")

    # ------------------ STEP 10: FULL DATA PREDICTION ------------------
    if st.button("🔮 Predict Full Dataset"):
        try:
            with open("final_model.pkl", "rb") as f:
                data = pickle.load(f)

            model = data["model"]
            scaler = data["scaler"]
            encoder = data["encoder"]
            saved_cols = data["columns"]
            stock_col = data["stock_col"]
            price_col = data["price_col"]
            demand_col = data["demand_col"]

            X = df.drop(columns=[demand_col])

            cat_cols = X.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                X[cat_cols] = encoder.transform(X[cat_cols])

            for col in saved_cols:
                if col not in X:
                    X[col] = 0

            X = X[saved_cols]
            X_scaled = scaler.transform(X)

            pred_demand = model.predict(X_scaled)

            df["Predicted Demand"] = pred_demand
            df["Predicted Loss"] = np.maximum(
                pred_demand - df[stock_col], 0
            ) * df[price_col]

            st.dataframe(df.head(20))

        except Exception as e:
            st.error(f"Error: {e}")

    # ------------------ STEP 11: MANUAL PRODUCT CHECK ------------------
    st.header("🔍 Manual Product Loss Check")

    if st.button("Load Model for Manual Prediction"):
        try:
            with open("final_model.pkl", "rb") as f:
                data = pickle.load(f)

            model = data["model"]
            scaler = data["scaler"]
            encoder = data["encoder"]
            saved_cols = data["columns"]
            stock_col = data["stock_col"]
            price_col = data["price_col"]
            demand_col = data["demand_col"]

            if "Product ID" in df.columns:
                product_id = st.selectbox("Select Product ID", df["Product ID"].unique())
                product_data = df[df["Product ID"] == product_id].iloc[0]

                st.write("Product Details", product_data)

                stock_input = st.number_input("Stock Available", value=float(product_data[stock_col]))
                price_input = st.number_input("Price", value=float(product_data[price_col]))

                input_df = pd.DataFrame([product_data])

                if demand_col in input_df.columns:
                    input_df = input_df.drop(columns=[demand_col])

                cat_cols = input_df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    input_df[cat_cols] = encoder.transform(input_df[cat_cols])

                for col in saved_cols:
                    if col not in input_df:
                        input_df[col] = 0

                input_df = input_df[saved_cols]
                input_scaled = scaler.transform(input_df)

                pred_demand = model.predict(input_scaled)[0]
                pred_loss = max(pred_demand - stock_input, 0) * price_input

                st.subheader("📦 Manual Prediction Result")

                st.text(f"""
Product {product_id}:
Stock = {stock_input}
Predicted Demand = {round(pred_demand, 2)}
Predicted Loss = {round(pred_loss, 2)}
----------------------------------------
""")

            else:
                st.warning("No Product ID column found")

        except Exception as e:
            st.error(f"Error: {e}")
