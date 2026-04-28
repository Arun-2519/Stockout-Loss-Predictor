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

# ------------------ STEP 1: LOAD ------------------
file = st.file_uploader("Upload Retail Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("Dataset Loaded ✅")
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
        st.error("❌ Dataset must have Demand, Stock, and Price columns")
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

    # ------------------ STEP 5: TRAIN DEMAND MODEL ------------------
    if st.button("🚀 Train Model"):

        target = demand_col   # 🔥 Predict demand instead of loss

        X = df.drop(columns=[target])
        y = df[target]

        product_ids = df["Product ID"] if "Product ID" in df.columns else pd.Series(range(len(df)))

        # Encoding
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        cat_cols = X.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            X[cat_cols] = encoder.fit_transform(X[cat_cols])

        # Split
        X_train, X_test, y_train, y_test, pid_test = train_test_split(
            X, y, product_ids, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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

        for name in selected_models:
            model = model_dict[name]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            results.append([name, train_acc, test_acc, r2, rmse, mae])

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name

        results_df = pd.DataFrame(results, columns=[
            "Model", "Train R2", "Test R2", "R2 Score", "RMSE", "MAE"
        ])

        st.subheader("📊 Model Comparison")
        st.dataframe(results_df)

        st.success(f"🏆 Best Model: {best_name}")

        # ------------------ STEP 6: PREDICT DEMAND ------------------
        y_pred_best = best_model.predict(X_test)

        # ------------------ STEP 7: CALCULATE LOSS ------------------
        stock_values = df.loc[pid_test.index, stock_col].values
        price_values = df.loc[pid_test.index, price_col].values

        predicted_loss = np.maximum(y_pred_best - stock_values, 0) * price_values

        # ------------------ STEP 8: OUTPUT ------------------
        st.subheader("📦 Product-wise Output")

        result_df = pd.DataFrame({
            "Product ID": pid_test.values,
            "Stock": stock_values,
            "Actual Demand": df.loc[pid_test.index, demand_col].values,
            "Predicted Demand": y_pred_best,
            "Predicted Loss": predicted_loss
        })

        for _, row in result_df.head(10).iterrows():
            st.text(f"""
Product {row['Product ID']}:
Stock = {int(row['Stock'])}
Demand = {round(row['Predicted Demand'],2)}
Predicted Loss = {round(row['Predicted Loss'],2)}
----------------------------------------
""")

        # ------------------ SAVE MODEL ------------------
        model_data = {
            "model": best_model,
            "scaler": scaler,
            "encoder": encoder,
            "columns": X.columns.tolist(),
            "stock_col": stock_col,
            "price_col": price_col
        }

        with open("smart_model.pkl", "wb") as f:
            pickle.dump(model_data, f)

        st.success("Model Saved ✅")

        with open("smart_model.pkl", "rb") as f:
            st.download_button("📥 Download Model", f, "smart_model.pkl")

    # ------------------ STEP 9: FULL PREDICTION ------------------
    if st.button("🔮 Predict Full Dataset"):
        try:
            with open("smart_model.pkl", "rb") as f:
                data = pickle.load(f)

            model = data["model"]
            scaler = data["scaler"]
            encoder = data["encoder"]
            saved_cols = data["columns"]

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
            st.error(e)
