import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

st.title("🏠 House Price Prediction App")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("final_house_price_model.pkl")

model = load_model()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("Bengaluru_House_Data.csv")

df = load_data()

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Model Metrics", "Prediction"]
)

# ---------------- HOME ----------------
if menu == "Home":
    st.subheader("📌 Project Overview")
    st.write("""
    This project predicts **house prices** using Machine Learning.

    **Type:** Regression  
    **Algorithm:** Trained ML Regressor  
    **Target Variable:** Price
    """)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

# ---------------- METRICS ----------------
elif menu == "Model Metrics":
    st.subheader("📈 Model Performance")

    metrics = pd.read_csv("model_metrics.csv")
    st.table(metrics)

    st.subheader("🔍 Feature Importance")
    importance = pd.read_csv("feature_importance.csv")
    st.dataframe(importance)

# ---------------- PREDICTION ----------------
elif menu == "Prediction":
    st.subheader("🔮 Predict House Price")

    st.info("Enter numeric values only")

    # Use only numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = numeric_cols.drop("price", errors="ignore")

    user_input = {}
    for col in numeric_cols:
        user_input[col] = st.number_input(
            col,
            value=float(df[col].median())
        )

    if st.button("Predict Price"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]

        st.success(f"🏷️ Predicted House Price: ₹ {prediction:,.2f}")
