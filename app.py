import streamlit as st
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Prediction App",
    layout="centered"
)

st.title("🏠 House Price Prediction App")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("Bengaluru_House_Data.csv")

df = load_data()

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Dataset", "Prediction (Demo)", "Model Info"]
)

# ---------------- OVERVIEW ----------------
if menu == "Overview":
    st.subheader("📌 Project Overview")
    st.write("""
    **Problem Type:** Regression  
    **Target Variable:** House Price  
    **Dataset:** Bengaluru House Data  

    This project demonstrates the **end-to-end ML workflow**:
    - Data preprocessing
    - Feature selection
    - Model training
    - Evaluation
    - Deployment using Streamlit
    """)

# ---------------- DATASET ----------------
elif menu == "Dataset":
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Dataset Shape")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

# ---------------- PREDICTION (DEMO) ----------------
elif menu == "Prediction (Demo)":
    st.subheader("🔮 House Price Prediction (Demo Mode)")

    st.info("This is a demo prediction to showcase deployment.")

    sqft = st.number_input("Total Square Feet", value=1000)
    bath = st.number_input("Number of Bathrooms", value=2)
    bhk = st.number_input("Number of BHK", value=2)

    if st.button("Predict Price"):
        # Simple mock logic (demo)
        estimated_price = sqft * 5000 + bath * 200000 + bhk * 300000
        st.success(f"🏷️ Estimated House Price: ₹ {estimated_price:,.2f}")

# ---------------- MODEL INFO ----------------
elif menu == "Model Info":
    st.subheader("🤖 Model Details")

    metrics = pd.read_csv("model_metrics.csv")
    st.table(metrics)

    st.subheader("🔍 Feature Importance")
    importance = pd.read_csv("feature_importance.csv")
    st.dataframe(importance)
