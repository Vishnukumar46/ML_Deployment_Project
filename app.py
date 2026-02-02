import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# ---------------- SIDEBAR NAVIGATION ----------------
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "EDA", "Model Metrics", "Prediction"]
)

# ---------------- OVERVIEW ----------------
if menu == "Overview":
    st.subheader("📌 Project Overview")
    st.write("""
    **Problem Type:** Regression  
    **Target Variable:** Price  
    **Dataset:** Bengaluru House Data  

    This project demonstrates:
    - Data preprocessing  
    - Exploratory Data Analysis (EDA)  
    - Model training & evaluation  
    - Deployment using Streamlit  
    """)

# ---------------- EDA ----------------
elif menu == "EDA":
    st.subheader("📊 Exploratory Data Analysis")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    # Price Distribution
    st.write("### 📈 Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["price"], kde=True, ax=ax)
    st.pyplot(fig)

    # Sqft vs Price
    st.write("### 📐 Total Sqft vs Price")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df["total_sqft"],
        y=df["price"],
        ax=ax
    )
    st.pyplot(fig)

    # BHK vs Price
    st.write("### 🏘️ BHK vs Price")
    fig, ax = plt.subplots()
    sns.boxplot(
        x=df["bhk"],
        y=df["price"],
        ax=ax
    )
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("### 🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------- MODEL METRICS ----------------
elif menu == "Model Metrics":
    st.subheader("🤖 Model Performance")

    metrics = pd.read_csv("model_metrics.csv")
    st.table(metrics)

    st.subheader("🔍 Feature Importance")
    importance = pd.read_csv("feature_importance.csv")
    st.dataframe(importance)

# ---------------- PREDICTION ----------------
elif menu == "Prediction":
    st.subheader("🔮 House Price Prediction")

    sqft = st.number_input("Total Square Feet", value=1000)
    bath = st.number_input("Number of Bathrooms", value=2)
    bhk = st.number_input("Number of BHK", value=2)

    if st.button("Predict Price"):
        # Demo logic
        estimated_price = sqft * 5000 + bath * 200000 + bhk * 300000
        st.success(f"🏷️ Estimated House Price: ₹ {estimated_price:,.2f}")
