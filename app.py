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

# ---------------- SQFT CLEANING (FOR EDA) ----------------
def convert_sqft_to_num(x):
    try:
        if "-" in str(x):
            a, b = x.split("-")
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return np.nan

df["total_sqft_num"] = df["total_sqft"].apply(convert_sqft_to_num)

# ---------------- SIDEBAR ----------------
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
    - Data understanding  
    - Exploratory Data Analysis (EDA)  
    - Model evaluation  
    - Deployment using Streamlit  
    """)

    st.write("### 📄 Example Dataset")
    st.dataframe(df.sample(5))

    st.write("### 📊 Statistical Summary")
    st.dataframe(df.describe())

# ---------------- EDA ----------------
elif menu == "EDA":
    st.subheader("📊 Exploratory Data Analysis")

    # Price Distribution
    st.write("### 📈 Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["price"], kde=True, ax=ax)
    st.pyplot(fig)

    # Sqft vs Price
    st.write("### 📐 Total Sqft vs Price")
    sqft_price_df = df.dropna(subset=["total_sqft_num", "price"])

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=sqft_price_df["total_sqft_num"],
        y=sqft_price_df["price"],
        ax=ax
    )
    ax.set_xlabel("Total Sqft")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    # Heatmap
    st.write("### 🔥 Correlation Heatmap")

    numeric_df = df[[
        "price",
        "total_sqft_num",
        "bath"
    ]].dropna()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
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

    if st.button("Predict Price"):
        # Demo prediction logic
        estimated_price = sqft * 5000 + bath * 200000
        st.success(f"🏷️ Estimated House Price: ₹ {estimated_price:,.2f}")
