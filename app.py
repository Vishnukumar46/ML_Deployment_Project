import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("house_data.csv")   # <-- your dataset
    return df

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Navigation")

menu = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Model Metrics", "Prediction"]
)

# ---------------- OVERVIEW ----------------
if menu == "Overview":
    st.title("🏠 House Price Prediction System")

    st.markdown("""
    ### End-to-End Machine Learning Regression Project
    This application predicts the **price of a house** based on multiple features
    using Machine Learning regression models.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(20))

    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe())

# ---------------- EDA ----------------
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Target Variable Distribution (House Price)")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["price"], kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("📌 Key Insights")
    st.markdown("""
    - House price increases with **area and number of rooms**
    - Location-related features have strong influence
    - Linear & tree-based models perform well
    """)

# ---------------- MODEL METRICS ----------------
elif menu == "Model Metrics":
    st.title("📈 Model Performance")

    metrics = pd.read_csv("model_metrics.csv")  # MAE, RMSE, R2
    st.subheader("Regression Model Comparison")
    st.dataframe(metrics)

    st.subheader("Actual vs Predicted")
    st.image("actual_vs_predicted.png", use_column_width=True)

# ---------------- PREDICTION ----------------
elif menu == "Prediction":
    st.title("🔮 House Price Prediction")

    model = joblib.load("final_house_price_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")

    st.subheader("Enter House Details")

    user_input = []
    for feature in features:
        value = st.number_input(feature, value=0.0)
        user_input.append(value)

    if st.button("Predict House Price"):
        input_array = np.array(user_input).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]

        st.success(f"🏷️ Estimated House Price: ₹ {prediction:,.2f}")




