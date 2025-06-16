import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load encoders and models
with open("brand_encoder.pkl", "rb") as f:
    brand_encoder = pickle.load(f)
with open("fuel_encoder.pkl", "rb") as f:
    fuel_encoder = pickle.load(f)
with open("model_lr.pkl", "rb") as f:
    model_lr = pickle.load(f)
with open("model_rf.pkl", "rb") as f:
    model_rf = pickle.load(f)

# Load data
df = pd.read_csv("car_data_set.csv")

# App layout
st.set_page_config(layout="wide")

st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio("Go to", ["🏠 Home", "📁 Dataset", "📊 Visualizations", "🧠 Predictor"])

# Main Title
st.markdown("<h1 style='text-align:center;'>🚗 Car Price Prediction Tool</h1>", unsafe_allow_html=True)

# Sections
if section == "🏠 Home":
    st.markdown("""
    ### 🔧 About This App
    This tool lets you:
    - View the training dataset
    - Explore key data visualizations
    - Predict car prices using:
        - Linear Regression
        - Random Forest

    Use the sidebar to begin ➡️
    """)

elif section == "📁 Dataset":
    st.subheader("🔍 Dataset Used for Training")
    st.dataframe(df)

elif section == "📊 Visualizations":
    st.subheader("📊 Visual Data Insights")

    fig1 = px.histogram(df, x='Selling_Price', nbins=50, title="Selling Price Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(df, x='Fuel', y='Selling_Price', title="Selling Price by Fuel Type")
    st.plotly_chart(fig2, use_container_width=True)

elif section == "🧠 Predictor":
    st.subheader("⚙️ Select Model")
    model_option = st.radio("Choose a Model", ["Linear Regression", "Random Forest"])

    st.markdown("### 🔢 Enter Car Details")
    brand = st.selectbox("Brand", df['Brand'].unique())
    year = st.slider("Manufacturing Year", 1995, 2025, 2015)
    km_driven = st.number_input("Kilometers Driven", 0, 500000, 30000)
    fuel = st.selectbox("Fuel Type", df['Fuel'].unique())

    if st.button("🚀 Predict"):
        brand_encoded = brand_encoder.transform([brand])[0]
        fuel_encoded = fuel_encoder.transform([fuel])[0]
        car_age = 2025 - year

        input_data = [[brand_encoded, car_age, km_driven, fuel_encoded]]

        if model_option == "Linear Regression":
            prediction = model_lr.predict(input_data)[0]
            st.success(f"💸 Predicted Price (Linear Regression): ₹ {max(0, int(prediction)):,}")
        else:
            prediction = model_rf.predict(input_data)[0]
            st.success(f"💸 Predicted Price (Random Forest): ₹ {max(0, int(prediction)):,}")
