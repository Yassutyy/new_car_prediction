import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load all models and encoders
with open("brand_encoder.pkl", "rb") as f:
    brand_encoder = pickle.load(f)
with open("fuel_encoder.pkl", "rb") as f:
    fuel_encoder = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model_lr.pkl", "rb") as f:
    model_lr = pickle.load(f)
with open("model_rf.pkl", "rb") as f:
    model_rf = pickle.load(f)

# Load dataset
df = pd.read_csv("car_data_set.csv")

# Streamlit config
st.set_page_config(layout="wide")
st.sidebar.title("🧭 Navigation")
option = st.sidebar.radio("Go to", ["🏠 Home", "📁 Dataset", "📊 Visualizations", ,"📄 Model Evaluation ","🧠 Predictor"])

# Title
st.markdown("<h1 style='text-align: center;'>🚗 Car Price Prediction Tool</h1>", unsafe_allow_html=True)

# Home
if option == "🏠 Home":
    st.markdown("""
        ### 🧾 What this app does:
        - Shows dataset used for training the car price predictor
        - Visualizes car price trends
        - Predicts price using:
            - Linear Regression
            - Random Forest
        ---
        👉 Use the sidebar to explore!
    """)
    st.caption("Developed by B.Yaswanth, A.Dinesh, SK.Baji")

# Dataset
elif option == "📁 Dataset":
    st.subheader("📁 Training Dataset")
    st.dataframe(df)

# Visualizations
elif option == "📊 Visualizations":
    st.subheader("📊 Visualizations")
    fig1 = px.histogram(df, x="Selling_Price", nbins=50, title="Selling Price Distribution", marginal="box")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(df, x="Fuel", y="Selling_Price", title="Selling Price by Fuel Type")
    st.plotly_chart(fig2, use_container_width=True)

elif option == " Model Evaluation ":
    st.subheader("R2 Scores and Mean Squared error ")
    st.subheader("Linear Regression :  R2 score = 0.31030 and Mean Squared Error : 234920020059.23")
     st.subheader("Random Forest  :  R2 score = 0.648 and Mean Squared Error : 107115499435.73")

# Predictor
elif option == "🧠 Predictor":
    st.subheader("⚙️ Choose Model")
    model_choice = st.radio("Select Model", ["Linear Regression : R2 Score () ", "Random Forest : R2 Score () "])

    st.markdown("### 📥 Input Car Details")
    brand = st.selectbox("Brand", df["Brand"].unique())
    year = st.slider("Year of Manufacture", 1995, 2025, 2015)
    km_driven = st.number_input("KM Driven", 0, 500000, 30000)
    fuel = st.selectbox("Fuel Type", df["Fuel"].unique())

    if st.button("🚀 Predict"):
        try:
            brand_encoded = brand_encoder.transform([brand])[0]
            fuel_encoded = fuel_encoder.transform([fuel])[0]
            car_age = 2025 - year

            input_data = [[brand_encoded, car_age, km_driven, fuel_encoded]]
            input_scaled = scaler.transform(input_data)

            if model_choice == "Linear Regression":
                pred = model_lr.predict(input_scaled)[0]
                st.success(f"💰 Predicted Price : ₹ {int(pred):,}")
            else:
                pred = model_rf.predict(input_scaled)[0]
                st.success(f"🌲 Predicted Price : ₹ {int(pred):,}")
        except Exception as e:
            st.error("⚠️ Prediction failed. Check your inputs")
