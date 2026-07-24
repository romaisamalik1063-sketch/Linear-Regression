import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("/content/drive/MyDrive/ml_model_deployment/linear_regression_model.joblib")
scaler = joblib.load("/content/drive/MyDrive/ml_model_deployment/scaler.joblib")

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction")
st.write("Enter the property details below.")

# Numeric Inputs
area = st.number_input("Area (sq ft)", min_value=500, value=5000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
stories = st.number_input("Stories", min_value=1, max_value=5, value=2)
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)

# Categorical Inputs
mainroad = st.selectbox("Main Road", ["Yes", "No"])
guestroom = st.selectbox("Guest Room", ["Yes", "No"])
basement = st.selectbox("Basement", ["Yes", "No"])
hotwater = st.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
furnishing = st.selectbox(
    "Furnishing Status",
    ["Furnished", "Semi-furnished", "Unfurnished"]
)

# Convert categorical values to model format
mainroad_yes = 1 if mainroad == "Yes" else 0
guestroom_yes = 1 if guestroom == "Yes" else 0
basement_yes = 1 if basement == "Yes" else 0
hotwater_yes = 1 if hotwater == "Yes" else 0
airconditioning_yes = 1 if airconditioning == "Yes" else 0
prefarea_yes = 1 if prefarea == "Yes" else 0

furnishing_semi = 1 if furnishing == "Semi-furnished" else 0
furnishing_un = 1 if furnishing == "Unfurnished" else 0

# Feature order must match training
features = np.array([[
    area,
    bedrooms,
    bathrooms,
    stories,
    parking,
    mainroad_yes,
    guestroom_yes,
    basement_yes,
    hotwater_yes,
    airconditioning_yes,
    prefarea_yes,
    furnishing_semi,
    furnishing_un
]])

# Scale features
features_scaled = scaler.transform(features)

if st.button("Predict House Price"):
    prediction = model.predict(features_scaled)
    st.success(f"Predicted House Price: Rs. {prediction[0]:,.0f}")
