import streamlit as st
import joblib
import numpy as np

st.title("Housing Price Prediction Dashboard")

# Load the model and preprocessing objects
try:
    theta = joblib.load("model_theta.pkl")
    scaler = joblib.load("scaler.pkl")
    poly = joblib.load("polynomial_features.pkl")
except FileNotFoundError as e:
    st.error(f"Error: {e}. Please check if the model and scaler files exist.")
    st.stop()  # Stop execution if files are missing

# Define sigmoid function BEFORE usage
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Feature names after preprocessing (including one-hot encoding)
feature_names = [
    "Longitude", 
    "Latitude", 
    "Housing Median Age", 
    "Total Rooms", 
    "Total Bedrooms", 
    "Population", 
    "Households", 
    "Median Income", 
    "Ocean Proximity: Near Bay", 
    "Ocean Proximity: Near Ocean",  
    "Ocean Proximity: Island", 
    "Ocean Proximity: Inland"
]

final_feature_count = len(feature_names)  # Adjusted based on the actual feature count

# Sidebar input fields for features
st.sidebar.header("Input Features")
features = []

for i, feature_name in enumerate(feature_names):  # Iterate through feature names
    if feature_name.startswith("Ocean Proximity"):
        # Use radio buttons for one-hot encoded ocean proximity features
        option = st.sidebar.radio(
            f"{feature_name}",
            options=[0, 1],  # 0 means "No", 1 means "Yes"
            index=0,  # Default to 0
            horizontal=True
        )
        features.append(option)
    else:
        # Use number input for the other features
        feature_value = st.sidebar.number_input(f"{feature_name}", value=0.0)
        features.append(feature_value)

# Make prediction
if st.sidebar.button("Predict"):
    try:
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data) 
        input_poly = poly.transform(input_scaled) 
        prediction = (sigmoid(input_poly @ theta) >= 0.5).astype(int)

        st.success(f"Prediction: {'Above Median' if prediction[0] else 'Below Median'}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
