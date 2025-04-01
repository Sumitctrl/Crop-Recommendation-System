import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained models and encoders
scaler = joblib.load("scaler.pkl")
best_model = joblib.load("best_model.pkl")

# Define the prediction function
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    input_data = scaler.transform(input_data)  # Standardize input
    prediction = best_model.predict(input_data)[0]
    return prediction

# Streamlit UI
st.title("🌱 Crop Recommendation System")

st.sidebar.header("Enter Soil and Weather Parameters")

# User input fields
N = st.sidebar.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.sidebar.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
ph = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)

# Prediction button
if st.sidebar.button("Predict Crop"):
    recommended_crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    st.success(f"🌾 Recommended Crop: *{recommended_crop}*")

    # Show crop image based on prediction
    crop_images = {
        "rice": "rice.jpg",
        "apple": "apple.jpg",
        "papaya": "papaya.png",
        "banana": "banana.webp",
        "orange": "orange.webp",
        "coconut": "Coconut.jpg",
        "cotton": "cotton.webp",
        "jute": "jute.webp",
        "coffee": "Coffee.webp",
        "maize": "maize.jpg",
        "chickpea": "chickpea.jpg",
        "kidneybeans": "kidney beans.webp",
        "pigeonpeas": "pigeon-peas.webp",
        "mothbeans": "mothbean.jpg",
        "mungbean": "mungbean.jpg",
        "blackgram": "blackgram.jpg",
        "lentil": "lentil.jpg",
        "pomegranate": "pomegranate.jpg",
        "mango": "mango.jpg",
        "grapes": "grapes.jpg",
        "watermelon": "watermelon.jpg",
        "muskmelon": "muskmelon.jpg",
    }

    # Display the corresponding image
    crop_key = recommended_crop.lower()
    if crop_key in crop_images:
        st.image(crop_images[crop_key], caption=f"Recommended Crop: {recommended_crop}", use_column_width=True)
