import streamlit as st
import pickle
import numpy as np
from PIL import Image

st.title('🌱 SAI Crop Recommendation System 🌱')

st.markdown("""
    <style>
    .stNumberInput label {
        font-weight: bold;
    }
    .stButton button {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    N = st.number_input('Nitrogen (N)')
    P = st.number_input('Phosphorus (P)')
    temperature = st.number_input('Temperature (°C)')
    ph = st.number_input('pH Level')
with col2:
    K = st.number_input('Potassium (K)')
    humidity = st.number_input('Humidity (%)')
    rainfall = st.number_input('Rainfall (mm)')

if st.button('Get Recommendation'):
    # Load the model and scaler
    with open('crop_recommendation_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Calculate NPK_ratio
    NPK_ratio = N + P + K

    # Preprocess the input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall, NPK_ratio]])
    input_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_scaled)[0]

    st.success(f"🌿 The recommended crop for these conditions is: **{prediction.upper()}** 🌿")

    # Display the image if it exists
    try:
        image_path = f"images/{prediction}.jpg"  # Assuming images are in an 'images' folder
        image = Image.open(image_path)
        st.image(image, caption=f"Image of {prediction}", width=300)
    except FileNotFoundError:
        st.warning(f"Image for {prediction} not found at '{image_path}'.")