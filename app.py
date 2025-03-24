import streamlit as st
import pickle
import numpy as np

st.title('Crop Recommendation System')

N = st.number_input('Nitrogen (N)')
P = st.number_input('Phosphorus (P)')
K = st.number_input('Potassium (K)')
temperature = st.number_input('Temperature')
humidity = st.number_input('Humidity')
ph = st.number_input('pH')
rainfall = st.number_input('Rainfall')

if st.button('Get Recommendation'):
    # Load the model and scaler
    with open('crop_recommendation_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Calculate NPK_ratio
    NPK_ratio = N + P + K

    # Preprocess the input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall, NPK_ratio]]) # add the NPK_ratio here.
    input_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_scaled)[0]

    st.write(f"Recommended Crop: {prediction}")