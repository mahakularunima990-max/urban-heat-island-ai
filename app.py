import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

st.title("Urban Heat Island Risk Prediction")

st.write("Enter environmental data to predict heat risk")

# User inputs
temperature = st.number_input("Temperature (°C)")
ndvi = st.number_input("NDVI (Vegetation Index)")
building_density = st.number_input("Building Density")
population = st.number_input("Population Density")

if st.button("Predict Heat Risk"):
    
    input_data = np.array([[temperature, ndvi, building_density, population]])
    
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Heat Risk: {prediction[0]}")