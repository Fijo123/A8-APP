import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---- Load all files (no artifacts folder) ----
@st.cache_resource
def load_files():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, features = load_files()

# ---- Streamlit UI ----
st.title("A8-APP : Prediction App")

# Input fields dynamically based on features
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

# When button is pressed
if st.button("Predict"):
    # Convert input to dataframe
    input_df = pd.DataFrame([user_input])
    
    # Scale input
    scaled_input = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(scaled_input)

    st.success(f"Prediction: {prediction[0]}")
