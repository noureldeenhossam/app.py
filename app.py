# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and features
with open('booking_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

st.set_page_config(page_title="Booking Status Predictor", layout="centered")

st.title("🧠 Booking Status Predictor")
st.write("Enter booking information below to predict if it will be **cancelled** or **not cancelled**.")

# Create input form
user_input = {}
with st.form(key="booking_form"):
    for col in feature_columns:
        user_input[col] = st.number_input(f"{col}", format="%.4f")

    submit = st.form_submit_button("Predict")

if submit:
    try:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        label = "Cancelled ❌" if prediction == 1 else "Not Cancelled ✅"
        st.success(f"Prediction: {label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
