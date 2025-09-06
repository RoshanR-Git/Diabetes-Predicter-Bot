import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load("diabetes_predicter_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction App 🩺")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Collect inputs
inputs = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Scale inputs
inputs_scaled = scaler.transform(inputs)

# Predict
if st.button("Predict"):
    prediction = model.predict(inputs_scaled)[0]
    probability = model.predict_proba(inputs_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ You are Prone to Diabetes!!!(Probability: {probability:.2f})")
    else:
        st.success(f"✅ You have zero possibility for Diabetes...(Probability: {probability:.2f})")

