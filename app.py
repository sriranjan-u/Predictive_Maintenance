import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from huggingface_hub import hf_hub_download

# 1. Page Configuration
st.set_page_config(page_title="Engine Health Monitor", layout="wide", page_icon="🚢")
st.title("🚢 Predictive Maintenance Decision Support System")
st.markdown("---")

# 2. Load Model from Hugging Face Hub
@st.cache_resource
def load_model():
    # Replace 'Sriranjan' with your exact HF username if different
    repo_id = "Sriranjan/Predictive_Maintenance_Model" 
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename="engine_model.joblib")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model from Hub: {e}")
        return None

model = load_model()

# 3. Sidebar Inputs for Real-time Sensors
st.sidebar.header("Real-time Sensor Telemetry")
st.sidebar.info("Input current engine readings to assess failure risk.")

rpm = st.sidebar.slider("Engine RPM", 400, 1500, 800)
lub_oil_pres = st.sidebar.number_input("Lub Oil Pressure (bar)", 0.0, 10.0, 3.5)
fuel_pres = st.sidebar.number_input("Fuel Pressure (bar)", 0.0, 25.0, 10.0)
cool_pres = st.sidebar.number_input("Coolant Pressure (bar)", 0.0, 10.0, 3.0)
lub_oil_temp = st.sidebar.slider("Lub Oil Temp (°C)", 60, 100, 75)
cool_temp = st.sidebar.slider("Coolant Temp (°C)", 60, 100, 80)

# 4. Data Formatting
# NOTE: Column names must match the training set exactly!
input_data = pd.DataFrame([[rpm, lub_oil_pres, fuel_pres, cool_pres, lub_oil_temp, cool_temp]], 
                          columns=['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                                   'Coolant pressure', 'lub oil temp', 'Coolant temp'])

# 5. Prediction Dashboard
if model:
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Live Sensor Profile")
        st.dataframe(input_data.T.rename(columns={0: "Value"}))
        
    with col2:
        st.subheader("🚨 Diagnostic Result")
        if prediction[0] == 1:
            st.error(f"**CRITICAL FAULT DETECTED**")
            st.metric("Failure Probability", f"{probability:.2%}")
            st.warning("**Recommendation:** Immediate inspection of cooling and lubrication lines. Risk of seizure is high.")
        else:
            st.success(f"**SYSTEM STATUS: HEALTHY**")
            st.metric("Confidence Level", f"{1-probability:.2%}")
            st.info("**Recommendation:** Continue normal operations. Next routine check in 250 operational hours.")
else:
    st.warning("Model not found. Please ensure the model is registered on Hugging Face Hub.")
