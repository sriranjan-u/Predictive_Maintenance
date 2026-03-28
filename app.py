import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# 1. Page Configuration
st.set_page_config(page_title="Engine Health Monitor", layout="wide")
st.title("🚢 Predictive Maintenance Decision Support System")
st.markdown("---")

# 2. Load Model from Hugging Face Hub
@st.cache_resource
def load_model():
    # Replace with your actual repo ID
    repo_id = "Sriranjan/Predictive_Maintenance_Model" 
    model_path = hf_hub_download(repo_id=repo_id, filename="engine_model.joblib")
    return joblib.load(model_path)

model = load_model()

# 3. Sidebar Inputs for Sensors
st.sidebar.header("Real-time Sensor Telemetry")
rpm = st.sidebar.slider("Engine RPM", 400, 1500, 800)
lub_oil_pres = st.sidebar.number_input("Lub Oil Pressure", 0.0, 10.0, 3.5)
fuel_pres = st.sidebar.number_input("Fuel Pressure", 0.0, 25.0, 10.0)
coolant_pres = st.sidebar.number_input("Coolant Pressure", 0.0, 10.0, 3.0)
lub_oil_temp = st.sidebar.slider("Lub Oil Temp (°C)", 60, 100, 75)
coolant_temp = st.sidebar.slider("Coolant Temp (°C)", 60, 100, 80)

# 4. Prediction Logic
input_data = pd.DataFrame([[rpm, lub_oil_pres, fuel_pres, coolant_pres, lub_oil_temp, coolant_temp]], 
                          columns=['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                                   'Coolant pressure', 'lub oil temp', 'Coolant temp'])

prediction = model.predict(input_data)
probability = model.predict_proba(input_data)[0][1]

# 5. Dashboard Display
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sensor Profile")
    st.write(input_data)
    
with col2:
    st.subheader("Status Prediction")
    if prediction[0] == 1:
        st.error(f"FAULT DETECTED (Probability: {probability:.2%})")
        st.warning("Action Required: Schedule immediate inspection of cooling and lubrication systems.")
    else:
        st.success(f"ENGINE HEALTHY (Confidence: {1-probability:.2%})")
        st.info("Status: Normal operations. No immediate maintenance required.")
