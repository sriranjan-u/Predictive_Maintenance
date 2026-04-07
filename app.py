import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Engine Diagnostic", page_icon="🏎️", layout="wide")

# --- CUSTOM THEME CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #1f77b4; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1f77b4; color: white; }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    repo_id = "Sriranjan/Predictive_Maintenance_Model"
    path = hf_hub_download(repo_id=repo_id, filename="engine_pipeline.joblib")
    return joblib.load(path)

model = load_model()

# --- SIDEBAR: INPUT CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car-service.png", width=80)
    st.header("Diagnostic Inputs")
    st.info("Adjust sliders to simulate live engine telemetry.")
    
    with st.expander("🌡️ Temperature Settings", expanded=True):
        lub_t = st.slider("Lub Oil Temp (°C)", 60, 100, 77)
        cool_t = st.slider("Coolant Temp (°C)", 60, 100, 78)
        
    with st.expander("🧪 Pressure Settings", expanded=True):
        fuel_p = st.slider("Fuel Pressure (bar)", 0.0, 20.0, 6.6)
        lub_p = st.slider("Lub Oil Pressure (bar)", 0.0, 10.0, 3.3)
        cool_p = st.slider("Coolant Pressure (bar)", 0.0, 10.0, 2.3)
        
    with st.expander("⚙️ Engine Speed", expanded=True):
        rpm = st.number_input("RPM", 0, 3000, 800)

# --- MAIN CONTENT ---
st.title("🏎️ Engine Health Diagnostic System")
st.markdown("Automated Failure Prediction using High-Fidelity Sensor Data")

# Create a 3-column top row for key metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Lubrication System", f"{lub_p} bar", f"{lub_t}°C")
with col2:
    st.metric("Fuel System", f"{fuel_p} bar", "Stable")
with col3:
    st.metric("Cooling System", f"{cool_p} bar", f"{cool_t}°C")

st.divider()

# Lower Section: Analysis and Visualization
left_col, right_col = st.columns([1.5, 1])

with left_col:
    st.subheader("📡 Live Telemetry Stream")
    input_df = pd.DataFrame({
        'Engine rpm': [rpm], 'Lub oil pressure': [lub_p], 
        'Fuel pressure': [fuel_p], 'Coolant pressure': [cool_p], 
        'lub oil temp': [lub_t], 'Coolant temp': [cool_t]
    })
    st.table(input_df)
    
    # Simple Visual feedback
    st.write("Temperature Saturation")
    st.progress(max(0, min((cool_t - 60) / 40, 1.0)))

with right_col:
    st.subheader("🧠 AI Verdict")
    if st.button("RUN SYSTEM CHECK"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][pred] * 100
        
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        if pred == 0:
            st.balloons()
            st.markdown(f"<h1 style='color:green;'>✅ HEALTHY</h1>", unsafe_allow_html=True)
            st.write(f"System Confidence: **{prob:.1f}%**")
        else:
            st.markdown(f"<h1 style='color:red;'>⚠️ FAULT RISK</h1>", unsafe_allow_html=True)
            st.write(f"System Confidence: **{prob:.1f}%**")
            st.warning("Immediate inspection required: Abnormal thermal/pressure signature detected.")
        st.markdown('</div>', unsafe_allow_html=True)

st.caption("© 2026 MLOps Framework | Developed by Sriranjan Uppoor")
