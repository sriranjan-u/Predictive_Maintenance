import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Engine Diagnostic", page_icon="🏎️", layout="wide")

# --- CUSTOM THEME CSS (Streamlined UI) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 24px; color: #007bff; }
    .stButton>button { border-radius: 8px; height: 3.5em; background: linear-gradient(to right, #007bff, #0056b3); color: white; font-weight: bold; }
    .prediction-card { padding: 25px; border-radius: 12px; background-color: white; border: 1px solid #dee2e6; text-align: center; }
    /* Compact Sidebar */
    section[data-testid="stSidebar"] { width: 350px !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    repo_id = "Sriranjan/Predictive_Maintenance_Model"
    path = hf_hub_download(repo_id=repo_id, filename="engine_pipeline.joblib")
    return joblib.load(path)

model = load_model()

# --- SIDEBAR: COMPACT CONTROLS ---
with st.sidebar:
    st.header("🕹️ Telemetry Control")
    
    # 🌡️ Temperature Group (Side-by-Side)
    st.markdown("**Thermal Sensors**")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        lub_t = st.number_input("Lub Oil (°C)", 60, 100, 77)
    with t_col2:
        cool_t = st.number_input("Coolant (°C)", 60, 100, 78)
        
    st.divider()

    # 🧪 Pressure Group (Compact Layout)
    st.markdown("**Pressure Sensors (bar)**")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        fuel_p = st.number_input("Fuel P.", 0.0, 20.0, 6.6, step=0.1)
        cool_p = st.number_input("Coolant P.", 0.0, 10.0, 2.3, step=0.1)
    with p_col2:
        lub_p = st.number_input("Lub Oil P.", 0.0, 10.0, 3.3, step=0.1)
        rpm = st.number_input("RPM Speed", 0, 3000, 800, step=50)

    st.divider()
    st.info("💡 Adjust values to run diagnostic.")

# --- MAIN DASHBOARD ---
st.title("🏎️ Engine Health Diagnostic System")
st.markdown("Automated Failure Prediction using High-Fidelity Sensor Data [cite: 3]")

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Lubrication", f"{lub_p} bar", f"{lub_t}°C")
m2.metric("Fuel System", f"{fuel_p} bar", "Stable")
m3.metric("Cooling", f"{cool_p} bar", f"{cool_t}°C")
m4.metric("Engine Speed", f"{rpm} RPM", "Active")

st.divider()

# Analysis Section
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("📡 Live Telemetry Stream")
    input_df = pd.DataFrame({
        'Engine rpm': [rpm], 'Lub oil pressure': [lub_p], 
        'Fuel pressure': [fuel_p], 'Coolant pressure': [cool_p], 
        'lub oil temp': [lub_t], 'Coolant temp': [cool_t]
    })
    st.dataframe(input_df, hide_index=True, use_container_width=True)
    
    st.write("**Thermal Saturation Index**")
    st.progress(max(0, min((cool_t - 60) / 40, 1.0)))

with right_col:
    st.subheader("🧠 AI Diagnostic Verdict")
    if st.button("EXECUTE SYSTEM CHECK"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][pred] * 100
        
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        if pred == 0:
            st.balloons()
            st.markdown("<h2 style='color:#28a745;'>✅ SYSTEM HEALTHY</h2>", unsafe_allow_html=True)
            st.write(f"Confidence Level: **{prob:.1f}%**")
        else:
            st.markdown("<h2 style='color:#dc3545;'>⚠️ MAINTENANCE REQUIRED</h2>", unsafe_allow_html=True)
            st.write(f"Failure Probability: **{prob:.1f}%**")
            st.error("Diagnostic Note: Abnormal thermal-pressure signature detected.")
        st.markdown('</div>', unsafe_allow_html=True)

st.caption(f"© 2026 MLOps Framework | Developed by Sriranjan Uppoor [cite: 4, 292]")
