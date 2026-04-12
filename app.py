import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Engine Diagnostic", page_icon="🏎️", layout="wide")

# --- CUSTOM THEME CSS (Compact & High-Contrast) ---
st.markdown("""
    <style>
    /* Reduce top padding of the main block */
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
    
    /* Style for the main header to stay compact */
    .main-title { font-size: 32px; font-weight: bold; margin-bottom: 0px; padding-bottom: 0px; }
    
    /* Metric styling */
    [data-testid="stMetricValue"] { font-size: 22px !important; color: #007bff; }
    
    /* HIGH-CONTRAST SYSTEM CHECK BUTTON */
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        height: 3.5em;
        width: 100%;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #0056b3;
        border: none;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Prediction card optimization */
    .prediction-card { 
        padding: 15px; 
        border-radius: 10px; 
        background-color: #ffffff; 
        border: 1px solid #dee2e6; 
        text-align: center;
        margin-top: 5px;
    }
    
    /* Sidebar width and spacing */
    section[data-testid="stSidebar"] { width: 320px !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Download and load the serialized SVM model pipeline from Hugging Face."""
    repo_id = "Sriranjan/Predictive_Maintenance_Model"
    path = hf_hub_download(repo_id=repo_id, filename="engine_pipeline.joblib")
    return joblib.load(path)

model = load_model()

# --- SIDEBAR: COMPACT CONTROLS ---
with st.sidebar:    
    # Engine_RPM input
    current_rpm = st.number_input("⚙️ Engine RPM", 0, 3000, 800, step=50, key="rpm_input")
    
    st.divider()

    # Temperature Sensors Group
    st.markdown("**🌡️ Temperature (°C)**")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        lub_t = st.number_input("Lub Oil", 60, 100, 77, key="t_l")
    with t_col2:
        cool_t = st.number_input("Coolant", 60, 100, 78, key="t_c")
        
    st.divider()

    # Pressure Sensors Group
    st.markdown("**🧪 Pressure (bar)**")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        fuel_p = st.number_input("Fuel P.", 0.0, 20.0, 6.6, step=0.1)
        cool_p = st.number_input("Coolant P.", 0.0, 10.0, 2.3, step=0.1)
    with p_col2:
        lub_p = st.number_input("Lub Oil P.", 0.0, 10.0, 3.3, step=0.1)

    st.caption("© 2026 Developed by Sriranjan Uppoor")

# --- MAIN DASHBOARD (Compact Layout) ---

# Unified Header Row
header_col, metrics_col = st.columns([1, 2])

with header_col:
    st.markdown('<p class="main-title">🏎️ Engine AI Diagnostic</p>', unsafe_allow_html=True)
    st.caption("High-Fidelity Predictive Maintenance")

with metrics_col:
    m1, m2, m3 = st.columns(3)
    m1.metric("Lubrication", f"{lub_p} bar", f"{lub_t}°C")
    m2.metric("Fuel System", f"{fuel_p} bar", "Stable")
    m3.metric("Cooling", f"{cool_p} bar", f"{cool_t}°C")

st.markdown("---")

# Analysis & Results Section (Split Screen)
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("📡 Data Preview")
    input_df = pd.DataFrame({
        'Engine rpm': [current_rpm],
        'Lub oil pressure': [lub_p],
        'Fuel pressure': [fuel_p],
        'Coolant pressure': [cool_p],
        'lub oil temp': [lub_t],
        'Coolant temp': [cool_t]
    })
    st.dataframe(input_df, hide_index=True, use_container_width=True)
    
    st.write("**Thermal Saturation Index**")
    st.progress(max(0, min((cool_t - 60) / 40, 1.0)))

with right_col:
    st.subheader("🧠 AI Verdict")
    if st.button("EXECUTE SYSTEM CHECK"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][pred] * 100
        
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        if pred == 0:
            st.balloons()
            st.markdown("<h2 style='color:#28a745; margin:0;'>✅ HEALTHY</h2>", unsafe_allow_html=True)
            st.write(f"Confidence: **{prob:.1f}%**")
        else:
            st.markdown("<h2 style='color:#dc3545; margin:0;'>⚠️ FAULT RISK</h2>", unsafe_allow_html=True)
            st.write(f"Probability: **{prob:.1f}%**")
            st.error("Action: Maintenance required.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("System ready. Click button to analyze engine condition.")
