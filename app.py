import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vehicle Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD ASSETS (Image & Model) ---
def load_assets():
    """Download and load the model pipeline from Hugging Face."""
    try:
        # Download the unified pipeline
        repo_id = "Sriranjan/Predictive_Maintenance_Model"
        model_path = hf_hub_download(repo_id=repo_id, filename="engine_pipeline.joblib")
        
        # 2. Load the pipeline
        pipeline = joblib.load(model_path)
        return pipeline, None
    except Exception as e:
        return None, f"Failed to load assets: {e}"

pipeline, error = load_assets()

# --- HEADER SECTION ---
st.title("🚗 AI-Powered Vehicle Predictive Maintenance Dashboard")

st.markdown("---")

st.markdown("""
    This dashboard analyzes real-time engine sensor data to predict the **probability of failure**.
    By identifying potential faults before they occur, you can shift from reactive to proactive maintenance.
""")

st.markdown("---")

if error:
    st.error(error)
    st.stop()

# --- SIDEBAR INPUTS (Keep Main View Clean) ---
st.sidebar.header("🔧 Enter Real-Time Sensor Readings")
st.sidebar.markdown("Use the sliders below to provide current operational data.")

def user_input_features():
    """Capture sensor inputs via the sidebar."""
    
    # Pressure Sensors Group
    st.sidebar.subheader("Pressure (PSI)")
    fuel_p = st.sidebar.slider("Fuel Pressure", 0.0, 100.0, 41.0, step=0.1, help="Expected range: 30-50 PSI")
    lub_p = st.sidebar.slider("Lub Oil Pressure", 0.0, 100.0, 4.3, step=0.1, help="Expected range: 3-6 PSI")
    coolant_p = st.sidebar.slider("Coolant Pressure", 0.0, 100.0, 1.8, step=0.1, help="Expected range: 1-3 PSI")
    
    # Temperature Sensors Group
    st.sidebar.subheader("Temperature (°F)")
    lub_t = st.sidebar.slider("Lub Oil Temp", 0.0, 200.0, 81.0, step=1.0, help="Expected range: 75-90 °F")
    coolant_t = st.sidebar.slider("Coolant Temp", 0.0, 200.0, 83.0, step=1.0, help="Expected range: 75-90 °F")
    
    # Operational Group
    st.sidebar.subheader("Operational")
    rpm = st.sidebar.slider("Engine RPM", 0, 3000, 1200, step=10, help="Normal idle: 800-1500 RPM")

    # Save inputs into a Dataframe
    data = {
        'Engine rpm': rpm,
        'Lub oil pressure': lub_p,
        'Fuel pressure': fuel_p,
        'Coolant pressure': coolant_p,
        'lub oil temp': lub_t,
        'Coolant temp': coolant_t
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- MAIN VIEW: Display Inputs & Results ---

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Captured Sensor Data Summary")
    
    # UX Feature: Use Metrics containers for better visual representation
    m_col1, m_col2, m_col3 = st.columns(3)
    
    m_col1.metric("Engine RPM", int(input_df['Engine rpm'][0]), "RPM", help="Rotations per Minute")
    m_col2.metric("Lub Oil Pressure", f"{input_df['Lub oil pressure'][0]:.1f}", "PSI")
    m_col3.metric("Fuel Pressure", f"{input_df['Fuel pressure'][0]:.1f}", "PSI")
    
    m_col4, m_col5 = st.columns(2)
    m_col4.metric("Lub Oil Temp", f"{input_df['lub oil temp'][0]:.0f}°F", help="Normal range: 75-90°F")
    m_col5.metric("Coolant Temp", f"{input_df['Coolant temp'][0]:.0f}°F", help="Normal range: 75-90°F")

with col2:
    st.subheader("🚀 AI Prediction Engine")
    predict_btn = st.button("Analyze Engine Health")
    
    if predict_btn:
        with st.spinner('Analyzing sensor patterns...'):
            try:
                # Run prediction
                prediction = pipeline.predict(input_df)
                prediction_proba = pipeline.predict_proba(input_df)
                
                confidence = prediction_proba[0][prediction[0]] * 100
                
                st.markdown("---")
                
                # UX Feature: Use Prominent Status boxes for the final verdict
                if prediction[0] == 0:
                    st.success(f"### Engine Status: ACTIVE/HEALTHY")
                    st.write(f"Confidence: **{confidence:.1f}%** (This reading shows a **low risk** of imminent failure.)")
                else:
                    st.error(f"### Engine Status: FAULTY/RISK")
                    st.write(f"Confidence: **{confidence:.1f}%** (This pattern indicates a **high probability** of failure. Maintenance is **required**.)")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("Vehicle Predictive Maintenance Project - By Sriranjan Uppoor")
