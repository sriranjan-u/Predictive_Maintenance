import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# 1. Page Config (MUST be the very first Streamlit command)
st.set_page_config(page_title="Engine Health Monitor", layout="wide")

st.title("Car Predictive Maintenance Decision Support System")
st.markdown("---")

# 2. Pure Cached Function (Strictly no UI elements inside)
@st.cache_resource
def load_model():
    repo_id = "Sriranjan/Predictive_Maintenance_Model"
    
    # Explicitly setting repo_type to "dataset" based on your CI/CD architecture
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="engine_pipeline.joblib",
        repo_type="dataset" 
    )
    return joblib.load(model_path)

# 3. Handle loading UI outside the cache
with st.spinner("Downloading and configuring model from Hugging Face..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model loading failed. Please verify the repo_id, filename, and repo_type.\nError details: {e}")
        st.stop()

if model is None:
    st.stop()

# 4. SIDEBAR INPUTS
st.sidebar.header("Sensor Inputs")

rpm = st.sidebar.slider("Engine RPM", 400, 1500, 800)
lub_oil_pres = st.sidebar.number_input("Lub Oil Pressure", 0.0, 10.0, 3.5)
fuel_pres = st.sidebar.number_input("Fuel Pressure", 0.0, 25.0, 10.0)
coolant_pres = st.sidebar.number_input("Coolant Pressure", 0.0, 10.0, 3.0)
lub_oil_temp = st.sidebar.slider("Lub Oil Temp (°C)", 60, 100, 75)
coolant_temp = st.sidebar.slider("Coolant Temp (°C)", 60, 100, 80)

# 5. INPUT DATA (Exact feature names verified)
input_data = pd.DataFrame(
    [[rpm, lub_oil_pres, fuel_pres, coolant_pres, lub_oil_temp, coolant_temp]],
    columns=[
        'Engine rpm',
        'Lub oil pressure',
        'Fuel pressure',
        'Coolant pressure',
        'lub oil temp',
        'Coolant temp'
    ]
)

# 6. PREDICTION LOGIC
try:
    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
    else:
        probability = None

except Exception as e:
    st.error(f"Prediction failed due to data mismatch: {e}")
    st.stop()

# 7. DISPLAY
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sensor Data")
    st.dataframe(input_data)

with col2:
    st.subheader("Prediction")

    if prediction == 1:
        st.error(" !!! FAULT DETECTED")
        if probability is not None:
            st.write(f"**Probability:** {probability:.2%}")
    else:
        st.success("ENGINE HEALTHY :) ")
        if probability is not None:
            st.write(f"**Confidence:** {(1 - probability):.2%}")
