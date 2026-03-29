import os
from huggingface_hub import HfApi

def register_artifacts():
    # 1. Securely load the token from the CI/CD environment
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not found! Check GitHub Secrets.")

    api = HfApi(token=token)
    repo_id = "Sriranjan/Predictive_Maintenance_Model"

    print("Initiating automated model registry...")

    # 2. Upload model
    api.upload_file(
        path_or_fileobj="Predictive_Maintenance/models/engine_pipeline.joblib", # Corrected filename
        path_in_repo="engine_model.joblib", # Keep the path in repo as engine_model.joblib for consistency with app.py
        repo_id=repo_id,
        repo_type="model"
    )

    # 3. Upload scaler
    api.upload_file(
        path_or_fileobj="Predictive_Maintenance/models/scaler.pkl",
        path_in_repo="scaler.pkl",
        repo_id=repo_id,
        repo_type="model"
    )

    print("Model & scaler successfully registered to Hugging Face")

if __name__ == "__main__":
    register_artifacts()
