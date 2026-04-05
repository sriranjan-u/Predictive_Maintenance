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

    # 2. Upload the unified Pipeline (contains BOTH the Scaler and the Model)
    print("Uploading unified engine pipeline...")
    api.upload_file(
        path_or_fileobj="models/engine_pipeline.joblib",
        path_in_repo="engine_pipeline.joblib",
        repo_id=repo_id,
        repo_type="model"
    )

    # 3. Upload the Model Comparison Report (Great for tracking performance over time)
    report_path = "reports/model_comparison.csv"
    if os.path.exists(report_path):
        print("Uploading performance comparison report...")
        api.upload_file(
            path_or_fileobj=report_path,
            path_in_repo="reports/model_comparison.csv",
            repo_id=repo_id,
            repo_type="model"
        )

    print("Unified Pipeline and artifacts successfully registered to Hugging Face!")

if __name__ == "__main__":
    register_artifacts()
