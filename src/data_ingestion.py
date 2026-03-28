from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
import pandas as pd

# Define the new project details
repo_id = "Sriranjan/Predictive_Maintenance_Data"
repo_type = "dataset"

# Get token from environment or direct input
# Note: In Colab, we will pass this via os.environ in the next step
token = os.getenv("HF_TOKEN")

print("Initializing Hugging Face API Client...")
api = HfApi(token=token)

# Step 1: Ensure the Dataset Repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Repository '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
    print(f"Repository '{repo_id}' created successfully.")

# Step 2: Upload the data folder
print(f"Uploading data from Predictive_Maintenance/data to {repo_id}...")
api.upload_folder(
    folder_path="Predictive_Maintenance/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Data Registration Complete.")
