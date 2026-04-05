from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
import pandas as pd

# Define the new project details
repo_id = "Sriranjan/Predictive_Maintenance_Data"
repo_type = "dataset"

# Get token from environment
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable not found! Check GitHub Secrets.")

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

# Step 2: Upload the data folder (FIXED PATH)
print(f"Uploading data from local 'data/' directory to {repo_id}...")
api.upload_folder(
    folder_path="data", # <-- This is the crucial path fix!
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Data Registration Complete.")
