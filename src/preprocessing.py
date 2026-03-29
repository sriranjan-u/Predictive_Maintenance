import pandas as pd
from sklearn.model_selection import train_test_split
import os
from huggingface_hub import HfApi

def clean_and_register_data(repo_id):
    # Load from Hub
    url = f"https://huggingface.co/datasets/{repo_id}/raw/main/engine_data.csv"
    df = pd.read_csv(url)

    # IQR Capping (Data Cleaning)
    sensor_cols = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
    for col in sensor_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

    # Train-Test Split
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Engine Condition'])

    # Save Locally
    os.makedirs('Predictive_Maintenance/data', exist_ok=True)
    train_path = 'Predictive_Maintenance/data/train.csv'
    test_path = 'Predictive_Maintenance/data/test.csv'
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print("Local split complete.")

    # Upload back to HF Dataset Space (Rubric Requirement)
    token = os.getenv("HF_TOKEN")
    if token:
        api = HfApi(token=token)
        # Upload Train
        api.upload_file(
            path_or_fileobj=train_path,
            path_in_repo="train.csv",
            repo_id=repo_id,
            repo_type="dataset"
        )
        # Upload Test
        api.upload_file(
            path_or_fileobj=test_path,
            path_in_repo="test.csv",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"Train and Test sets uploaded to HF Dataset: {repo_id}")
    else:
        print("HF_TOKEN not found. Cloud upload skipped.")

if __name__ == "__main__":
    REPO_ID = "Sriranjan/Predictive_Maintenance_Data"
    clean_and_register_data(REPO_ID)
