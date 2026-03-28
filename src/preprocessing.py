import pandas as pd
from sklearn.model_selection import train_test_split
import os

def clean_data(repo_id):
    # 1. Load from Hub
    url = f"https://huggingface.co/datasets/{repo_id}/raw/main/engine_data.csv"
    df = pd.read_csv(url)
    
    # 2. Exact Column Names from your dataset
    # Note: 'lub oil temp' is lowercase as per your output
    sensor_cols = [
        'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 
        'lub oil temp', 'Coolant temp'
    ]
    
    print(f"Applying IQR Capping to: {sensor_cols}")
    
    for col in sensor_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Capping outliers (Standard MLOps practice)
            df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    
    # 3. Train-Test Split (Target is 'Engine Condition')
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Engine Condition']
    )
    
    # 4. Save locally
    os.makedirs('Predictive_Maintenance/data', exist_ok=True)
    train.to_csv('Predictive_Maintenance/data/train.csv', index=False)
    test.to_csv('Predictive_Maintenance/data/test.csv', index=False)
    print("Preprocessing Complete: Casing fixed and files generated.")

if __name__ == "__main__":
    REPO_ID = "Sriranjan/Predictive_Maintenance_Data"
    clean_data(REPO_ID)
