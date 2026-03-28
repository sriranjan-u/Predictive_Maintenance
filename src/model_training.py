import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from huggingface_hub import HfApi

def train_model():
    # 1. Load the processed data
    train = pd.read_csv('Predictive_Maintenance/data/train.csv')
    test = pd.read_csv('Predictive_Maintenance/data/test.csv')
    
    # Identify target column (handling potential casing differences)
    target = 'Engine Condition' if 'Engine Condition' in train.columns else 'Engine_Condition'
    
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    print(f"Training XGBoost Model on {len(X_train)} samples...")

    # 2. Initialize and Train Model
    # Using parameters optimized for tabular sensor data
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # 3. Evaluate
    preds = model.predict(X_test)
    print("\nModel Performance Report:")
    print(classification_report(y_test, preds))

    # 4. Save Locally
    os.makedirs('Predictive_Maintenance/models', exist_ok=True)
    model_path = 'Predictive_Maintenance/models/engine_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # 5. Push Model to Hugging Face Hub (8 Points)
    token = os.getenv("HF_TOKEN")
    if token:
        api = HfApi(token=token)
        user_info = api.whoami()
        repo_id = f"{user_info['name']}/Predictive_Maintenance_Model"
        
        print(f"Pushing model to Hub: {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="engine_model.joblib",
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Model registered successfully at: https://huggingface.co/{repo_id}")
    else:
        print("HF_TOKEN not found. Skipping Hub registration.")

if __name__ == "__main__":
    train_model()
