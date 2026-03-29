
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import joblib

from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def train_model():

    print("\n===== LOADING TRAIN & TEST FROM HUGGING FACE =====")

    dataset = load_dataset("Sriranjan/Predictive_Maintenance_Data")

    # Ensure splits exist
    if 'train' not in dataset or 'test' not in dataset:
        raise ValueError("Train/Test split not found in Hugging Face dataset")

    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()

    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)

    # ------------------------------
    # FEATURE / TARGET SPLIT
    # ------------------------------
    X_train = train_df.drop('Engine Condition', axis=1)
    y_train = train_df['Engine Condition']

    X_test = test_df.drop('Engine Condition', axis=1)
    y_test = test_df['Engine Condition']

    # ------------------------------
    # SCALING
    # ------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------
    # 7 MODELS
    # ------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }

    results = []

    mlflow.set_experiment("Predictive Maintenance")

    best_model = None
    best_recall = 0
    best_model_name = ""

    print("\n===== TRAINING 7 MODELS =====")

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            # Scaling required models
            if name in ["Logistic Regression", "SVM", "KNN"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"\n{name}")
            print(f"Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, name)

            results.append([name, acc, prec, rec, f1])

            if rec > best_recall:
                best_recall = rec
                best_model = model
                best_model_name = name

    # ------------------------------
    # MODEL COMPARISON
    # ------------------------------
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
    results_df = results_df.sort_values(by="Recall", ascending=False)

    print("\n===== MODEL COMPARISON =====")
    print(results_df)

    print("\n===== BEST BASE MODEL =====")
    print(f"{best_model_name} (Recall: {best_recall:.4f})")

    # ------------------------------
    # HYPERPARAMETER TUNING
    # ------------------------------
    print("\n===== HYPERPARAMETER TUNING =====")

    tuned_models = {}

    # RANDOM FOREST
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10]
    }

    rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, scoring='recall', cv=3)
    rf_grid.fit(X_train, y_train)
    tuned_models["Random Forest"] = rf_grid.best_estimator_

    with mlflow.start_run(run_name="RF_Tuned"):
        mlflow.log_params(rf_grid.best_params_)
        mlflow.log_metric("best_recall", rf_grid.best_score_)

    # GRADIENT BOOSTING
    gb_params = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1]
    }

    gb_grid = GridSearchCV(GradientBoostingClassifier(), gb_params, scoring='recall', cv=3)
    gb_grid.fit(X_train, y_train)
    tuned_models["Gradient Boosting"] = gb_grid.best_estimator_

    with mlflow.start_run(run_name="GB_Tuned"):
        mlflow.log_params(gb_grid.best_params_)
        mlflow.log_metric("best_recall", gb_grid.best_score_)

    # SVM
    svm_params = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }

    svm_grid = GridSearchCV(SVC(probability=True), svm_params, scoring='recall', cv=3)
    svm_grid.fit(X_train_scaled, y_train)
    tuned_models["SVM"] = svm_grid.best_estimator_

    with mlflow.start_run(run_name="SVM_Tuned"):
        mlflow.log_params(svm_grid.best_params_)
        mlflow.log_metric("best_recall", svm_grid.best_score_)

    # ------------------------------
    # SELECT BEST TUNED MODEL
    # ------------------------------
    print("\n===== SELECTING BEST TUNED MODEL =====")

    best_tuned_model = None
    best_tuned_recall = 0
    best_tuned_name = ""

    for name, model in tuned_models.items():

        if name == "SVM":
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)

        rec = recall_score(y_test, y_pred)

        print(f"{name} Tuned Recall: {rec:.4f}")

        if rec > best_tuned_recall:
            best_tuned_recall = rec
            best_tuned_model = model
            best_tuned_name = name

    print("\n===== FINAL BEST MODEL =====")
    print(f"{best_tuned_name} (Recall: {best_tuned_recall:.4f})")

    # ------------------------------
    # SAVE FINAL MODEL
    # ------------------------------
    os.makedirs("Predictive_Maintenance/models", exist_ok=True)

    joblib.dump(best_tuned_model, "Predictive_Maintenance/models/best_model.pkl")
    joblib.dump(scaler, "Predictive_Maintenance/models/scaler.pkl")

    print("\nFinal model saved for deployment.")


if __name__ == "__main__":
    train_model()
