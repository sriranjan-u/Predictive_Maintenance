import pandas as pd
import os
import mlflow
import mlflow.sklearn
import joblib

from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train_model():
    print("\n===== LOADING TRAIN & TEST FROM HUGGING FACE =====")

    dataset = load_dataset("Sriranjan/Predictive_Maintenance_Data")

    if 'train' not in dataset or 'test' not in dataset:
        raise ValueError("Train/Test split not found in Hugging Face dataset")

    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()

    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)

    # Feature / Target split
    X_train = train_df.drop('Engine Condition', axis=1)
    y_train = train_df['Engine Condition']

    X_test = test_df.drop('Engine Condition', axis=1)
    y_test = test_df['Engine Condition']

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
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
    tuned_results = []

    mlflow.set_experiment("Predictive Maintenance")

    best_model = None
    best_recall = 0
    best_model_name = ""

    print("\n===== TRAINING MODELS =====")

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

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

            mlflow.sklearn.log_model(model, name=name)

            results.append([name, acc, prec, rec, f1])

            if rec > best_recall:
                best_recall = rec
                best_model = model
                best_model_name = name

    # Baseline comparison
    baseline_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
    baseline_df = baseline_df.sort_values(by="Recall", ascending=False)

    print("\n===== BASELINE MODEL COMPARISON =====")
    print(baseline_df)

    print("\n===== BEST BASE MODEL =====")
    print(f"{best_model_name} (Recall: {best_recall:.4f})")

    # Hyperparameter tuning
    print("\n===== HYPERPARAMETER TUNING =====")

    tuned_models = {}

    rf_grid = GridSearchCV(
        RandomForestClassifier(),
        {"n_estimators": [100, 200], "max_depth": [5, 10]},
        scoring='recall',
        cv=3
    )
    rf_grid.fit(X_train, y_train)
    tuned_models["Random Forest"] = rf_grid.best_estimator_

    gb_grid = GridSearchCV(
        GradientBoostingClassifier(),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        scoring='recall',
        cv=3
    )
    gb_grid.fit(X_train, y_train)
    tuned_models["Gradient Boosting"] = gb_grid.best_estimator_

    svm_grid = GridSearchCV(
        SVC(probability=True),
        {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        scoring='recall',
        cv=3
    )
    svm_grid.fit(X_train_scaled, y_train)
    tuned_models["SVM"] = svm_grid.best_estimator_

    # Evaluate tuned models
    print("\n===== EVALUATING TUNED MODELS =====")

    best_tuned_model = None
    best_tuned_recall = 0
    best_tuned_name = ""

    for name, model in tuned_models.items():

        if name == "SVM":
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"{name} Tuned \u2192 Recall: {rec:.4f}, F1: {f1:.4f}")

        tuned_results.append([name, acc, prec, rec, f1])

        if rec > best_tuned_recall:
            best_tuned_recall = rec
            best_tuned_model = model
            best_tuned_name = name

    print("\n===== FINAL BEST MODEL =====")
    print(f"{best_tuned_name} (Recall: {best_tuned_recall:.4f})")

    # Performance comparison
    tuned_df = pd.DataFrame(tuned_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

    baseline_df["Type"] = "Baseline"
    tuned_df["Type"] = "Tuned"

    comparison_df = pd.concat([baseline_df, tuned_df])

    print("\n===== PERFORMANCE COMPARISON =====")
    print(comparison_df.sort_values(by=["Model", "Type"]))

    # Save comparison
    os.makedirs("Predictive_Maintenance/reports", exist_ok=True)
    comparison_path = "Predictive_Maintenance/reports/model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    mlflow.log_artifact(comparison_path)

    print(f"\nComparison report saved at: {comparison_path}")

    # Save pipeline
    print("\n===== SAVING FINAL PIPELINE =====")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", best_tuned_model)
    ])

    pipeline.fit(X_train, y_train)

    os.makedirs("Predictive_Maintenance/models", exist_ok=True)
    model_path = "Predictive_Maintenance/models/engine_pipeline.joblib"

    joblib.dump(pipeline, model_path)
    mlflow.log_artifact(model_path)

    print(f"\nModel pipeline saved at: {model_path}")

if __name__ == "__main__":
    train_model()
