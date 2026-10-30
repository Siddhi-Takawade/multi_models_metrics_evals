from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
MODEL_DIR = Path("model")
TEST_DATA_PATH = Path("test_data.csv")
METRICS_PATH = Path("metrics_summary.csv")


def build_models() -> Dict[str, Any]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
                ),
            ]
        ),
    }


def get_positive_class_score(model: Any, x: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(x_test)
    y_score = get_positive_class_score(model, x_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    test_df = x_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(TEST_DATA_PATH, index=False)

    models = build_models()
    rows = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)

        safe_name = model_name.lower().replace(" ", "_")
        model_path = MODEL_DIR / f"{safe_name}.joblib"
        joblib.dump(model, model_path)

        metrics = evaluate_model(model, x_test, y_test)
        row = {"ML Model Name": model_name, **metrics}
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("Training complete.")
    print(f"Saved test data to: {TEST_DATA_PATH.resolve()}")
    print(f"Saved metrics to: {METRICS_PATH.resolve()}")
    print(f"Saved models to: {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
