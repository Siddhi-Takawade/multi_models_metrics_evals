from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


st.set_page_config(page_title="ML Assignment 2 - Classifier Evaluation", layout="wide")

MODEL_DIR = Path("model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def load_model(model_name: str) -> Any:
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run train_models.py first."
        )
    return joblib.load(model_path)


def get_positive_class_score(model: Any, x: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def compute_metrics(model: Any, x: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(x)
    y_score = get_positive_class_score(model, x)

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def render_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def main() -> None:
    st.title("Machine Learning Assignment 2")
    st.subheader("Classification Model Evaluation App")

    st.markdown(
        "Upload a CSV with all input features and a `target` column. "
        "The app loads saved models and evaluates on uploaded test data."
    )

    uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to compare all five saved models.")
        return

    df = pd.read_csv(uploaded_file)
    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
        return

    y = df["target"]
    x = df.drop(columns=["target"])

    try:
        comparison_rows = []
        models = {
            model_name: load_model(model_name)
            for model_name in MODEL_FILES
        }
        expected_features = list(models[next(iter(models))].feature_names_in_)
        missing_features = [name for name in expected_features if name not in x.columns]
        extra_features = [name for name in x.columns if name not in expected_features]
        if missing_features or extra_features:
            details = []
            if missing_features:
                details.append(f"missing: {', '.join(missing_features)}")
            if extra_features:
                details.append(f"unexpected: {', '.join(extra_features)}")
            st.error("CSV feature columns do not match the trained models (" + "; ".join(details) + ").")
            return

        x = x[expected_features]
        predictions = {}
        for model_name, model in models.items():
            model_metrics = compute_metrics(model, x, y)
            predictions[model_name] = model.predict(x)
            comparison_rows.append({"ML Model Name": model_name, **model_metrics})

        comparison_df = pd.DataFrame(comparison_rows)
        st.markdown("### Model comparison on uploaded CSV")
        st.dataframe(comparison_df, hide_index=True, width="stretch")

        selected_model_name = st.selectbox("Select model for detailed results", list(MODEL_FILES.keys()))
        metrics = comparison_df.set_index("ML Model Name").loc[selected_model_name].to_dict()
        y_pred = predictions[selected_model_name]
    except Exception as exc:
        st.error(f"Failed to evaluate model. Details: {exc}")
        return

    st.markdown(f"### Evaluation metrics for {selected_model_name}")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    metric_cols[1].metric("AUC", f"{metrics['AUC']:.4f}")
    metric_cols[2].metric("Precision", f"{metrics['Precision']:.4f}")
    metric_cols[3].metric("Recall", f"{metrics['Recall']:.4f}")
    metric_cols[4].metric("F1", f"{metrics['F1']:.4f}")
    metric_cols[5].metric("MCC", f"{metrics['MCC']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        render_confusion_matrix(y, y_pred)

    with col2:
        report = classification_report(y, y_pred)
        st.markdown("#### Classification Report")
        st.code(report)


if __name__ == "__main__":
    main()
