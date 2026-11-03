# Breast Cancer Classification Model Evaluation

This project trains and compares five binary classification models using the Breast Cancer Wisconsin Diagnostic dataset. A Streamlit app evaluates the saved models against an uploaded, labeled CSV file.

## a. Problem Statement

The goal is to use the dataset's diagnostic measurements to classify each case as malignant or benign. Five classification algorithms are trained on the same training split and compared using standard classification metrics. This project is for coursework and is not intended for medical diagnosis or clinical decision-making.

## b. Dataset Description

| Item | Description |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic) |
| Source | Scikit-learn's bundled `load_breast_cancer` dataset; originally from the UCI Machine Learning Repository |
| Observations | 569 |
| Input features | 30 numerical diagnostic measurements |
| Target | `target`: 0 = malignant, 1 = benign |
| Task | Binary classification |

The data is split into 80% training and 20% test data using stratification and `random_state=42`. Standard scaling is used inside the Logistic Regression, KNN, and SVM pipelines. Decision Tree and Gaussian Naive Bayes are trained without scaling.

## c. GitHub Repository

[Siddhi-Takawade/multi_models_metrics_evals](https://github.com/Siddhi-Takawade/multi_models_metrics_evals)

Live Streamlit app: add the app URL after deployment has been verified.

## d. Models and Evaluation

The following models are evaluated on the same held-out test split:

1. Logistic Regression with StandardScaler
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN, 7 neighbors) with StandardScaler
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel, probability scores enabled) with StandardScaler

### Holdout Results

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

For all binary metrics, class 1 (benign) is treated as the positive class. AUC uses each model's positive-class probability or decision score. Values are rounded to four decimal places from `metrics_summary.csv`.

Logistic Regression has the highest AUC (0.9954) and ties the SVM for the highest accuracy (0.9825), F1, and MCC. KNN has the highest recall (1.0000) on this split. The Decision Tree has the lowest accuracy and MCC among these runs. These results are from one fixed holdout split and do not establish how the models will perform on other data.

## Streamlit App

The app accepts a labeled CSV containing the same 30 input feature columns and a `target` column. It compares all five saved models and displays the selected model's metrics, confusion matrix, and classification report.

## Project Structure

```text
multi_models_metrics_evals/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- metrics_summary.csv
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- support_vector_machine.joblib
```

## Run Locally

Install the dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads the saved files from `model/`. To reproduce the training artifacts locally, run `python train_models.py`; this writes the model files, `test_data.csv`, and `metrics_summary.csv` beside the scripts.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub, including all five `.joblib` files in `model/`.
2. In Streamlit Community Cloud, create an app from this repository, select branch `main`, and set the app file to `app.py`.
3. Deploy the app. For an existing deployment, reboot it after the required files are pushed.

Streamlit Cloud installs the packages in `requirements.txt`; it does not train the models. The deployed app loads the committed model files and does not need to retrain them at startup. The `model/` directory must be present in the selected branch beside `app.py`.

## References

- [UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [Scikit-learn: Breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Streamlit Community Cloud](https://share.streamlit.io/)