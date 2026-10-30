# Machine Learning Assignment 2 - Classification and Streamlit Deployment

## a. Problem statement
Build and compare five machine learning classification models on one dataset, evaluate them with standard metrics, and deploy the results through an interactive Streamlit web application.

## b. Dataset description
- Dataset Name: Breast Cancer Wisconsin (Diagnostic)
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic); loaded locally from scikit-learn's bundled copy
- Task Type: Binary classification
- Instances: 569
- Features: 30 numerical input features
- Target: `target` (0 = malignant, 1 = benign)

This dataset satisfies assignment constraints:
- Minimum features required: 12 (available: 30)
- Minimum instances required: 500 (available: 569)

## c. Github Repository Link
- Add your repository link here after pushing code:
- Example: https://github.com/<your-username>/<your-repo-name>

## d. Models used
Implemented models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Naive Bayes Classifier (GaussianNB)
5. Support Vector Machine (SVM)

### Comparison Table (Evaluation Metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

Metrics use the stratified 20% holdout split with `random_state=42`. For binary metrics, class `1` (benign) is treated as the positive class. Values are rounded to four decimal places from `metrics_summary.csv`.

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Tied for top accuracy (0.9825) and achieved the highest AUC (0.9954), with balanced precision and recall. Scaling is included in its pipeline. |
| Decision Tree | Lowest holdout accuracy (0.9123) and MCC (0.8174) among these runs, though precision remained high (0.9559). This single tree may not generalize as consistently as the scaled models. |
| KNN | Strong accuracy (0.9737) and the highest recall (1.0000), meaning it identified every positive-class example in this split; its precision was lower than the top two models. |
| Naive Bayes | AUC was strong (0.9878), but accuracy (0.9386) and MCC (0.8676) were below Logistic Regression, SVM, and KNN on this split. |
| Support Vector Machine | Tied for top accuracy (0.9825), with AUC 0.9950 and balanced precision/recall. It performed nearly identically to Logistic Regression on this holdout. |
| Overall, Winner for your dataset? | Logistic Regression is the winner by holdout AUC (0.9954) and is tied for best accuracy, F1, and MCC. These results reflect one fixed split and should not be treated as a guarantee of future performance. |

## Project Structure

```text
ml_assignment_2/
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

## How to run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train and save models + generate test data:
   ```bash
   python train_models.py
   ```

3. Run Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Streamlit app features (as required)
- CSV upload option for test data
- Uploaded-CSV comparison table for all five saved models
- Model selection dropdown for detailed results
- Evaluation metrics, confusion matrix, and classification report for the selected model

## Deployment on Streamlit Community Cloud
1. Push this project to GitHub.
2. Open https://streamlit.io/cloud and sign in.
3. Click New App and select your repository.
4. Select branch and `app.py`.
5. Deploy and copy the live app URL.

## Notes for submission PDF
Include in your final PDF:
1. GitHub repository link
2. Live Streamlit app link
3. Screenshot proving execution on BITS Virtual Lab
4. Full README content

Before submission, replace the repository and live-app link placeholders after publishing your own GitHub repository and deploying it. Run the project in BITS Virtual Lab yourself and insert your genuine execution screenshot; this workspace cannot provide proof of a session in that external lab.
