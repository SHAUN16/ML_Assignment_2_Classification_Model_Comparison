# Bank Marketing Classification

## Problem Statement

This project predicts whether a bank customer will subscribe to a term deposit using supervised machine learning classification models. The goal is to compare multiple classification algorithms on the same dataset and identify the best-performing model based on standard evaluation metrics.

## Dataset Description

The project uses the Bank Marketing dataset from the UCI Machine Learning Repository. The data is related to direct marketing campaigns of a Portuguese banking institution, where customers were contacted by phone.

The target variable is `y`, which indicates whether the customer subscribed to a term deposit:

| Target Value | Meaning |
|---|---|
| 0 | Customer did not subscribe |
| 1 | Customer subscribed |

The dataset contains bank client information, campaign details, previous campaign outcomes, and the final subscription result. This project uses 16 input features and 1 output target column, satisfying the assignment requirement of at least 12 features and at least 500 instances.

## GitHub Repository Link

https://github.com/SHAUN16/ML_Assignment_2_Classification_Model_Comparison

## Live Streamlit App Link

https://shaun16-ml-assignment-2-classification-model-compari-app-waecyl.streamlit.app

## Models Used

The following five classification models were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier
5. Random Forest Classifier

## Model Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | 0.4701 | 0.3990 |
| KNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest | 0.9045 | 0.9263 | 0.6506 | 0.3960 | 0.4924 | 0.4597 |

## Best Model by Metric

| Metric | Best Model | Score |
|---|---|---:|
| Accuracy | Random Forest | 0.9045 |
| AUC | Random Forest | 0.9263 |
| Precision | Random Forest | 0.6506 |
| Recall | Naive Bayes | 0.5198 |
| F1 | Random Forest | 0.4924 |
| MCC | Random Forest | 0.4597 |

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved high accuracy and AUC, showing strong overall discrimination, but recall was low, meaning it missed many positive subscription cases. |
| Decision Tree | Produced more balanced precision and recall than some models, but its AUC and overall accuracy were lower than Random Forest and Logistic Regression. |
| KNN | Performed reasonably well in accuracy and precision, but recall was low, so it was less effective at identifying customers who subscribed. |
| Naive Bayes | Achieved the highest recall, making it better at detecting positive subscription cases, but it had the lowest precision and accuracy among the models. |
| Random Forest | Delivered the best accuracy, AUC, precision, F1 score, and MCC, making it the most consistent model overall for this dataset. |
| Overall Winner for this dataset | Random Forest is the overall winner because it performed best on five out of six evaluation metrics and had the strongest balance of predictive performance. |

## Project Structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- config.py
|-- data_utils.py
|-- evaluation_utils.py
|-- model_utils.py
|-- state_utils.py
|-- ui_components.py
|-- models/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- random_forest.joblib
|-- notebooks/
|   |-- bank_marketing.ipynb
```

## Streamlit Application Features

The Streamlit application includes:

- CSV dataset upload option
- Sample test dataset loading
- Model selection dropdown
- Prediction output display
- Evaluation metrics
- Confusion matrix
- Model comparison table

## Run Locally

The repository includes helper scripts that pull Git LFS model files, create a local `.venv` virtual environment, install dependencies, and start the Streamlit app.

On Linux or BITS VM, run:

```bash
bash run_app.sh
```

On Windows PowerShell, run:

```powershell
.\run_app.ps1
```

If PowerShell blocks script execution, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

## Manual Setup

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Run the Streamlit app locally using:

```bash
streamlit run app.py
```

If the app shows model loading errors after cloning, make sure Git LFS is installed and run:

```bash
git lfs install
git lfs pull
```
