# Bank Marketing Classification

A machine learning classification project using the **UCI Bank Marketing dataset** to predict whether a customer will subscribe to a term deposit.

## Project Overview

This project implements and compares five supervised machine learning classification models:

* Logistic Regression
* Decision Tree
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Random Forest

The models are trained and evaluated using the Bank Marketing dataset. Model performance is compared using multiple classification metrics, and the trained models are saved for later use in the Streamlit application.

## Dataset

The project uses the **Bank Marketing dataset** from the UCI Machine Learning Repository.

The dataset is retrieved programmatically using `ucimlrepo`, rather than storing the complete dataset in this repository.

## Models

The following models are evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Naive Bayes
5. Random Forest

## Evaluation Metrics

The models are compared using:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

## Project Structure

```text
bank-marketing-classification/
│
├── app.py
├── requirements.txt
├── test_data.csv
├── README.md
│
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── random_forest.joblib
│
└── notebooks/
    └── bank_marketing.ipynb
```

## Current Status

* [x] Dataset selection and retrieval
* [ ] Data exploration
* [ ] Train/test split
* [ ] Preprocessing pipeline
* [ ] Train five classification models
* [ ] Evaluate model performance
* [ ] Save trained models
* [ ] Streamlit application
* [ ] Deployment
* [ ] Final documentation

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```


