import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

from config import POSITIVE_CLASS_ALIASES


def to_positive_class_binary(values):
    return pd.Series(values).apply(
        lambda value: int(
            value in POSITIVE_CLASS_ALIASES
            or str(value).strip().lower() in POSITIVE_CLASS_ALIASES
        )
    )


def calculate_auc(y_true, y_pred, positive_probabilities):
    y_true_binary = to_positive_class_binary(y_true)

    if positive_probabilities is not None:
        return roc_auc_score(y_true_binary, positive_probabilities)

    y_pred_binary = to_positive_class_binary(y_pred)
    return roc_auc_score(y_true_binary, y_pred_binary)


def build_evaluation(y_true, y_pred, positive_probabilities):
    y_true_binary = to_positive_class_binary(y_true)
    y_pred_binary = to_positive_class_binary(y_pred)

    try:
        auc_value = calculate_auc(y_true_binary, y_pred_binary, positive_probabilities)
    except ValueError:
        auc_value = None

    try:
        if positive_probabilities is not None:
            fpr, tpr, _ = roc_curve(y_true_binary, positive_probabilities)
        else:
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
    except ValueError:
        fpr, tpr = None, None

    labels = [0, 1]
    matrix = confusion_matrix(y_true_binary, y_pred_binary, labels=labels)
    tn, fp, fn, tp = matrix.ravel()
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual 0 (No)", "Actual 1 (Yes)"],
        columns=["Predicted 0 (No)", "Predicted 1 (Yes)"]
    )

    return {
        "accuracy": accuracy_score(y_true_binary, y_pred_binary),
        "auc": auc_value,
        "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
        "mcc": matthews_corrcoef(y_true_binary, y_pred_binary),
        "confusion_matrix": matrix_df,
        "fpr": fpr,
        "tpr": tpr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
