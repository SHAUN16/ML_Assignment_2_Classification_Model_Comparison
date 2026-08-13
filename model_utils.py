import joblib
import streamlit as st
from sklearn.compose import _column_transformer

from config import (
    MODEL_FILES,
    MODELS_DIR,
    POSITIVE_CLASS_ALIASES,
    POSITIVE_CLASS_LABEL,
    TARGET_COLUMN,
)
from data_utils import split_features_and_target
from evaluation_utils import build_evaluation


def apply_sklearn_model_compatibility_patch():
    if not hasattr(_column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        _column_transformer._RemainderColsList = _RemainderColsList


@st.cache_resource
def load_models():
    loaded_models = {}
    errors = []
    apply_sklearn_model_compatibility_patch()

    for model_name, file_name in MODEL_FILES.items():
        model_path = MODELS_DIR / file_name

        if not model_path.exists():
            errors.append(f"{model_name} model file is missing: {model_path.name}")
            continue

        try:
            loaded_models[model_name] = joblib.load(model_path)
        except Exception:
            errors.append(f"{model_name} model file could not be loaded. It may be corrupt.")

    return loaded_models, errors


def get_model_classes(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)

    final_estimator = getattr(model, "steps", [(None, None)])[-1][1]
    if hasattr(final_estimator, "classes_"):
        return list(final_estimator.classes_)

    return None


def get_positive_class_index(classes):
    if classes is None:
        return None

    for index, class_value in enumerate(classes):
        if class_value in POSITIVE_CLASS_ALIASES:
            return index

        class_text = str(class_value).strip().lower()
        if class_text in POSITIVE_CLASS_ALIASES:
            return index

    return None


def get_positive_class_probabilities(model, features):
    if not hasattr(model, "predict_proba"):
        return None, None

    classes = get_model_classes(model)
    positive_class_index = get_positive_class_index(classes)
    if positive_class_index is None:
        return None, f"Could not find the `{POSITIVE_CLASS_LABEL}` class in model probabilities."

    probabilities = model.predict_proba(features)
    return probabilities[:, positive_class_index], None


def create_prediction_result(model, input_df):
    features, y_true = split_features_and_target(input_df)
    y_pred = model.predict(features)
    positive_probabilities, probability_warning = get_positive_class_probabilities(model, features)

    result_df = input_df.copy()
    if y_true is not None and "actual_y" not in result_df.columns:
        result_df["actual_y"] = y_true

    result_df["predicted_y"] = y_pred
    if positive_probabilities is not None:
        result_df["prediction_probability"] = positive_probabilities

    return {
        "result_df": result_df,
        "y_true": y_true,
        "y_pred": y_pred,
        "positive_probabilities": positive_probabilities,
        "probability_warning": probability_warning,
    }


def build_model_comparison(models, input_df):
    if TARGET_COLUMN not in input_df.columns:
        return None, ["Model comparison is available only when the dataset contains the target column `y`."]

    rows = []
    errors = []

    for model_name, model in models.items():
        try:
            prediction = create_prediction_result(model, input_df)
            evaluation = build_evaluation(
                prediction["y_true"],
                prediction["y_pred"],
                prediction["positive_probabilities"]
            )
            rows.append(
                {
                    "Model": model_name,
                    "Accuracy": evaluation["accuracy"],
                    "AUC": evaluation["auc"],
                    "Precision": evaluation["precision"],
                    "Recall": evaluation["recall"],
                    "F1": evaluation["f1"],
                    "MCC": evaluation["mcc"],
                }
            )
        except Exception:
            errors.append(f"Could not evaluate {model_name} for model comparison.")

    if not rows:
        return None, errors

    return rows, errors
