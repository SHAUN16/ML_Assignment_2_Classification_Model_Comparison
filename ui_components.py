import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from config import (
    COLUMN_DEFINITIONS,
    DATASET_DESCRIPTION,
    DATASET_SOURCE,
    DATASET_VERSION_DESCRIPTION,
    PREDICTION_DATASET_NOTE,
    ROWS_PER_PAGE,
    TARGET_DESCRIPTION,
)
from evaluation_utils import build_evaluation


METRIC_DEFINITIONS = [
    {
        "Metric": "Accuracy",
        "Formula": "(TP + TN) / (TP + TN + FP + FN)",
        "Meaning": "Overall share of correct predictions.",
    },
    {
        "Metric": "Precision",
        "Formula": "TP / (TP + FP)",
        "Meaning": "Of the clients predicted as subscribers, how many actually subscribed.",
    },
    {
        "Metric": "Recall",
        "Formula": "TP / (TP + FN)",
        "Meaning": "Of the actual subscribers, how many the model found.",
    },
    {
        "Metric": "F1",
        "Formula": "2 * (Precision * Recall) / (Precision + Recall)",
        "Meaning": "Balanced score between precision and recall.",
    },
    {
        "Metric": "AUC",
        "Formula": "Area under the ROC curve",
        "Meaning": "How well the model separates subscribers from non-subscribers across thresholds.",
    },
    {
        "Metric": "MCC",
        "Formula": "((TP * TN) - (FP * FN)) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))",
        "Meaning": "Matthews Correlation Coefficient. Ranges from -1 to 1; 1 is perfect, 0 is no better than random, -1 is completely wrong.",
    },
]


def render_dataset_description():
    st.subheader("Dataset Description")
    st.write(DATASET_DESCRIPTION)
    st.write(TARGET_DESCRIPTION)
    st.write(DATASET_VERSION_DESCRIPTION)
    st.info(PREDICTION_DATASET_NOTE)
    st.write(DATASET_SOURCE)

    st.write("Column Definitions")
    st.dataframe(
        pd.DataFrame(COLUMN_DEFINITIONS),
        use_container_width=True,
        hide_index=True
    )


def display_paginated_table(df, rows_per_page=20, page_key="current_page", key_prefix="table"):
    total_rows = len(df)
    total_pages = max(math.ceil(total_rows / rows_per_page), 1)

    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    st.session_state[page_key] = min(
        max(st.session_state[page_key], 1),
        total_pages
    )

    current_page = st.session_state[page_key]
    start_index = (current_page - 1) * rows_per_page
    end_index = start_index + rows_per_page

    st.dataframe(df.iloc[start_index:end_index], use_container_width=True)

    previous_col, page_col, next_col = st.columns([1, 2, 1])

    with previous_col:
        if st.button(
            "Previous",
            disabled=current_page == 1,
            use_container_width=True,
            key=f"{key_prefix}_previous"
        ):
            st.session_state[page_key] -= 1
            st.rerun()

    with page_col:
        st.markdown(
            f"<div style='text-align: center;'>"
            f"Page {current_page:,} of {total_pages:,}"
            f"</div>",
            unsafe_allow_html=True
        )

    with next_col:
        if st.button(
            "Next",
            disabled=current_page == total_pages,
            use_container_width=True,
            key=f"{key_prefix}_next"
        ):
            st.session_state[page_key] += 1
            st.rerun()


def render_dataset_preview(df, key_prefix="dataset"):
    st.write(
        f"**Rows:** {df.shape[0]:,}  |  "
        f"**Columns:** {df.shape[1]:,}"
    )
    st.subheader("Dataset Preview")
    display_paginated_table(
        df,
        rows_per_page=ROWS_PER_PAGE,
        page_key=f"{key_prefix}_page",
        key_prefix=key_prefix
    )


def render_prediction_results():
    result_df = st.session_state.get("prediction_result_df")
    if result_df is None:
        return

    st.subheader("Results")
    st.write("Prediction Results")
    st.write(f"**Model:** {st.session_state.prediction_model_name}")
    st.write(f"**Total predictions:** {len(result_df):,}")
    display_paginated_table(
        result_df,
        rows_per_page=ROWS_PER_PAGE,
        page_key="results_page",
        key_prefix="results"
    )

    st.write("Prediction Distribution")
    distribution = result_df["predicted_y"].value_counts().rename_axis("Predicted Class").reset_index(name="Count")
    st.dataframe(distribution, use_container_width=True)


def render_evaluation():
    y_true = st.session_state.get("prediction_y_true")
    if y_true is None:
        return

    st.subheader("Evaluation")

    try:
        evaluation = build_evaluation(
            y_true,
            st.session_state.get("prediction_y_pred"),
            st.session_state.get("prediction_probabilities")
        )
    except Exception:
        st.error("Could not calculate evaluation metrics for this dataset.")
        return

    st.write("Confusion Matrix")
    st.dataframe(evaluation["confusion_matrix"], use_container_width=True)
    render_confusion_matrix_plot(evaluation["confusion_matrix"])

    st.write("Evaluation Metrics")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Accuracy", f"{evaluation['accuracy']:.4f}")
    metric_cols[1].metric(
        "AUC",
        f"{evaluation['auc']:.4f}" if evaluation["auc"] is not None else "Unavailable"
    )
    metric_cols[2].metric("Precision", f"{evaluation['precision']:.4f}")
    metric_cols[3].metric("Recall", f"{evaluation['recall']:.4f}")
    metric_cols[4].metric("F1", f"{evaluation['f1']:.4f}")
    metric_cols[5].metric("MCC", f"{evaluation['mcc']:.4f}")
    render_metric_definitions(evaluation)

    st.write("ROC Curve")
    render_roc_curve_plot(evaluation)


def render_confusion_matrix_plot(confusion_matrix_df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        confusion_matrix_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        ax=ax
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)


def render_metric_definitions(evaluation):
    metric_values = {
        "Accuracy": evaluation["accuracy"],
        "Precision": evaluation["precision"],
        "Recall": evaluation["recall"],
        "F1": evaluation["f1"],
        "AUC": evaluation["auc"],
        "MCC": evaluation["mcc"],
    }

    metrics_df = pd.DataFrame(METRIC_DEFINITIONS)
    metrics_df["Value"] = metrics_df["Metric"].map(
        lambda metric: (
            f"{metric_values[metric]:.4f}"
            if metric_values[metric] is not None
            else "Unavailable"
        )
    )

    st.dataframe(
        metrics_df[["Metric", "Value", "Formula", "Meaning"]],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"TP={evaluation['tp']:,}, TN={evaluation['tn']:,}, "
        f"FP={evaluation['fp']:,}, FN={evaluation['fn']:,}."
    )


def render_roc_curve_plot(evaluation):
    if evaluation["fpr"] is None or evaluation["tpr"] is None:
        st.info("ROC curve is unavailable because the target contains only one class.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(evaluation["fpr"], evaluation["tpr"], label="Model ROC", color="#1f77b4", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    if evaluation["auc"] is not None:
        ax.legend(title=f"AUC = {evaluation['auc']:.4f}")
    else:
        ax.legend()

    st.pyplot(fig)
    plt.close(fig)


def render_download():
    result_df = st.session_state.get("prediction_result_df")
    if result_df is None:
        return

    st.subheader("Download")
    model_file_name = st.session_state.prediction_model_name.lower().replace(" ", "_")
    csv_data = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results CSV",
        data=csv_data,
        file_name=f"bank_marketing_{model_file_name}_results.csv",
        mime="text/csv"
    )
