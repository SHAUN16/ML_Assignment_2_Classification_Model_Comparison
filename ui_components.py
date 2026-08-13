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
from model_utils import build_model_comparison


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


def apply_app_styles():
    st.markdown(
        """
        <style>
            :root {
                --accent: #2563eb;
                --accent-soft: #eff6ff;
                --border: #d8dee9;
                --ink-muted: #4b5563;
                --panel: #f8fafc;
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1180px;
            }

            h1 {
                font-size: 2.15rem;
                font-weight: 750;
                letter-spacing: 0;
                margin-bottom: 0.25rem;
            }

            h3 {
                padding-top: 1rem;
                border-top: 1px solid var(--border);
                margin-top: 1.8rem;
            }

            [data-testid="stMetric"] {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 0.8rem 0.95rem;
                color: #111827;
            }

            [data-testid="stMetricLabel"] p {
                color: var(--ink-muted);
                font-size: 0.85rem;
            }

            [data-testid="stMetricValue"] {
                color: #111827;
                font-size: 1.35rem;
                font-weight: 700;
            }

            [data-testid="stMetricDelta"] {
                color: #374151;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid var(--border);
                border-radius: 8px;
                overflow: hidden;
            }

            .info-panel {
                background: var(--accent-soft);
                border-left: 4px solid var(--accent);
                border-radius: 8px;
                padding: 0.9rem 1rem;
                margin: 0.8rem 0 1rem;
                color: #1f2937;
            }

            .source-note {
                color: var(--ink-muted);
                font-size: 0.92rem;
                margin-top: 0.3rem;
            }

            .page-indicator {
                text-align: center;
                color: var(--ink-muted);
                font-size: 0.95rem;
                padding-top: 0.45rem;
            }

            .stButton > button {
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_intro():
    st.caption("Pre-trained classification model comparison")


def render_dataset_description():
    st.subheader("Dataset Description")
    st.write(DATASET_DESCRIPTION)

    overview_cols = st.columns(3)
    overview_cols[0].metric("Input Variables", "16")
    overview_cols[1].metric("Target", "y")
    overview_cols[2].metric("Task", "Binary Classification")

    st.markdown(
        f"<div class='info-panel'>{TARGET_DESCRIPTION}</div>",
        unsafe_allow_html=True
    )
    st.info(PREDICTION_DATASET_NOTE)

    with st.expander("Dataset Version and Source", expanded=False):
        st.write(DATASET_VERSION_DESCRIPTION)
        st.markdown(
            f"<div class='source-note'>{DATASET_SOURCE}</div>",
            unsafe_allow_html=True
        )

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
            f"<div class='page-indicator'>"
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
    summary_cols = st.columns(2)
    summary_cols[0].metric("Rows", f"{df.shape[0]:,}")
    summary_cols[1].metric("Columns", f"{df.shape[1]:,}")

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
    result_cols = st.columns(2)
    result_cols[0].metric("Model", st.session_state.prediction_model_name)
    result_cols[1].metric("Total Predictions", f"{len(result_df):,}")

    display_paginated_table(
        result_df,
        rows_per_page=ROWS_PER_PAGE,
        page_key="results_page",
        key_prefix="results"
    )

    st.write("Prediction Distribution")
    distribution = result_df["predicted_y"].value_counts().rename_axis("Predicted Class").reset_index(name="Count")
    dist_cols = st.columns([1, 2])
    with dist_cols[0]:
        st.dataframe(distribution, use_container_width=True, hide_index=True)
    with dist_cols[1]:
        render_prediction_distribution_plot(distribution)


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
    fig, ax = plt.subplots(figsize=(6.5, 4.3), facecolor="white")
    ax.set_facecolor("white")
    sns.heatmap(
        confusion_matrix_df,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#2563eb", as_cmap=True),
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 12, "fontweight": "bold", "color": "#111827"},
        ax=ax
    )
    ax.set_xlabel("Predicted Class", color="#111827")
    ax.set_ylabel("Actual Class", color="#111827")
    ax.set_title("Confusion Matrix", color="#111827")
    ax.tick_params(colors="#111827")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_prediction_distribution_plot(distribution):
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor="white")
    ax.set_facecolor("white")
    sns.barplot(
        data=distribution,
        x="Predicted Class",
        y="Count",
        color="#2563eb",
        ax=ax
    )
    ax.set_title("Predicted Class Counts", color="#111827")
    ax.set_xlabel("Predicted Class", color="#111827")
    ax.set_ylabel("Count", color="#111827")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(colors="#111827")

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, color="#111827")

    sns.despine(ax=ax)
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(6.5, 4.3), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(evaluation["fpr"], evaluation["tpr"], label="Model ROC", color="#2563eb", linewidth=2.4)
    ax.fill_between(evaluation["fpr"], evaluation["tpr"], alpha=0.12, color="#2563eb")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Random baseline")
    ax.set_xlabel("False Positive Rate", color="#111827")
    ax.set_ylabel("True Positive Rate", color="#111827")
    ax.set_title("ROC Curve", color="#111827")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="#111827")

    if evaluation["auc"] is not None:
        legend = ax.legend(title=f"AUC = {evaluation['auc']:.4f}")
    else:
        legend = ax.legend()

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d1d5db")
    for text in legend.get_texts():
        text.set_color("#111827")
    legend.get_title().set_color("#111827")

    sns.despine(ax=ax)
    fig.tight_layout()
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


def render_model_comparison(models, df, dataset_id):
    st.subheader("Model Comparison")

    if st.session_state.get("model_comparison_dataset_id") != dataset_id:
        with st.spinner("Comparing all models on the selected dataset..."):
            comparison_rows, comparison_errors = build_model_comparison(models, df)

        st.session_state.model_comparison_rows = comparison_rows
        st.session_state.model_comparison_errors = comparison_errors
        st.session_state.model_comparison_dataset_id = dataset_id
    else:
        comparison_rows = st.session_state.get("model_comparison_rows")
        comparison_errors = st.session_state.get("model_comparison_errors", [])

    for error in comparison_errors:
        st.warning(error)

    if comparison_rows is None:
        return

    comparison_df = pd.DataFrame(comparison_rows)
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    best_by_metric = build_best_model_summary(comparison_df, metric_columns)

    st.write("Models vs Metrics")
    st.dataframe(
        style_model_comparison(comparison_df, metric_columns),
        use_container_width=True,
        hide_index=True
    )

    st.write("Best Model by Metric")
    st.dataframe(best_by_metric, use_container_width=True, hide_index=True)


def style_model_comparison(comparison_df, metric_columns):
    def highlight_best(column):
        if column.name not in metric_columns:
            return [""] * len(column)

        best_value = column.max(skipna=True)
        return [
            "background-color: #dcfce7; color: #14532d; font-weight: 700;"
            if value == best_value
            else ""
            for value in column
        ]

    return comparison_df.style.apply(highlight_best, axis=0).format(
        {metric: "{:.4f}" for metric in metric_columns}
    )


def build_best_model_summary(comparison_df, metric_columns):
    rows = []

    for metric in metric_columns:
        metric_values = comparison_df[metric].dropna()
        if metric_values.empty:
            rows.append(
                {
                    "Metric": metric,
                    "Best Model": "Unavailable",
                    "Score": "Unavailable",
                }
            )
            continue

        best_index = metric_values.idxmax()
        rows.append(
            {
                "Metric": metric,
                "Best Model": comparison_df.loc[best_index, "Model"],
                "Score": f"{comparison_df.loc[best_index, metric]:.4f}",
            }
        )

    return pd.DataFrame(rows)
