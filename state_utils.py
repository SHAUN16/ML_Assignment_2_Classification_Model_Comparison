import streamlit as st


def clear_prediction_state():
    for key in [
        "prediction_result_df",
        "prediction_y_true",
        "prediction_y_pred",
        "prediction_probabilities",
        "prediction_model_name",
        "prediction_dataset_id",
        "results_page",
        "model_comparison_rows",
        "model_comparison_errors",
        "model_comparison_dataset_id",
    ]:
        st.session_state.pop(key, None)


def reset_page_if_dataset_changed(dataset_id):
    if st.session_state.get("dataset_id") != dataset_id:
        st.session_state.dataset_id = dataset_id
        st.session_state.dataset_page = 1
        clear_prediction_state()


def save_prediction_state(prediction, model_name, dataset_id):
    st.session_state.prediction_result_df = prediction["result_df"]
    st.session_state.prediction_y_true = prediction["y_true"]
    st.session_state.prediction_y_pred = prediction["y_pred"]
    st.session_state.prediction_probabilities = prediction["positive_probabilities"]
    st.session_state.prediction_model_name = model_name
    st.session_state.prediction_dataset_id = dataset_id
    st.session_state.results_page = 1
