import streamlit as st

from config import (
    APP_TITLE,
    DEFAULT_MODEL_NAME,
    MODEL_FILES,
    PAGE_ICON,
)
from data_utils import load_sample_dataset, load_uploaded_dataset
from model_utils import create_prediction_result, load_models
from state_utils import clear_prediction_state, reset_page_if_dataset_changed, save_prediction_state
from ui_components import (
    render_dataset_description,
    render_dataset_preview,
    render_download,
    render_evaluation,
    render_prediction_results,
)


# --- Page Setup ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="wide"
)

st.title(APP_TITLE)


# --- Dataset Description ---
render_dataset_description()


# --- Dataset ---
df = None
dataset_id = None

st.subheader("Dataset")

dataset_option = st.radio(
    "Choose how to provide the dataset:",
    ["Use Sample Test Dataset", "Upload CSV"],
    index=0
)

if dataset_option == "Use Sample Test Dataset":
    df, dataset_id, dataset_error = load_sample_dataset()
    if dataset_error:
        st.error(dataset_error)
    else:
        reset_page_if_dataset_changed(dataset_id)
        st.success("Sample test dataset loaded successfully.")
        render_dataset_preview(df)

else:
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        df, dataset_id, dataset_error = load_uploaded_dataset(uploaded_file)
        if dataset_error:
            st.error(dataset_error)
        else:
            reset_page_if_dataset_changed(dataset_id)
            st.success(
                f"Successfully uploaded `{uploaded_file.name}` "
                f"({uploaded_file.size / (1024 * 1024):.2f} MB)"
            )
            render_dataset_preview(df)


# --- Model ---
if df is not None:
    st.subheader("Model")

    models, model_errors = load_models()
    for error in model_errors:
        st.error(error)

    available_model_names = [name for name in MODEL_FILES if name in models]

    if available_model_names:
        default_model_index = (
            available_model_names.index(DEFAULT_MODEL_NAME)
            if DEFAULT_MODEL_NAME in available_model_names
            else 0
        )
        selected_model_name = st.selectbox(
            "Select Model",
            available_model_names,
            index=default_model_index
        )

        if st.button("Run Prediction"):
            try:
                prediction = create_prediction_result(models[selected_model_name], df)
                save_prediction_state(prediction, selected_model_name, dataset_id)

                if prediction["probability_warning"]:
                    st.warning(prediction["probability_warning"])
                st.success("Prediction completed successfully.")
            except Exception:
                st.error("Could not run prediction with the selected model and dataset.")
    else:
        st.error("No trained models are available for prediction.")

else:
    clear_prediction_state()


# --- Results ---
if df is not None:
    render_prediction_results()


# --- Evaluation ---
if df is not None:
    render_evaluation()


# --- Download ---
if df is not None:
    render_download()
