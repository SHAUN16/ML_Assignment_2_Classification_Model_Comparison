from io import BytesIO
import hashlib

import pandas as pd

from config import (
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    SAMPLE_DATASET_PATH,
    TARGET_COLUMN,
)


def create_sample_dataset_id():
    dataset_stats = SAMPLE_DATASET_PATH.stat()
    return (
        "sample:"
        f"{SAMPLE_DATASET_PATH.resolve()}:"
        f"{dataset_stats.st_size}:"
        f"{dataset_stats.st_mtime_ns}"
    )


def load_sample_dataset():
    if not SAMPLE_DATASET_PATH.exists():
        return None, None, "Sample test dataset could not be found: test_data.csv"

    try:
        return pd.read_csv(SAMPLE_DATASET_PATH), create_sample_dataset_id(), None
    except pd.errors.ParserError:
        return None, None, "Could not parse the sample dataset as a CSV file."
    except OSError:
        return None, None, "Could not read the sample dataset file."


def load_uploaded_dataset(uploaded_file):
    if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
        return None, None, f"Please upload a CSV file no larger than {MAX_UPLOAD_SIZE_MB} MB."

    try:
        uploaded_bytes = uploaded_file.getvalue()
        uploaded_hash = hashlib.sha256(uploaded_bytes).hexdigest()
        return pd.read_csv(BytesIO(uploaded_bytes)), f"upload:{uploaded_hash}", None
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
        return None, None, "Could not parse the uploaded file as a CSV. Please upload a valid CSV file."
    except OSError:
        return None, None, "Could not read the uploaded CSV file."


def split_features_and_target(df):
    if TARGET_COLUMN in df.columns:
        return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]

    return df, None
