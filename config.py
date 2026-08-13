from pathlib import Path


BASE_DIR = Path(__file__).parent
SAMPLE_DATASET_PATH = BASE_DIR / "test_data.csv"
MODELS_DIR = BASE_DIR / "models"

APP_TITLE = "Bank Marketing Classification"
PAGE_ICON = ":bar_chart:"

MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ROWS_PER_PAGE = 20

TARGET_COLUMN = "y"
NEGATIVE_CLASS = 0
POSITIVE_CLASS = 1
POSITIVE_CLASS_LABEL = "yes"
POSITIVE_CLASS_ALIASES = {POSITIVE_CLASS_LABEL, "1", POSITIVE_CLASS}

DATASET_DESCRIPTION = (
    "The data is related to direct marketing campaigns of a Portuguese banking "
    "institution. The campaigns were based on phone calls, and more than one "
    "contact to the same client was sometimes required."
)

TARGET_DESCRIPTION = (
    "The classification goal is to predict whether the client will subscribe "
    "to a term deposit. The target column is `y`: `0` means no subscription, "
    "and `1` means subscription."
)

DATASET_VERSION_DESCRIPTION = (
    "This application uses the older Bank Marketing dataset structure with "
    "16 input variables and 1 output variable. Smaller dataset variants are "
    "commonly used for testing more computationally demanding ML algorithms."
)

PREDICTION_DATASET_NOTE = (
    "The dataset selected in this app is used only as a test/prediction dataset. "
    "The saved models have already been trained previously and are loaded from "
    "the models directory; no model training is performed in this Streamlit app."
)

DATASET_SOURCE = (
    "Source: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5K306"
)

COLUMN_DEFINITIONS = [
    {
        "Column": "age",
        "Group": "Bank client data",
        "Type": "Numeric",
        "Definition": "Client age.",
    },
    {
        "Column": "job",
        "Group": "Bank client data",
        "Type": "Categorical",
        "Definition": "Type of job: admin., unknown, unemployed, management, housemaid, entrepreneur, student, blue-collar, self-employed, retired, technician, services.",
    },
    {
        "Column": "marital",
        "Group": "Bank client data",
        "Type": "Categorical",
        "Definition": "Marital status: married, divorced, single. Divorced includes divorced or widowed.",
    },
    {
        "Column": "education",
        "Group": "Bank client data",
        "Type": "Categorical",
        "Definition": "Education level: unknown, secondary, primary, tertiary.",
    },
    {
        "Column": "default",
        "Group": "Bank client data",
        "Type": "Binary",
        "Definition": "Whether the client has credit in default: yes or no.",
    },
    {
        "Column": "balance",
        "Group": "Bank client data",
        "Type": "Numeric",
        "Definition": "Average yearly balance, in euros.",
    },
    {
        "Column": "housing",
        "Group": "Bank client data",
        "Type": "Binary",
        "Definition": "Whether the client has a housing loan: yes or no.",
    },
    {
        "Column": "loan",
        "Group": "Bank client data",
        "Type": "Binary",
        "Definition": "Whether the client has a personal loan: yes or no.",
    },
    {
        "Column": "contact",
        "Group": "Last contact",
        "Type": "Categorical",
        "Definition": "Contact communication type: unknown, telephone, cellular.",
    },
    {
        "Column": "day",
        "Group": "Last contact",
        "Type": "Numeric",
        "Definition": "Last contact day of the month.",
    },
    {
        "Column": "month",
        "Group": "Last contact",
        "Type": "Categorical",
        "Definition": "Last contact month of year: jan, feb, mar, ..., nov, dec.",
    },
    {
        "Column": "duration",
        "Group": "Last contact",
        "Type": "Numeric",
        "Definition": "Last contact duration, in seconds.",
    },
    {
        "Column": "campaign",
        "Group": "Other attributes",
        "Type": "Numeric",
        "Definition": "Number of contacts performed during this campaign for this client, including the last contact.",
    },
    {
        "Column": "pdays",
        "Group": "Other attributes",
        "Type": "Numeric",
        "Definition": "Number of days since the client was last contacted in a previous campaign. -1 means not previously contacted.",
    },
    {
        "Column": "previous",
        "Group": "Other attributes",
        "Type": "Numeric",
        "Definition": "Number of contacts performed before this campaign for this client.",
    },
    {
        "Column": "poutcome",
        "Group": "Other attributes",
        "Type": "Categorical",
        "Definition": "Outcome of the previous marketing campaign: unknown, other, failure, success.",
    },
    {
        "Column": "y",
        "Group": "Output target",
        "Type": "Binary",
        "Definition": "Whether the client subscribed to a term deposit. Original labels are yes or no; this prepared dataset uses 1 for yes and 0 for no.",
    },
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

DEFAULT_MODEL_NAME = "Random Forest"
