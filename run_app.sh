#!/usr/bin/env bash
set -e

echo "Preparing local Streamlit run..."

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git lfs version >/dev/null 2>&1; then
        echo "Pulling Git LFS model files..."
        git lfs install
        git lfs pull
    else
        echo "Warning: Git LFS is not installed. Large model files may not be available."
        echo "Install Git LFS, then run: git lfs pull"
    fi
else
    echo "Git repository not detected. Skipping Git LFS pull."
fi

if [ -f "models/random_forest.joblib" ] && head -n 1 "models/random_forest.joblib" | grep -q "version https://git-lfs.github.com/spec/v1"; then
    echo "Warning: models/random_forest.joblib is still a Git LFS pointer file."
    echo "The app may fail to load models until Git LFS files are pulled."
fi

SYSTEM_PYTHON_CMD=""

for candidate in python python3 py; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" --version >/dev/null 2>&1; then
        SYSTEM_PYTHON_CMD="$candidate"
        break
    fi
done

if [ -z "$SYSTEM_PYTHON_CMD" ]; then
    echo "Error: Python was not found. Install Python or activate a virtual environment, then rerun this script."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    "$SYSTEM_PYTHON_CMD" -m venv .venv
fi

if [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON_CMD=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    echo "Error: Could not find Python inside .venv."
    exit 1
fi

echo "Using virtual environment Python: $PYTHON_CMD"

echo "Installing Python dependencies..."
"$PYTHON_CMD" -m pip install -r requirements.txt

echo "Starting Streamlit app..."
"$PYTHON_CMD" -m streamlit run app.py
