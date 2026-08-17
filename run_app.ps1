$ErrorActionPreference = "Stop"

Write-Host "Preparing local Streamlit run..."

$insideGitRepo = $false
if (Get-Command git -ErrorAction SilentlyContinue) {
    git rev-parse --is-inside-work-tree *> $null
    $insideGitRepo = ($LASTEXITCODE -eq 0)
}

if ($insideGitRepo) {
    git lfs version *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Pulling Git LFS model files..."
        git lfs install
        git lfs pull
    }
    else {
        Write-Warning "Git LFS is not installed. Large model files may not be available."
        Write-Host "Install Git LFS, then run: git lfs pull"
    }
}
else {
    Write-Host "Git repository not detected. Skipping Git LFS pull."
}

$randomForestPath = Join-Path "models" "random_forest.joblib"
if (Test-Path $randomForestPath) {
    $firstLine = Get-Content -LiteralPath $randomForestPath -TotalCount 1 -ErrorAction SilentlyContinue
    if ($firstLine -eq "version https://git-lfs.github.com/spec/v1") {
        Write-Warning "models/random_forest.joblib is still a Git LFS pointer file."
        Write-Host "The app may fail to load models until Git LFS files are pulled."
    }
}

$systemPythonCmd = "python"
if (Get-Command py -ErrorAction SilentlyContinue) {
    $systemPythonCmd = "py"
}

$venvDir = ".venv"
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment in .venv..."
    & $systemPythonCmd -m venv .venv
}

$venvPython = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = Join-Path ".venv" "bin/python"
}

if (-not (Test-Path $venvPython)) {
    throw "Could not find Python inside .venv."
}

Write-Host "Using virtual environment Python: $venvPython"

Write-Host "Installing Python dependencies..."
& $venvPython -m pip install -r requirements.txt

Write-Host "Starting Streamlit app..."
& $venvPython -m streamlit run app.py
