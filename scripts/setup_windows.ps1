param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

& $Python -m venv (Join-Path $ProjectRoot ".venv")
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
& $VenvPython -m pip install openmim
& $VenvPython -m mim install "mmcv==2.1.0"
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
& $VenvPython (Join-Path $ProjectRoot "scripts\check_environment.py")

Write-Host "Environment is ready: $VenvPython"
