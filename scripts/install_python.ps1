<#
Install Python env and dependencies for `ai_assistant` (Windows PowerShell)
Usage: Open PowerShell and run:
  .\scripts\install_python.ps1
#>
param(
  [string]$venvName = ".venv"
)

Write-Host "== Create virtual environment: $venvName =="
python -m venv $venvName

Write-Host "== Activate virtual environment =="
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
& "$PWD\$venvName\Scripts\Activate.ps1"

Write-Host "== Upgrade pip and setuptools =="
python -m pip install --upgrade pip setuptools

if (Test-Path "ai_assistant/requirements.txt") {
  Write-Host "== Install requirements.txt =="
  pip install -r ai_assistant/requirements.txt
} else {
  Write-Host "requirements file not found at ai_assistant/requirements.txt"
}

Write-Host "== Done =="
Write-Host "Activate later with: & '$PWD\$venvName\Scripts\Activate.ps1'"