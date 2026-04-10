param(
    [string]$PythonExe = "c:/Users/brahim/OneDrive/Bureau/example erp stage pfe/javaerp/.venv/Scripts/python.exe",
    [string]$WebApiProjectDir = "C:/Users/brahim/OneDrive/Bureau/stage2026/web/Webservices_webclient/stagepfe26/WebApi",
    [string]$FrontendDir = "frontend",
    [switch]$UseOllama = $true,
    [switch]$AutoPullModels = $true,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendScript = Join-Path $rootDir "ai_assistant/start_stack.ps1"
$frontendNpm = Join-Path $rootDir $FrontendDir

function Write-PlanLine {
    param([string]$Text)
    Write-Host $Text
}

if (-not (Test-Path $backendScript)) {
    throw "Script backend introuvable: $backendScript"
}
if (-not (Test-Path $frontendNpm)) {
    throw "Dossier frontend introuvable: $frontendNpm"
}

$backendArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $backendScript,
    "-PythonExe", $PythonExe,
    "-WebApiProjectDir", $WebApiProjectDir
)
if ($UseOllama) {
    $backendArgs += "-UseOllama"
}
if ($AutoPullModels) {
    $backendArgs += "-AutoPullModels"
}

$frontendArgs = @(
    "run",
    "dev"
)

Write-Host "[all] Start plan:"
Write-Host "[all] Backend: powershell.exe $($backendArgs -join ' ')"
Write-Host "[all] Frontend: npm $($frontendArgs -join ' ') (cwd=$frontendNpm)"

if ($DryRun) {
    Write-Host "[all] DryRun active - nothing started."
    return
}

Write-Host "[all] Démarrage backend (Ollama + WebApi + LangGraph)..."
Start-Process -FilePath "powershell.exe" -ArgumentList $backendArgs -WorkingDirectory $rootDir -WindowStyle Minimized | Out-Null

Write-Host "[all] Démarrage frontend React..."
Start-Process -FilePath "npm" -ArgumentList $frontendArgs -WorkingDirectory $frontendNpm -WindowStyle Minimized | Out-Null

Write-Host "[all] Stack lancé."
Write-Host "[all] Frontend: http://localhost:5173"
Write-Host "[all] WebApi: attendu sur http://localhost:5006"
Write-Host "[all] Ollama: http://localhost:11434"
