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
$expectedWebApiPort = 5006

function Test-TcpPort {
    param([string]$HostName, [int]$Port)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect($HostName, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(600)
        if (-not $ok) {
            $client.Close()
            return $false
        }
        $client.EndConnect($iar)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

function Wait-PortUp {
    param([string]$HostName, [int]$Port, [int]$MaxSeconds = 120)
    $deadline = (Get-Date).AddSeconds($MaxSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-TcpPort -HostName $HostName -Port $Port) {
            return $true
        }
        Start-Sleep -Milliseconds 900
    }
    return $false
}

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
Write-Host "[all] Ollama answer model default: llama3.2:latest"

if ($DryRun) {
    Write-Host "[all] DryRun active - nothing started."
    return
}

Write-Host "[all] Démarrage backend (Ollama + WebApi + LangGraph)..."
$backendProc = Start-Process -FilePath "powershell.exe" -ArgumentList $backendArgs -WorkingDirectory $rootDir -WindowStyle Minimized -PassThru

Write-Host "[all] Vérification WebApi sur localhost:$expectedWebApiPort ..."
$backendReady = Wait-PortUp -HostName "127.0.0.1" -Port $expectedWebApiPort -MaxSeconds 130
if (-not $backendReady) {
    Write-Host "[all] Echec: WebApi n'est pas joignable sur localhost:$expectedWebApiPort"
    if ($backendProc -and -not $backendProc.HasExited) {
        Write-Host "[all] Process backend actif (PID=$($backendProc.Id)) mais port indisponible."
    } elseif ($backendProc) {
        Write-Host "[all] Process backend terminé prématurément (PID=$($backendProc.Id), ExitCode=$($backendProc.ExitCode))."
    }
    throw "Le backend ne s'est pas initialisé correctement."
}
Write-Host "[all] WebApi actif sur localhost:$expectedWebApiPort"

Write-Host "[all] Démarrage frontend React..."
Start-Process -FilePath "npm" -ArgumentList $frontendArgs -WorkingDirectory $frontendNpm -WindowStyle Minimized | Out-Null

Write-Host "[all] Stack lancé."
Write-Host "[all] Frontend: http://localhost:5173"
Write-Host "[all] WebApi: attendu sur http://localhost:5006"
Write-Host "[all] Ollama: http://localhost:11434"
