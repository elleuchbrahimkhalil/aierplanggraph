param(
    [string]$PythonExe = "c:/Users/brahim/OneDrive/Bureau/example erp stage pfe/javaerp/.venv/Scripts/python.exe",
    [string]$WebApiProjectDir = "C:/Users/brahim/OneDrive/Bureau/stage2026/web/Webservices_webclient/stagepfe26/WebApi",
    [string]$FrontendDir = "frontend",
    [int]$AssistantPort = 8000,
    [int]$AssistantMaxWaitSeconds = 150,
    [string]$ExtractorModel = "llama3.2",
    [string]$RouterModel = "deepseek-coder:6.7b",
    [string]$TransformerModel = "deepseek-coder:6.7b",
    [string]$AnswerModel = "llama3.2",
    [int]$RouterCandidateLimit = 24,
    [int]$RouterColumnLimit = 24,
    [string]$ApiAuthUrl = "",
    [string]$ApiUsername = "",
    [string]$ApiPassword = "",
    [string]$ApiSociete = "",
    [switch]$GenerateTextAnswer,
    [switch]$UseOllama = $true,
    [switch]$AutoPullModels = $true,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Force UTF-8 console output to avoid mojibake (ex: VÃ©rification).
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [Console]::OutputEncoding
    chcp 65001 | Out-Null
}
catch {
    # Non-blocking: continue with default encoding if console cannot switch.
}

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendScript = Join-Path $rootDir "ai_assistant/start_stack.ps1"
$frontendNpm = Join-Path $rootDir $FrontendDir
$expectedWebApiPort = 5006
$expectedAssistantPort = $AssistantPort
$expectedFrontendPorts = @(5173)

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
    }
    catch {
        return $false
    }
}

function Wait-PortUp {
    param([string[]]$HostNames, [int]$Port, [int]$MaxSeconds = 120)
    $deadline = (Get-Date).AddSeconds($MaxSeconds)
    while ((Get-Date) -lt $deadline) {
        foreach ($h in $HostNames) {
            if (Test-TcpPort -HostName $h -Port $Port) {
                return $h
            }
        }
        Start-Sleep -Milliseconds 900
    }
    return $null
}

function Wait-AnyPortUp {
    param([string[]]$HostNames, [int[]]$Ports, [int]$MaxSeconds = 120)
    $deadline = (Get-Date).AddSeconds($MaxSeconds)
    while ((Get-Date) -lt $deadline) {
        foreach ($port in $Ports) {
            foreach ($h in $HostNames) {
                if (Test-TcpPort -HostName $h -Port $port) {
                    return @{
                        Host = $h
                        Port = $port
                    }
                }
            }
        }
        Start-Sleep -Milliseconds 900
    }
    return $null
}

function Test-AssistantHttpHealth {
    param(
        [string]$HostName,
        [int]$Port,
        [int]$TimeoutSeconds = 4
    )

    $url = "http://${HostName}:$Port/health"
    try {
        $resp = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec $TimeoutSeconds
        if ($null -ne $resp -and $resp.status -eq "ok") {
            return $true
        }
    }
    catch {
        return $false
    }
    return $false
}

function Wait-AssistantUp {
    param(
        [string[]]$HostNames,
        [int]$Port,
        [int]$MaxSeconds = 150
    )

    $deadline = (Get-Date).AddSeconds($MaxSeconds)
    while ((Get-Date) -lt $deadline) {
        foreach ($h in $HostNames) {
            if (Test-TcpPort -HostName $h -Port $Port) {
                if (Test-AssistantHttpHealth -HostName $h -Port $Port -TimeoutSeconds 4) {
                    return $h
                }
            }
        }
        Start-Sleep -Milliseconds 900
    }
    return $null
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
    "-File", "`"$backendScript`"",
    "-PythonExe", "`"$PythonExe`"",
    "-WebApiProjectDir", "`"$WebApiProjectDir`"",
    "-AssistantPort", $AssistantPort,
    "-ExtractorModel", "`"$ExtractorModel`"",
    "-RouterModel", "`"$RouterModel`"",
    "-TransformerModel", "`"$TransformerModel`"",
    "-AnswerModel", "`"$AnswerModel`"",
    "-RouterCandidateLimit", $RouterCandidateLimit,
    "-RouterColumnLimit", $RouterColumnLimit
)
if (-not [string]::IsNullOrWhiteSpace($ApiAuthUrl)) {
    $backendArgs += @("-ApiAuthUrl", "`"$ApiAuthUrl`"")
}
if (-not [string]::IsNullOrWhiteSpace($ApiUsername)) {
    $backendArgs += @("-ApiUsername", "`"$ApiUsername`"")
}
if (-not [string]::IsNullOrWhiteSpace($ApiPassword)) {
    $backendArgs += @("-ApiPassword", "`"$ApiPassword`"")
}
if (-not [string]::IsNullOrWhiteSpace($ApiSociete)) {
    $backendArgs += @("-ApiSociete", "`"$ApiSociete`"")
}
if ($UseOllama) {
    $backendArgs += "-UseOllama"
}
if ($AutoPullModels) {
    $backendArgs += "-AutoPullModels"
}
if ($GenerateTextAnswer) {
    $backendArgs += "-GenerateTextAnswer"
}

$frontendArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-Command", "Set-Location -LiteralPath '$frontendNpm'; & npm.cmd run dev"
)

Write-Host "[all] Start plan:"
$displayBackendArgs = @()
for ($i = 0; $i -lt $backendArgs.Count; $i++) {
    $displayBackendArgs += $backendArgs[$i]
    if ($backendArgs[$i] -eq "-ApiPassword" -and ($i + 1) -lt $backendArgs.Count) {
        $displayBackendArgs += '"***"'
        $i++
    }
}
Write-Host "[all] Backend: powershell.exe $($displayBackendArgs -join ' ')"
Write-Host "[all] Frontend: powershell.exe $($frontendArgs -join ' ')"
Write-Host "[all] Models: extractor=$ExtractorModel router=$RouterModel transform=$TransformerModel answer=$AnswerModel"

if ($DryRun) {
    Write-Host "[all] DryRun active - nothing started."
    return
}

Write-Host "[all] Démarrage backend (Ollama + WebApi + LangGraph)..."
$backendProc = Start-Process -FilePath "powershell.exe" -ArgumentList $backendArgs -WorkingDirectory $rootDir -WindowStyle Minimized -PassThru

Write-Host "[all] Vérification WebApi sur localhost:$expectedWebApiPort ..."
$backendReadyOn = Wait-PortUp -HostNames @("localhost", "127.0.0.1", "::1") -Port $expectedWebApiPort -MaxSeconds 130
if (-not $backendReadyOn) {
    Write-Host "[all] Echec: WebApi n'est pas joignable sur localhost:$expectedWebApiPort"
    if ($backendProc -and -not $backendProc.HasExited) {
        Write-Host "[all] Process backend actif (PID=$($backendProc.Id)) mais port indisponible."
    }
    elseif ($backendProc) {
        Write-Host "[all] Process backend terminé prématurément (PID=$($backendProc.Id), ExitCode=$($backendProc.ExitCode))."
    }
    throw "Le backend ne s'est pas initialisé correctement."
}
Write-Host "[all] WebApi actif sur ${backendReadyOn}:$expectedWebApiPort"

Write-Host "[all] Vérification Assistant API sur 127.0.0.1:$expectedAssistantPort ..."
$assistantReadyOn = Wait-AssistantUp -HostNames @("127.0.0.1", "localhost", "::1") -Port $expectedAssistantPort -MaxSeconds $AssistantMaxWaitSeconds
if (-not $assistantReadyOn) {
    Write-Host "[all] Echec: Assistant API non prêt sur :$expectedAssistantPort (TCP/health)."
    if ($backendProc -and -not $backendProc.HasExited) {
        Write-Host "[all] Process backend actif (PID=$($backendProc.Id))."
        Write-Host "[all] Conseil: vérifier si le serveur Python est lancé avec --serve et s'il bind sur 127.0.0.1:8000."
    }
    elseif ($backendProc) {
        Write-Host "[all] Process backend terminé prématurément (PID=$($backendProc.Id), ExitCode=$($backendProc.ExitCode))."
    }
    throw "L'assistant LangGraph n'est pas joignable sur le port $expectedAssistantPort."
}
Write-Host "[all] Assistant API actif sur ${assistantReadyOn}:$expectedAssistantPort"

Write-Host "[all] Démarrage frontend React..."
$frontendProc = Start-Process -FilePath "powershell.exe" -ArgumentList $frontendArgs -WorkingDirectory $rootDir -WindowStyle Normal -PassThru

Write-Host "[all] Vérification Frontend sur les ports possibles: $($expectedFrontendPorts -join ', ') ..."
$frontendReady = Wait-AnyPortUp -HostNames @("localhost", "127.0.0.1", "::1") -Ports $expectedFrontendPorts -MaxSeconds 45
if (-not $frontendReady) {
    Write-Host "[all] Echec: Frontend non joignable sur les ports attendus: $($expectedFrontendPorts -join ', ')"
    if ($frontendProc -and -not $frontendProc.HasExited) {
        Write-Host "[all] Process frontend actif (PID=$($frontendProc.Id)) mais port indisponible."
        Write-Host "[all] Conseil: vérifier la fenêtre frontend (npm.cmd run dev) pour les erreurs npm/vite."
    }
    elseif ($frontendProc) {
        Write-Host "[all] Process frontend terminé prématurément (PID=$($frontendProc.Id), ExitCode=$($frontendProc.ExitCode))."
    }
    throw "Le frontend Vite ne s'est pas lancé correctement."
}
Write-Host "[all] Frontend actif sur $($frontendReady.Host):$($frontendReady.Port)"

Write-Host "[all] Stack lancé."
Write-Host "[all] Frontend: http://$($frontendReady.Host):$($frontendReady.Port)"
Write-Host "[all] WebApi: attendu sur http://localhost:5006"
Write-Host "[all] Assistant API: http://127.0.0.1:$expectedAssistantPort"
Write-Host "[all] Ollama: http://localhost:11434"
