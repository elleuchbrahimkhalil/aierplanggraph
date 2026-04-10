param(
    [string]$PythonExe = "c:/Users/brahim/OneDrive/Bureau/example erp stage pfe/javaerp/.venv/Scripts/python.exe",
    [string]$LangGraphScript = "ai_assistant/langgraph_skeleton.py",
    [string]$EndpointsJson = "C:/Users/brahim/OneDrive/Bureau/example erp stage pfe/aierpjava/test/src/main/resources/endpoints.json",
    [string]$WebApiProjectDir = "C:/Users/brahim/OneDrive/Bureau/stage2026/web/Webservices_webclient/stagepfe26/WebApi",
    [string]$RouterModel = "deepseek-coder:6.7b",
    [string]$AnswerModel = "llama3.1:8b",
    [switch]$UseOllama,
    [switch]$AutoPullModels
)

$ErrorActionPreference = "Stop"

function Test-TcpPort {
    param([string]$Host, [int]$Port)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect($Host, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(500)
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

function Wait-HttpUp {
    param([string[]]$Urls, [int]$MaxSeconds = 45)
    $deadline = (Get-Date).AddSeconds($MaxSeconds)
    while ((Get-Date) -lt $deadline) {
        foreach ($u in $Urls) {
            try {
                $r = Invoke-WebRequest -Uri $u -Method Get -UseBasicParsing -TimeoutSec 3
                if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) {
                    return $u
                }
            } catch {
                # Keep retrying.
            }
        }
        Start-Sleep -Milliseconds 900
    }
    return $null
}

function Get-WebApiUrlsFromLaunchSettings {
    param([string]$ProjectDir)
    $launch = Join-Path $ProjectDir "Properties/launchSettings.json"
    if (-not (Test-Path $launch)) {
        return @()
    }

    $raw = Get-Content -Path $launch -Raw -Encoding UTF8
    if ($raw.Length -gt 0 -and $raw[0] -eq [char]0xFEFF) {
        $raw = $raw.Substring(1)
    }

    $json = $raw | ConvertFrom-Json
    $profiles = $json.profiles
    if (-not $profiles) {
        return @()
    }

    $ordered = @()
    if ($profiles.PSObject.Properties.Name -contains "WebApi") {
        $ordered += $profiles.WebApi
    }
    foreach ($p in $profiles.PSObject.Properties) {
        if ($p.Name -ne "WebApi") {
            $ordered += $p.Value
        }
    }

    $urls = @()
    foreach ($profile in $ordered) {
        if ($null -ne $profile.applicationUrl -and "$($profile.applicationUrl)".Trim() -ne "") {
            $parts = "$($profile.applicationUrl)".Split(";", [System.StringSplitOptions]::RemoveEmptyEntries)
            foreach ($part in $parts) {
                $u = $part.Trim().TrimEnd('/')
                if ($u -and -not ($urls -contains $u)) {
                    $urls += $u
                }
            }
        }
    }
    return $urls
}

function Ensure-OllamaRunning {
    Write-Host "[stack] Vérification Ollama..."
    if (Test-TcpPort -Host "127.0.0.1" -Port 11434) {
        Write-Host "[stack] Ollama déjà actif sur 11434"
    } else {
        $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
        if (-not $ollamaCmd) {
            throw "Ollama n'est pas installé ou non présent dans PATH."
        }

        Write-Host "[stack] Démarrage Ollama server..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Minimized | Out-Null

        $up = Wait-HttpUp -Urls @("http://127.0.0.1:11434/api/tags") -MaxSeconds 25
        if (-not $up) {
            throw "Impossible de démarrer Ollama server sur 11434."
        }
        Write-Host "[stack] Ollama actif"
    }

    if ($AutoPullModels) {
        Write-Host "[stack] Vérification modèles Ollama..."
        $models = (& ollama list) 2>$null
        if (-not ($models -match [regex]::Escape($RouterModel))) {
            Write-Host "[stack] Téléchargement modèle routeur: $RouterModel"
            & ollama pull $RouterModel
        }
        if (-not ($models -match [regex]::Escape($AnswerModel))) {
            Write-Host "[stack] Téléchargement modèle réponse: $AnswerModel"
            & ollama pull $AnswerModel
        }
    }
}

function Ensure-WebApiRunning {
    param([string]$ProjectDir)

    if (-not (Test-Path $ProjectDir)) {
        throw "WebApi project introuvable: $ProjectDir"
    }

    $urls = Get-WebApiUrlsFromLaunchSettings -ProjectDir $ProjectDir
    if ($urls.Count -eq 0) {
        throw "Aucune applicationUrl trouvée dans launchSettings.json"
    }

    Write-Host "[stack] URLs WebApi détectées: $($urls -join ', ')"

    $probeUrls = @()
    foreach ($u in $urls) {
        $probeUrls += "$u/swagger"
        $probeUrls += "$u/swagger/index.html"
    }

    $online = Wait-HttpUp -Urls $probeUrls -MaxSeconds 2
    if ($online) {
        Write-Host "[stack] WebApi déjà actif: $online"
        return $urls
    }

    $dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
    if (-not $dotnet) {
        throw "dotnet SDK non trouvé dans PATH."
    }

    $logPath = Join-Path $ProjectDir "webapi-run.log"
    $errLogPath = Join-Path $ProjectDir "webapi-run.err.log"
    Write-Host "[stack] Démarrage WebApi..."
    Start-Process -FilePath "dotnet" -ArgumentList @("run", "--launch-profile", "WebApi") -WorkingDirectory $ProjectDir -WindowStyle Minimized -RedirectStandardOutput $logPath -RedirectStandardError $errLogPath | Out-Null

    $online = Wait-HttpUp -Urls $probeUrls -MaxSeconds 45
    if (-not $online) {
        Write-Host "[stack] Log WebApi (dernieres lignes):"
        if (Test-Path $logPath) {
            Get-Content -Path $logPath -Tail 60 | ForEach-Object { Write-Host $_ }
        }
        if (Test-Path $errLogPath) {
            Get-Content -Path $errLogPath -Tail 60 | ForEach-Object { Write-Host $_ }
        }
        throw "WebApi non joignable après démarrage."
    }

    Write-Host "[stack] WebApi actif: $online"
    return $urls
}

Write-Host "[stack] Initialisation du stack local..."

if ($UseOllama) {
    Ensure-OllamaRunning
}

$webApiUrls = Ensure-WebApiRunning -ProjectDir $WebApiProjectDir
$preferredUrl = $webApiUrls | Where-Object { $_ -like "http://*" } | Select-Object -First 1
if (-not $preferredUrl) {
    $preferredUrl = $webApiUrls[0]
}

$env:ERP_ENDPOINTS_JSON = $EndpointsJson
$env:ERP_WEBAPI_PROJECT_DIR = $WebApiProjectDir
$env:ERP_API_BASE_URL = $preferredUrl
$env:USE_OLLAMA = if ($UseOllama) { "1" } else { "0" }
$env:OLLAMA_MODEL_ROUTER = $RouterModel
$env:OLLAMA_MODEL_ANSWER = $AnswerModel

Write-Host "[stack] Variables de session définies"
Write-Host "[stack] ERP_API_BASE_URL=$env:ERP_API_BASE_URL"
Write-Host "[stack] USE_OLLAMA=$env:USE_OLLAMA"
if ($UseOllama) {
    Write-Host "[stack] Router=$env:OLLAMA_MODEL_ROUTER, Answer=$env:OLLAMA_MODEL_ANSWER"
}

Write-Host "[stack] Lancement LangGraph..."
& $PythonExe $LangGraphScript
