param(
    [string]$PythonExe = "c:/Users/brahim/OneDrive/Bureau/example erp stage pfe/javaerp/.venv/Scripts/python.exe",
    [string]$LangGraphScript = "ai_assistant/langgraph_skeleton.py",
    [string]$EndpointsJson = "C:/Users/brahim/OneDrive/Bureau/example erp stage pfe/javaerp/ai_assistant/data/endpoints.get.json",
    [string]$WebApiProjectDir = "C:/Users/brahim/OneDrive/Bureau/stage2026/web/Webservices_webclient/stagepfe26/WebApi",
    [int]$AssistantPort = 8000,
    [string]$RouterModel = "deepseek-coder:6.7b",
    [string]$AnswerModel = "llama3.2:latest",
    [switch]$UseOllama,
    [switch]$AutoPullModels
)

$ErrorActionPreference = "Stop"

function Test-TcpPort {
    param([string]$HostName, [int]$Port)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect($HostName, $Port, $null, $null)
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
            }
        }
        Start-Sleep -Milliseconds 900
    }
    return $null
}

function Get-HostPortsFromUrls {
    param([string[]]$Urls)
    $pairs = @()
    foreach ($u in $Urls) {
        try {
            $uri = [System.Uri]$u
            $key = "$($uri.Host):$($uri.Port)"
            if (-not ($pairs -contains $key)) {
                $pairs += $key
            }
        } catch {
        }
    }
    return $pairs
}

function Wait-TcpUp {
    param([string[]]$HostPorts, [int]$MaxSeconds = 45)
    $deadline = (Get-Date).AddSeconds($MaxSeconds)
    while ((Get-Date) -lt $deadline) {
        foreach ($hp in $HostPorts) {
            $parts = $hp.Split(":")
            if ($parts.Count -ne 2) {
                continue
            }
            $hostName = $parts[0]
            $port = [int]$parts[1]
            if (Test-TcpPort -HostName $hostName -Port $port) {
                return $hp
            }
        }
        Start-Sleep -Milliseconds 800
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
    Write-Host "[stack] Verification Ollama..."
    if (Test-TcpPort -HostName "127.0.0.1" -Port 11434) {
        Write-Host "[stack] Ollama already active on 11434"
    } else {
        $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
        if (-not $ollamaCmd) {
            throw "Ollama is not installed or not present in PATH."
        }

        Write-Host "[stack] Starting Ollama server..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Minimized | Out-Null

        $up = Wait-HttpUp -Urls @("http://127.0.0.1:11434/api/tags") -MaxSeconds 25
        if (-not $up) {
            throw "Unable to start Ollama server on 11434."
        }
        Write-Host "[stack] Ollama active"
    }

    if ($AutoPullModels) {
        Write-Host "[stack] Checking Ollama models..."
        $models = (& ollama list) 2>$null
        if (-not ($models -match [regex]::Escape($RouterModel))) {
            Write-Host "[stack] Downloading router model: $RouterModel"
            & ollama pull $RouterModel
        }
        if (-not ($models -match [regex]::Escape($AnswerModel))) {
            Write-Host "[stack] Downloading answer model: $AnswerModel"
            & ollama pull $AnswerModel
        }
    }
}

function Ensure-WebApiRunning {
    param([string]$ProjectDir)

    if (-not (Test-Path $ProjectDir)) {
        throw "WebApi project not found: $ProjectDir"
    }

    $urls = Get-WebApiUrlsFromLaunchSettings -ProjectDir $ProjectDir
    if ($urls.Count -eq 0) {
        throw "No applicationUrl found in launchSettings.json"
    }

    Write-Host "[stack] WebApi URLs detected: $($urls -join ', ')"

    $hostPorts = Get-HostPortsFromUrls -Urls $urls
    $online = Wait-TcpUp -HostPorts $hostPorts -MaxSeconds 2
    if ($online) {
        Write-Host "[stack] WebApi already active on: $online"
        return $urls
    }

    $dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
    if (-not $dotnet) {
        throw "dotnet SDK not found in PATH."
    }

    $logPath = Join-Path $ProjectDir "webapi-run.log"
    $errLogPath = Join-Path $ProjectDir "webapi-run.err.log"
    Write-Host "[stack] Starting WebApi..."
    Start-Process -FilePath "dotnet" -ArgumentList @("run", "--launch-profile", "WebApi") -WorkingDirectory $ProjectDir -WindowStyle Minimized -RedirectStandardOutput $logPath -RedirectStandardError $errLogPath | Out-Null

    $online = Wait-TcpUp -HostPorts $hostPorts -MaxSeconds 120
    if (-not $online) {
        Write-Host "[stack] WebApi log (last lines):"
        if (Test-Path $logPath) {
            Get-Content -Path $logPath -Tail 60 | ForEach-Object { Write-Host $_ }
        }
        if (Test-Path $errLogPath) {
            Get-Content -Path $errLogPath -Tail 60 | ForEach-Object { Write-Host $_ }
        }
        throw "WebApi unreachable after startup."
    }

    Write-Host "[stack] WebApi active on: $online"
    return $urls
}

Write-Host "[stack] Initializing local stack..."

if ($UseOllama) {
    Ensure-OllamaRunning
}

$webApiUrls = Ensure-WebApiRunning -ProjectDir $WebApiProjectDir
$preferredUrl = $webApiUrls | Where-Object { $_ -like "http://*" } | Select-Object -First 1
if (-not $preferredUrl) {
    $preferredUrl = $webApiUrls[0]
}

$env:ERP_ENDPOINTS_JSON = $EndpointsJson
$env:ERP_ENDPOINT_SOURCE = "json"
$env:ERP_LOAD_SWAGGER_ENDPOINTS = "0"
$env:ERP_SWAGGER_JSON = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "swagger_live.json")
$env:ERP_WEBAPI_PROJECT_DIR = $WebApiProjectDir
$env:ERP_API_BASE_URL = $preferredUrl
$env:ERP_ASSISTANT_PORT = "$AssistantPort"
$env:OLLAMA_TIMEOUT_SECONDS = "180"
$env:OLLAMA_ROUTER_TIMEOUT_SECONDS = "300"
$env:USE_OLLAMA = if ($UseOllama) { "1" } else { "0" }
$env:OLLAMA_MODEL_ROUTER = $RouterModel
$env:OLLAMA_MODEL_ANSWER = $AnswerModel

Write-Host "[stack] Session variables configured"
Write-Host "[stack] ERP_API_BASE_URL=$env:ERP_API_BASE_URL"
Write-Host "[stack] ERP_ENDPOINTS_JSON=$env:ERP_ENDPOINTS_JSON"
Write-Host "[stack] ERP_ENDPOINT_SOURCE=$env:ERP_ENDPOINT_SOURCE"
Write-Host "[stack] ERP_LOAD_SWAGGER_ENDPOINTS=$env:ERP_LOAD_SWAGGER_ENDPOINTS"
Write-Host "[stack] ERP_SWAGGER_JSON=$env:ERP_SWAGGER_JSON"
Write-Host "[stack] ERP_ASSISTANT_PORT=$env:ERP_ASSISTANT_PORT"
Write-Host "[stack] OLLAMA_TIMEOUT_SECONDS=$env:OLLAMA_TIMEOUT_SECONDS"
Write-Host "[stack] OLLAMA_ROUTER_TIMEOUT_SECONDS=$env:OLLAMA_ROUTER_TIMEOUT_SECONDS"
Write-Host "[stack] USE_OLLAMA=$env:USE_OLLAMA"
if ($UseOllama) {
    Write-Host "[stack] Router=$env:OLLAMA_MODEL_ROUTER, Answer=$env:OLLAMA_MODEL_ANSWER"
}

Write-Host "[stack] Starting LangGraph HTTP server..."
& $PythonExe $LangGraphScript --serve --host 127.0.0.1 --port $AssistantPort
