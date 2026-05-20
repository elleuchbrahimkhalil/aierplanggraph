param(
    [string]$PythonExe = "c:/Users/brahim/OneDrive/Bureau/example erp stage pfe/javaerp/.venv/Scripts/python.exe",
    [string]$LangGraphScript = "ai_assistant/langgraph_skeleton.py",
    [string]$EndpointsJson = "C:/Users/brahim/OneDrive/Bureau/example erp stage pfe/javaerp/ai_assistant/data/endpoints.get.json",
    [string]$WebApiProjectDir = "C:/Users/brahim/OneDrive/Bureau/stage2026/web/Webservices_webclient/stagepfe26/WebApi",
    [int]$AssistantPort = 8000,
    [string]$ExtractorModel = "deepseek-coder:6.7b",
    [string]$RouterModel = "deepseek-coder:6.7b",
    [string]$TransformerModel = "deepseek-coder:6.7b",
    [string]$AnswerModel = "deepseek-coder:6.7b",
    [int]$ExtractorTimeout = 600,
    [int]$TransformerTimeout = 300,
    [int]$RouterTimeout = 600,
    [int]$RouterCandidateLimit = 24,
    [int]$RouterColumnLimit = 24,
    [switch]$GenerateTextAnswer,
    [string]$ApiBearerToken = "",
    [string]$ApiAuthUrl = "",
    [string]$ApiUsername = "",
    [string]$ApiPassword = "",
    [string]$ApiSociete = "",
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
    }
    catch {
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
            }
            catch {
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
        }
        catch {
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

function Get-MySqlConnectionStringFromWebApiConfig {
    param([string]$ProjectDir)

    # Prefer env-style overrides if user set them
    $candidates = @(
        $env:ConnectionStrings__DefaultConnection,
        $env:ConnectionStrings__Default,
        $env:MYSQL_CONNECTION_STRING,
        $env:DB_CONNECTION_STRING
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    if ($candidates.Count -gt 0) {
        return [string]$candidates[0]
    }

    $files = @(
        (Join-Path $ProjectDir "appsettings.json"),
        (Join-Path $ProjectDir "appsettings.Development.json"),
        (Join-Path $ProjectDir "appsettings.Local.json")
    ) | Where-Object { Test-Path $_ }

    foreach ($path in $files) {
        try {
            $raw = Get-Content -Path $path -Raw -Encoding UTF8
            if ($raw.Length -gt 0 -and $raw[0] -eq [char]0xFEFF) {
                $raw = $raw.Substring(1)
            }
            $json = $raw | ConvertFrom-Json
            if ($json -and $json.ConnectionStrings) {
                foreach ($name in @('DefaultConnection', 'Default', 'MySql', 'Mysql', 'Database')) {
                    if ($json.ConnectionStrings.PSObject.Properties.Name -contains $name) {
                        $value = [string]$json.ConnectionStrings.$name
                        if (-not [string]::IsNullOrWhiteSpace($value)) {
                            return $value
                        }
                    }
                }
            }
        }
        catch {
            # ignore parse errors
        }
    }

    return ""
}

function Parse-MySqlHostPort {
    param([string]$ConnectionString)
    if ([string]::IsNullOrWhiteSpace($ConnectionString)) { return $null }

    # Common MySQL connection string keys: Server/Host/Data Source + Port
    $host = $null
    $port = $null

    $mHost = [regex]::Match($ConnectionString, "(?i)(^|;)(Server|Host|Data Source)\s*=\s*([^;]+)")
    if ($mHost.Success) {
        $host = $mHost.Groups[3].Value.Trim()
    }

    $mPort = [regex]::Match($ConnectionString, "(?i)(^|;)Port\s*=\s*([0-9]{2,5})")
    if ($mPort.Success) {
        $port = [int]$mPort.Groups[2].Value
    }

    if ([string]::IsNullOrWhiteSpace($host)) {
        return $null
    }
    if (-not $port) {
        $port = 3306
    }

    return @{
        Host = $host
        Port = $port
    }
}

function Test-MySqlConnectivity {
    param([string]$ProjectDir)

    $cs = Get-MySqlConnectionStringFromWebApiConfig -ProjectDir $ProjectDir
    if ([string]::IsNullOrWhiteSpace($cs)) {
        Write-Host "[stack] MySQL check: no connection string found (skipping)"
        return $true
    }

    $hp = Parse-MySqlHostPort -ConnectionString $cs
    if ($null -eq $hp) {
        Write-Host "[stack] MySQL check: could not parse host/port (skipping)"
        return $true
    }

    $host = [string]$hp.Host
    $port = [int]$hp.Port
    if ([string]::IsNullOrWhiteSpace($host)) { return $true }

    $ok = Test-TcpPort -HostName $host -Port $port
    if ($ok) {
        Write-Host "[stack] MySQL check: TCP OK on ${host}:${port}"
        return $true
    }

    Write-Host "[stack] MySQL check: FAILED to reach ${host}:${port}"
    Write-Host "[stack] Hint: start your MySQL server/container, or fix the WebApi connection string in appsettings.*"
    return $false
}

function Ensure-OllamaRunning {
    Write-Host "[stack] Verification Ollama..."
    if (Test-TcpPort -HostName "127.0.0.1" -Port 11434) {
        Write-Host "[stack] Ollama already active on 11434"
    }
    else {
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
        if (-not ($models -match [regex]::Escape($ExtractorModel))) {
            Write-Host "[stack] Downloading extractor model: $ExtractorModel"
            & ollama pull $ExtractorModel
        }
        if (-not ($models -match [regex]::Escape($RouterModel))) {
            Write-Host "[stack] Downloading router model: $RouterModel"
            & ollama pull $RouterModel
        }
        if (-not ($models -match [regex]::Escape($TransformerModel))) {
            Write-Host "[stack] Downloading transform model: $TransformerModel"
            & ollama pull $TransformerModel
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

function Resolve-ApiBearerToken {
    param(
        [string]$AuthUrl,
        [string]$Username,
        [string]$Password,
        [string]$Societe
    )

    if ([string]::IsNullOrWhiteSpace($AuthUrl) -or [string]::IsNullOrWhiteSpace($Username) -or [string]::IsNullOrWhiteSpace($Password)) {
        return ""
    }

    function Test-IsJwt {
        param([string]$Value)
        if ([string]::IsNullOrWhiteSpace($Value)) { return $false }
        return ($Value.Split('.').Count -eq 3 -and $Value.Length -gt 20)
    }

    function Find-TokenInObject {
        param($Obj)

        if ($null -eq $Obj) { return "" }

        if ($Obj -is [string]) {
            if (Test-IsJwt -Value $Obj) { return [string]$Obj }
            return ""
        }

        # Hashtable / IDictionary
        if ($Obj -is [System.Collections.IDictionary]) {
            foreach ($k in @('token', 'access_token', 'jwt', 'bearer')) {
                if ($Obj.Contains($k) -and ($Obj[$k] -is [string]) -and (Test-IsJwt -Value $Obj[$k])) {
                    return [string]$Obj[$k]
                }
            }
            foreach ($v in $Obj.Values) {
                $found = Find-TokenInObject -Obj $v
                if (-not [string]::IsNullOrWhiteSpace($found)) { return $found }
            }
            return ""
        }

        # Arrays / lists
        if (($Obj -is [System.Collections.IEnumerable]) -and -not ($Obj -is [string])) {
            foreach ($item in $Obj) {
                $found = Find-TokenInObject -Obj $item
                if (-not [string]::IsNullOrWhiteSpace($found)) { return $found }
            }
        }

        # PSCustomObject or other objects
        try {
            $props = $Obj.PSObject.Properties
            if ($null -ne $props) {
                foreach ($name in @('token', 'access_token', 'jwt', 'bearer')) {
                    $p = $props | Where-Object { $_.Name -eq $name } | Select-Object -First 1
                    if ($p -and ($p.Value -is [string]) -and (Test-IsJwt -Value $p.Value)) {
                        return [string]$p.Value
                    }
                }
                foreach ($p in $props) {
                    $found = Find-TokenInObject -Obj $p.Value
                    if (-not [string]::IsNullOrWhiteSpace($found)) { return $found }
                }
            }
        }
        catch {
        }

        return ""
    }

    # Expose last auth failure details to the caller (fail-fast decisions)
    $script:LastAuthErrorStatus = $null
    $script:LastAuthErrorBodyPreview = ""

    function Get-WebExceptionDetails {
        param($Exception)
        $details = @{}
        try {
            $resp = $Exception.Response
            if ($resp -and ($resp -is [System.Net.HttpWebResponse])) {
                $details.status = [int]$resp.StatusCode
                $details.statusText = [string]$resp.StatusDescription
                $stream = $resp.GetResponseStream()
                if ($stream) {
                    $reader = New-Object System.IO.StreamReader($stream)
                    $details.body = $reader.ReadToEnd()
                    $reader.Close()
                }
            }
        }
        catch {
        }
        return $details
    }

    try {
        $bodyPayload = @{ username = $Username; password = $Password }
        if (-not [string]::IsNullOrWhiteSpace($Societe)) {
            $bodyPayload.societe = $Societe
        }
        $body = $bodyPayload | ConvertTo-Json -Depth 3
        $response = Invoke-RestMethod -Uri $AuthUrl -Method Post -ContentType "application/json" -Body $body -TimeoutSec 30

        if ($null -eq $response) {
            return ""
        }

        $token = Find-TokenInObject -Obj $response
        if (-not [string]::IsNullOrWhiteSpace($token)) {
            return [string]$token
        }

        # Some APIs might return the JWT as a plain string.
        if ($response -is [string] -and (Test-IsJwt -Value $response)) {
            return [string]$response
        }

        return ""
    }
    catch {
        $ex = $_.Exception
        $details = Get-WebExceptionDetails -Exception $ex
        if ($details.ContainsKey('status')) {
            $script:LastAuthErrorStatus = $details.status
            $bodyPreview = if ($details.body) { ($details.body.Substring(0, [Math]::Min(800, $details.body.Length))) } else { "" }
            $script:LastAuthErrorBodyPreview = $bodyPreview
            Write-Host "[stack] API auth token fetch failed: HTTP $($details.status) $($details.statusText) on $AuthUrl"
            if (-not [string]::IsNullOrWhiteSpace($bodyPreview)) {
                Write-Host "[stack] API auth response body (preview): $bodyPreview"
            }
        }
        else {
            Write-Host "[stack] API auth token fetch failed: $($ex.Message)"
        }
        return ""
    }
}

Write-Host "[stack] Initializing local stack..."

# Charger les variables d'environnement depuis .env.auth si présent
$envAuthPath = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) ".env.auth"
if (Test-Path $envAuthPath) {
    Write-Host "[stack] Loading authentication from .env.auth..."
    Get-Content $envAuthPath | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line.Split("=", 2)
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            if ($key -and $value) {
                Set-Item -Path "env:$key" -Value $value
                Write-Host "[stack]   Loaded: $key"
            }
        }
    }
}

if ($UseOllama) {
    Ensure-OllamaRunning
}

$webApiUrls = Ensure-WebApiRunning -ProjectDir $WebApiProjectDir
$preferredUrl = $webApiUrls | Where-Object { $_ -like "http://*" } | Select-Object -First 1
if (-not $preferredUrl) {
    $preferredUrl = $webApiUrls[0]
}

# Fail fast if DB is unreachable; otherwise auth & API calls will 500.
$dbOk = Test-MySqlConnectivity -ProjectDir $WebApiProjectDir
if (-not $dbOk) {
    throw "WebApi database is unreachable (MySQL). Fix DB connectivity before starting the assistant."
}

$env:ERP_ENDPOINTS_JSON = $EndpointsJson
$env:ERP_ENDPOINT_SOURCE = "json"
$env:ERP_LOAD_SWAGGER_ENDPOINTS = "0"
$env:ERP_SWAGGER_JSON = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "swagger_live.json")
$env:ERP_WEBAPI_PROJECT_DIR = $WebApiProjectDir
$env:ERP_API_BASE_URL = $preferredUrl

$resolvedApiAuthUrl = if (-not [string]::IsNullOrWhiteSpace($ApiAuthUrl)) { $ApiAuthUrl } else { $env:ERP_API_AUTH_URL }
if ([string]::IsNullOrWhiteSpace($resolvedApiAuthUrl)) {
    $resolvedApiAuthUrl = "$preferredUrl/api/WebUser"
}
$resolvedApiUsername = if (-not [string]::IsNullOrWhiteSpace($ApiUsername)) { $ApiUsername } else { $env:ERP_API_USERNAME }
$resolvedApiPassword = if (-not [string]::IsNullOrWhiteSpace($ApiPassword)) { $ApiPassword } else { $env:ERP_API_PASSWORD }
$resolvedApiSociete = if (-not [string]::IsNullOrWhiteSpace($ApiSociete)) { $ApiSociete } else { $env:ERP_API_SOCIETE }

$resolvedToken = if (-not [string]::IsNullOrWhiteSpace($ApiBearerToken)) { $ApiBearerToken } else { $env:ERP_API_BEARER_TOKEN }
if ([string]::IsNullOrWhiteSpace($resolvedToken)) {
    $resolvedToken = Resolve-ApiBearerToken -AuthUrl $resolvedApiAuthUrl -Username $resolvedApiUsername -Password $resolvedApiPassword -Societe $resolvedApiSociete

    # Fail fast on common DB connectivity error so the user fixes infra first.
    if ([string]::IsNullOrWhiteSpace($resolvedToken)) {
        $preview = [string]$script:LastAuthErrorBodyPreview
        if (-not [string]::IsNullOrWhiteSpace($preview) -and ($preview -match "Unable to connect to any of the specified MySQL hosts")) {
            throw "WebApi auth failed because MySQL is unreachable. Start/fix the MySQL server (connection string) and retry."
        }
    }
}
if (-not [string]::IsNullOrWhiteSpace($resolvedToken)) {
    $env:ERP_API_BEARER_TOKEN = $resolvedToken
}

$env:ERP_ASSISTANT_PORT = "$AssistantPort"
$env:ERP_ROUTER_MODE = "columns"
$env:ERP_LOCAL_TRANSFORM = "1"
$env:OLLAMA_TIMEOUT_SECONDS = "$ExtractorTimeout"
$env:OLLAMA_ROUTER_TIMEOUT_SECONDS = "$RouterTimeout"
$env:OLLAMA_TRANSFORM_TIMEOUT_SECONDS = "$TransformerTimeout"
$env:ERP_ROUTER_CANDIDATE_LIMIT = "$RouterCandidateLimit"
$env:ERP_ROUTER_COLUMN_LIMIT = "$RouterColumnLimit"
$env:ERP_GENERATE_TEXT_ANSWER = if ($GenerateTextAnswer) { "1" } else { "0" }
$env:USE_OLLAMA = if ($UseOllama) { "1" } else { "0" }
$env:OLLAMA_MODEL_PARAM_EXTRACTOR = $ExtractorModel
$env:OLLAMA_MODEL_ROUTER = $RouterModel
$env:OLLAMA_MODEL_ANSWER = $AnswerModel
$env:OLLAMA_MODEL_TRANSFORM = $TransformerModel

Write-Host "[stack] Session variables configured"
Write-Host "[stack] ERP_API_BASE_URL=$env:ERP_API_BASE_URL"
Write-Host "[stack] ERP_API_AUTH_URL=$resolvedApiAuthUrl"
Write-Host "[stack] ERP_ENDPOINTS_JSON=$env:ERP_ENDPOINTS_JSON"
Write-Host "[stack] ERP_ENDPOINT_SOURCE=$env:ERP_ENDPOINT_SOURCE"
Write-Host "[stack] ERP_LOAD_SWAGGER_ENDPOINTS=$env:ERP_LOAD_SWAGGER_ENDPOINTS"
Write-Host "[stack] ERP_SWAGGER_JSON=$env:ERP_SWAGGER_JSON"
Write-Host "[stack] ERP_ASSISTANT_PORT=$env:ERP_ASSISTANT_PORT"
Write-Host "[stack] ERP_ROUTER_MODE=$env:ERP_ROUTER_MODE"
Write-Host "[stack] ERP_LOCAL_TRANSFORM=$env:ERP_LOCAL_TRANSFORM"
if (-not [string]::IsNullOrWhiteSpace($env:ERP_API_BEARER_TOKEN)) {
    Write-Host "[stack] ERP_API_BEARER_TOKEN is configured"
}
else {
    Write-Host "[stack] ERP_API_BEARER_TOKEN is empty"
}
Write-Host "[stack] OLLAMA_TIMEOUT_SECONDS=$env:OLLAMA_TIMEOUT_SECONDS"
Write-Host "[stack] OLLAMA_ROUTER_TIMEOUT_SECONDS=$env:OLLAMA_ROUTER_TIMEOUT_SECONDS"
Write-Host "[stack] ERP_ROUTER_CANDIDATE_LIMIT=$env:ERP_ROUTER_CANDIDATE_LIMIT"
Write-Host "[stack] ERP_ROUTER_COLUMN_LIMIT=$env:ERP_ROUTER_COLUMN_LIMIT"
Write-Host "[stack] ERP_GENERATE_TEXT_ANSWER=$env:ERP_GENERATE_TEXT_ANSWER"
Write-Host "[stack] USE_OLLAMA=$env:USE_OLLAMA"
if ($UseOllama) {
    Write-Host "[stack] Extractor=$env:OLLAMA_MODEL_PARAM_EXTRACTOR, Router=$env:OLLAMA_MODEL_ROUTER, Transform=$env:OLLAMA_MODEL_TRANSFORM, Answer=$env:OLLAMA_MODEL_ANSWER"
}

Write-Host "[stack] Starting LangGraph HTTP server..."
& $PythonExe $LangGraphScript --serve --host 127.0.0.1 --port $AssistantPort
