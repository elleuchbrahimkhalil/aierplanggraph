$body = @{ question = 'affiche les ventes par article' } | ConvertTo-Json
$response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/assistant/query' -Method Post -ContentType 'application/json' -Body $body
$response | ConvertTo-Json -Depth 12
