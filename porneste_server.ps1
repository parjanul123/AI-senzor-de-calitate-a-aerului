$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$env:CHATBOT_USE_OLLAMA = "true"
$env:OLLAMA_MODEL = "gpt-oss:20b-cloud"

if (Test-Path ".venv\Scripts\python.exe") {
    $pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
} else {
    $pythonExecutable = "python"
}

if ($null -eq (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama nu este instalat sau nu este disponibil in PATH. Instaleaza-l de la https://ollama.com/download."
}

$ollamaHealthUrl = "http://127.0.0.1:11434/api/tags"
try {
    Invoke-WebRequest -Uri $ollamaHealthUrl -TimeoutSec 1 -UseBasicParsing | Out-Null
    Write-Host "Ollama ruleaza deja."
} catch {
    Write-Host "Pornesc Ollama..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WorkingDirectory $projectRoot | Out-Null

    $ollamaReady = $false
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        try {
            Invoke-WebRequest -Uri $ollamaHealthUrl -TimeoutSec 1 -UseBasicParsing | Out-Null
            $ollamaReady = $true
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    if (-not $ollamaReady) {
        throw "Ollama nu a pornit pe portul 11434. Verifica instalarea si jurnalul Ollama."
    }
}

Write-Host "Pornesc API-ul FastAPI..."
$apiProcess = Start-Process -FilePath $pythonExecutable -ArgumentList "-m", "uvicorn", "app.api.main:app", "--reload" -WorkingDirectory $projectRoot -PassThru

$healthUrl = "http://127.0.0.1:8000/health"
$apiReady = $false
for ($attempt = 1; $attempt -le 30; $attempt++) {
    try {
        $healthResponse = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 1 -UseBasicParsing
        if ($healthResponse.StatusCode -eq 200) {
            $apiReady = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $apiReady) {
    if (-not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force
    }
    throw "API-ul FastAPI nu a pornit pe portul 8000. Verifica mesajele Uvicorn si daca portul este deja utilizat."
}

Write-Host "Pornesc aplicatia Streamlit..."
& $pythonExecutable -m streamlit run streamlit_app/main.py

# Pornire din PowerShell: .\porneste_server.ps1
.\porneste_server.ps1