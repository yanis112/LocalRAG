$venvPath = "C:\Users\Yanis\Documents\RAG\venv\Scripts\Activate.ps1"
. $venvPath

Set-Location "C:\Users\Yanis\Documents\RAG"
$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"

Start-Process -NoNewWindow streamlit -ArgumentList "run", "scripts\streamlit_app.py"

$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {
    $attempt++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501" -Method Head -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            break
        }
    }
    catch {
        Start-Sleep -Seconds 2
        if ($attempt -eq $maxAttempts) {
            Write-Host "Impossible de se connecter au serveur après $maxAttempts tentatives"
            exit
        }
    }
}