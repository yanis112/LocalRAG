# launch_streamlit.ps1
try {
    # Verify and activate virtual environment
    $venvPath = "C:\Users\Yanis\Documents\RAG\.venv\Scripts\Activate.ps1"
    if (-not (Test-Path $venvPath)) {
        throw "Virtual environment not found at: $venvPath"
    }
    & $venvPath

    # Change to project directory
    Set-Location "C:\Users\Yanis\Documents\RAG"

    # Set PYTHONPATH
    $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

    # Verify streamlit exists
    $streamlitPath = "$PWD\.venv\Scripts\streamlit.exe"
    if (-not (Test-Path $streamlitPath)) {
        throw "Streamlit not found at: $streamlitPath"
    }

    # Start streamlit and store the process
    $process = Start-Process -FilePath $streamlitPath -ArgumentList "run", "subprojects\movie_studio\streamlit_app_movie_studio.py" -PassThru

    # Wait for the process to complete
    $process.WaitForExit()

    # Kill any remaining streamlit processes
    Get-Process | Where-Object {$_.Name -like "*streamlit*"} | Stop-Process -Force
} catch {
    Write-Host "Error: $_"
    Exit 1
}