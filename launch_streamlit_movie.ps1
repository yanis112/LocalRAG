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

    # Run streamlit
    & $streamlitPath run subprojects\movie_studio\streamlit_app_movie_studio.py
} catch {
    Write-Host "Error: $_"
    Read-Host "Press Enter to exit"
}