@echo off
setlocal

set PYTHONPATH=C:\Users\Yanis\Documents\RAG
cd /d "C:\Users\Yanis\Documents\RAG"

:: Start the Streamlit app and wait for it to complete
poetry run streamlit run scripts/streamlit_app.py

:: Ensure all related processes are terminated
for /f "tokens=2 delims=," %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH') do taskkill /PID %%i /F

endlocal