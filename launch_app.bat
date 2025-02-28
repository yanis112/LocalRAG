@echo off
echo ========= LANCEMENT DE L'APPLICATION RAG =========
echo.

REM Change to project root directory
cd /d "C:\Users\Yanis\Documents\RAG"
echo Repertoire courant: %CD%
echo.

REM Activate uv virtual environment
call .\.venv\Scripts\activate
echo Environnement virtuel active
echo.

REM Nettoyer les anciens serveurs potentiellement en cours d'exécution
echo Nettoyage des processus en cours...
taskkill /f /im uvicorn.exe >nul 2>&1
echo.

REM Launch backend server in new window with visible terminal
echo Demarrage du serveur backend...
start "RAG Backend Server" cmd /k "cd backend && title Backend - UVICORN SERVER && color 0A && echo ==== BACKEND SERVER - LOGS ==== && echo. && uvicorn main:app --reload --port 8000 --log-level debug"

REM Wait a moment to ensure backend starts first
echo Attente du demarrage du serveur backend...
timeout /t 5
echo.

REM Launch frontend server in new window with visible terminal
echo Demarrage du serveur frontend...
start "RAG Frontend Server" cmd /k "cd frontend && title Frontend Server && color 09 && echo ==== FRONTEND SERVER - LOGS ==== && npm run dev"

REM Wait a few seconds for servers to start
echo Attente du demarrage des serveurs...
timeout /t 5
echo.

REM Open index.html in default browser
echo Ouverture du navigateur...
start "" "frontend/index.html"

echo.
echo ========= APPLICATION LANCEE AVEC SUCCES =========
echo Le backend s'execute sur http://127.0.0.1:8000
echo Les journaux du backend sont visibles dans la fenetre "RAG Backend Server" avec le fond vert
echo Les journaux du frontend sont visibles dans la fenetre "RAG Frontend Server" avec le fond bleu
echo.
echo Pour arreter les serveurs, fermez simplement les fenetres des terminaux.
echo.
echo Appuyez sur une touche pour fermer cette fenetre...
pause > nul