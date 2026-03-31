@echo off
REM Aktivace virtuálního prostředí
call .venv\Scripts\activate

REM Instalace závislostí backendu
pip install -r app\backend\requirements-backend.txt

REM Instalace závislostí frontend (pokud je potřeba)
cd app\frontend
if exist node_modules (
    echo Node modules already installed.
) else (
    npm install
)
cd ..\..

REM Spuštění backendu
start "backend" cmd /k "python -m uvicorn app.backend.app:app --host localhost --port 40150"

REM Spuštění frontendu
cd app\frontend
start "frontend" cmd /k "npx serve public -l 3000"
cd ..\..

echo Backend běží na http://localhost:40150
