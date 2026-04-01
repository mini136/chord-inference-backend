@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===== Chord Inference – setup a spuštění =====

REM ---- Python venv ----
if not exist .venv (
    echo Vytvářím virtuální prostředí...
    python -m venv .venv
)
call .venv\Scripts\activate

REM ---- Backend závislosti ----
echo Instaluji Python závislosti...
pip install -q -r backend\requirements-backend.txt

REM ---- Frontend závislosti (serve stačí) ----
cd frontend
if not exist node_modules\serve (
    echo Instaluji serve pro frontend...
    npm install --legacy-peer-deps
)
cd ..

REM ---- Spuštění backendu ----
echo Spouštím backend na http://localhost:40150 ...
start "backend" cmd /k "call .venv\Scripts\activate && python -m uvicorn backend.app:app --host localhost --port 40150"

REM ---- Spuštění frontendu ----
echo Spouštím frontend na http://localhost:3000 ...
cd frontend
start "frontend" cmd /k "npx serve public -l 3000"
cd ..

echo.
echo ===== Vše běží =====
echo Backend:  http://localhost:40150
echo Frontend: http://localhost:3000
echo.
