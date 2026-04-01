@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===== Chord Inference – setup a spusteni =====

REM ---- Python venv ----
if not exist .venv (
    echo Vytvarim virt prostredi...
    python -m venv .venv
)
call .venv\Scripts\activate

REM ---- Backend zavislosti ----
echo Instaluji Python zavislosti...
pip install -q -r backend\requirements-backend.txt

REM ---- Frontend zavislosti (serve staci) ----
cd frontend
if not exist node_modules\serve (
    echo Instaluji serve pro frontend...
    npm install --legacy-peer-deps
)
cd ..

REM ---- Spusteni backendu ----
echo Spoustim backend na http://localhost:40150 ...
start "backend" cmd /k "call .venv\Scripts\activate && python -m uvicorn backend.app:app --host localhost --port 40150"

REM ---- Spusteni frontendu ----
echo Spoustim frontend na http://localhost:3000 ...
cd frontend
start "frontend" cmd /k "npx serve public -l 3000"
cd ..

echo.
echo ===== vse bezii =====
echo Backend:  http://localhost:40150
echo Frontend: http://localhost:3000
echo.
