@echo on
chcp 65001 >nul
cd /d "%~dp0"

echo ===== Chord Inference - setup a spusteni =====
echo.

REM ---- Python venv ----
if not exist .venv (
    echo [1/4] Vytvarim virtualni prostredi .venv ...
    python -m venv .venv
) else (
    echo [1/4] .venv uz existuje, preskakuji.
)

echo [2/4] Aktivuji .venv ...
call .venv\Scripts\activate

REM ---- Backend zavislosti ----
echo [3/4] Instaluji Python zavislosti z backend\requirements-backend.txt ...
pip install -r backend\requirements-backend.txt

REM ---- Frontend zavislosti ----
cd frontend
if not exist node_modules\serve (
    echo [4/4] Instaluji npm zavislosti pro frontend ...
    npm install --legacy-peer-deps
) else (
    echo [4/4] npm zavislosti uz nainstalovane, preskakuji.
)
cd ..

echo.
echo ===== Setup hotov, spoustim servery =====
echo.

REM ---- Spusteni backendu ----
echo Spoustim BACKEND na http://localhost:40150 ...
start "backend" cmd /k "call .venv\Scripts\activate && python -m uvicorn backend.app:app --host localhost --port 40150"

REM ---- Spusteni frontendu ----
echo Spoustim FRONTEND na http://localhost:3000 ...
cd frontend
start "frontend" cmd /k "npx serve public -l 3000"
cd ..

echo.
echo =============================================
echo   Backend:  http://localhost:40150
echo   Frontend: http://localhost:3000
echo =============================================
echo.
pause
