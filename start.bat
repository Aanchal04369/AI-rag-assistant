@echo off
title AI Research Assistant - Quick Start
echo =========================================
echo      AI Research Assistant + NLP + LLM
echo =========================================
echo.

:: ── Check Python ─────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
  echo Python 3.10 or later is required but was not found on PATH.
  echo Install Python 3.10+ from https://www.python.org/downloads/
  pause
  exit /b 1
)

:: ── Check Node.js ────────────────────────────────────────────────────
node --version >nul 2>&1
if errorlevel 1 (
  echo Node.js 18 or later is required but was not found on PATH.
  echo Install Node.js 18+ from https://nodejs.org/
  pause
  exit /b 1
)

echo [1/4] Setting up Python virtual environment...
cd backend
if not exist venv python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/4] Installing backend dependencies...
pip install -r requirements.txt
if errorlevel 1 (
  echo Backend dependency installation failed. Check the error above.
  pause
  exit /b 1
)
cd ..

echo.
echo [3/4] Setting up React frontend...
cd frontend
if not exist node_modules (
  call npm install
  if errorlevel 1 (
    echo Frontend dependency installation failed. Check the error above.
    cd ..
    pause
    exit /b 1
  )
)
cd ..

echo.
echo [4/4] Launching backend and frontend...

cd backend
call venv\Scripts\activate.bat
start "AI Research Assistant - Backend" cmd /k python app.py
cd ..

timeout /t 3 /nobreak >nul

cd frontend
start "AI Research Assistant - Frontend" cmd /k npm start
cd ..

echo.
echo =========================================
echo   App:     http://localhost:3000
echo   Backend: http://localhost:5000
echo   LLM:     Gemini 2.5 Flash (cloud)
echo   NLP:     NLTK pipeline (local)
echo   BERT:    DistilBERT fine-tuning (local)
echo =========================================
echo.
echo Both windows will stay open even if a process exits.
echo Close this window when you are done.
pause
