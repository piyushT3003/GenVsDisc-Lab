@echo off
title Generative vs Discriminative Model Lab
cd /d "%~dp0"
echo ============================================================
echo   Generative vs. Discriminative Model Lab - Windows launcher
echo ============================================================
echo.

set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY ( where python >nul 2>nul && set PY=python )
if not defined PY (
  echo [ERROR] Python is not installed.
  echo Install Python 3.9+ from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add Python to PATH" during installation, then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment ...
  %PY% -m venv .venv
  if errorlevel 1 ( echo [ERROR] Could not create virtual environment. & pause & exit /b 1 )
)

echo [2/3] Installing requirements (first run needs internet, ~30 MB) ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed. Check your internet connection. & pause & exit /b 1 )

echo [3/3] Starting the app - your browser will open automatically.
echo       Keep this window open. Press CTRL+C to stop.
echo.
".venv\Scripts\python.exe" app.py
pause
