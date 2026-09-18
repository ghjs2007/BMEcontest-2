@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo [EatingSense] Python was not found on PATH.
  echo Install Python 3.11+ from https://www.python.org/downloads/ and run this file again.
  pause
  exit /b 1
)
%PY% -c "import numpy, sklearn, joblib, lightgbm" >nul 2>nul
if errorlevel 1 (
  echo [EatingSense] Python dependencies are missing.
  set /p ANS="Install them now from inferenceequirements.txt? [Y/N] "
  if /i not "!ANS!"=="Y" (
    echo Install them first from this folder:
    echo     python -m pip install -r inferenceequirements.txt
    pause
    exit /b 1
  )
  %PY% -m pip install --disable-pip-version-check -r "inferenceequirements.txt"
  if errorlevel 1 (
    echo [EatingSense] Dependency installation failed - see the pip output above.
    pause
    exit /b 1
  )
)
%PY% "%~dp0inference\serve.py" --open
if errorlevel 1 pause
endlocal
