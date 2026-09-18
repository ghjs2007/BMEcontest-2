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
rem Locate the local inference bridge entry (dist layout: inference\serve.py; submission layout: serve.py).
set "SERVE="
if exist "%~dp0inference\serve.py" set "SERVE=%~dp0inference\serve.py"
if not defined SERVE if exist "%~dp0serve.py" set "SERVE=%~dp0serve.py"
if not defined SERVE (
  echo [EatingSense] serve.py was not found next to this launcher.
  pause
  exit /b 1
)
rem Locate the pinned requirements file and a launcher-relative name for display.
set "REQ="
set "REQREL="
if exist "%~dp0inference\requirements.txt" (
  set "REQ=%~dp0inference\requirements.txt"
  set "REQREL=inference\requirements.txt"
)
if not defined REQ if exist "%~dp0requirements.txt" (
  set "REQ=%~dp0requirements.txt"
  set "REQREL=requirements.txt"
)
%PY% -c "import numpy, sklearn, joblib, lightgbm" >nul 2>nul
if errorlevel 1 (
  echo [EatingSense] Python dependencies are missing.
  set /p ANS="Install them now from !REQREL!? [Y/N] "
  if /i not "!ANS!"=="Y" (
    echo Install them first with:
    echo     python -m pip install -r !REQREL!
    pause
    exit /b 1
  )
  %PY% -m pip install --disable-pip-version-check -r "!REQ!"
  if errorlevel 1 (
    echo [EatingSense] Dependency installation failed - see the pip output above.
    pause
    exit /b 1
  )
)
%PY% "%SERVE%" --open
if errorlevel 1 pause
endlocal
