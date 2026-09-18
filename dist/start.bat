@echo off
setlocal
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% -c "import numpy, sklearn, joblib, lightgbm" >nul 2>nul
if errorlevel 1 (
  echo [EatingSense] Python dependencies are missing.
  echo Install them first from this folder:
  echo     python -m pip install -r inferenceequirements.txt
  pause
  exit /b 1
)
%PY% "%~dp0inference\serve.py" --open
if errorlevel 1 pause
endlocal
