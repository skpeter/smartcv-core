@echo off
setlocal
set "HERE=%~dp0"
if exist "%HERE%core\core.py" (
  cd /d "%HERE%"
) else if exist "%HERE%core.py" (
  cd /d "%HERE%.."
) else (
  echo SmartCV: cannot find project root ^(core\core.py^).
  pause
  exit /b 1
)

python -m core.core
pause
