@echo off
REM ─────────────────────────────────────────────────────────────
REM run.bat  –  Start the RAG Research Assistant backend (Windows)
REM
REM Usage:
REM   run.bat          (development, hot-reload)
REM   run.bat prod     (production, 1 worker)
REM ─────────────────────────────────────────────────────────────

SET SCRIPT_DIR=%~dp0
SET PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%

echo PYTHONPATH set to: %PYTHONPATH%

IF "%1"=="prod" (
    echo Starting in PRODUCTION mode...
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --app-dir "%SCRIPT_DIR%"
) ELSE (
    echo Starting in DEVELOPMENT mode (hot-reload)...
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload --app-dir "%SCRIPT_DIR%"
)
