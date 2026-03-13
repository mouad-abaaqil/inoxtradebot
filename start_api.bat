@echo off
REM ============================================================
REM  INOXTRADE — Start API server in background
REM  This bat file is registered in Windows Task Scheduler
REM  to run at user logon.
REM ============================================================

cd /d "%~dp0"

echo [%date% %time%] Starting INOXTRADE API server... >> logs\api_start.log

REM Launch the API server detached (no visible window)
start /B "" python api\server.py >> logs\api.log 2>&1

echo [%date% %time%] API server launched (PID pending) >> logs\api_start.log
