@echo off
REM ============================================================================
REM  AUTONOMOUS TRADING SYSTEM - QUICK START
REM ============================================================================
REM  This script starts the complete autonomous trading system with:
REM  - Master Controller (ML-driven trading)
REM  - Monitoring Dashboard (real-time UI)
REM  
REM  Usage: start_autonomous.bat [mode] [allocation]
REM    mode: paper, shadow, canary, production (default: paper)
REM    allocation: 1-100 (default: 1)
REM  
REM  Example: start_autonomous.bat canary 5
REM ============================================================================

set MODE=%1
if "%MODE%"=="" set MODE=paper

set ALLOCATION=%2
if "%ALLOCATION%"=="" set ALLOCATION=1

echo.
echo ╔═══════════════════════════════════════════════════════════════════════════════╗
echo ║                     🤖 AUTONOMOUS TRADING SYSTEM                              ║
echo ╠═══════════════════════════════════════════════════════════════════════════════╣
echo ║                                                                               ║
echo ║  Mode:         %MODE%                                                         ║
echo ║  Allocation:   %ALLOCATION%%%                                                 ║
echo ║                                                                               ║
echo ╚═══════════════════════════════════════════════════════════════════════════════╝
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Navigate to quant-bot directory
cd /d "%~dp0"
cd quant-bot

echo [1/3] Starting Monitoring Dashboard...
start "Monitoring Dashboard" cmd /k "python src/dashboard/autonomous_monitor.py --port 8080"

echo [2/3] Waiting for dashboard to initialize...
timeout /t 3 /nobreak >nul

echo [3/3] Starting Master Controller...
start "Master Controller" cmd /k "python src/engine/master_controller.py --mode %MODE% --allocation %ALLOCATION%"

echo.
echo ============================================================================
echo  SYSTEM STARTED
echo ============================================================================
echo  Dashboard:  http://localhost:8080
echo  
echo  To stop: Close both terminal windows or use Kill Switch in dashboard
echo ============================================================================
echo.

REM Open dashboard in browser
timeout /t 2 /nobreak >nul
start http://localhost:8080

pause
