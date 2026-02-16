@echo off
REM ============================================
REM AITradingBot - One-Click Start Script
REM ============================================

title AITradingBot - Live Trading Bot

REM Change to project directory (where this batch file is located)
cd /d "%~dp0"

REM Display startup message
echo ============================================
echo   AITradingBot - Starting Trading Bot
echo ============================================
echo.
echo Project Directory: %CD%
echo Starting bot...
echo.
echo Press Ctrl+C to stop the bot gracefully
echo ============================================
echo.

REM Start the bot
python runner\phase26_realtime_live.py

REM If bot exits, pause so user can see any error messages
if errorlevel 1 (
    echo.
    echo ============================================
    echo   Bot exited with an error!
    echo ============================================
    pause
) else (
    echo.
    echo ============================================
    echo   Bot stopped normally
    echo ============================================
    pause
)
