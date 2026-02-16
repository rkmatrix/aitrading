@echo off
REM ============================================
REM AITradingBot - Debug Mode Start Script
REM ============================================

title AITradingBot - Debug Mode

REM Change to project directory
cd /d "%~dp0"

REM Display startup message
echo ============================================
echo   AITradingBot - Starting in DEBUG Mode
echo ============================================
echo.
echo Project Directory: %CD%
echo Debug mode enabled - more verbose output
echo.
echo Press Ctrl+C to stop the bot gracefully
echo ============================================
echo.

REM Set debug environment variable
set FLASK_ENV=development

REM Start the bot
python runner\phase26_realtime_live.py

REM Pause on exit
pause
