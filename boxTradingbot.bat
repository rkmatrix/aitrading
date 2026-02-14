@echo off
REM ============================================================================
REM Box Trading Bot Launcher
REM ============================================================================
REM 
REM This script launches the Box Trading Bot with proper error handling
REM and environment setup.
REM 
REM Usage: 
REM   - Double-click to start the bot
REM   - Press Ctrl+C to stop the bot
REM 
REM Requirements:
REM   - Python 3.8+ installed
REM   - .env file configured with API keys
REM   - ENV variable set to PAPER_TRADING or LIVE in .env
REM ============================================================================

title Box Trading Bot - Automated Mean Reversion Strategy

REM Change to script directory (project root)
cd /d "%~dp0"

echo ============================================================================
echo   BOX TRADING BOT
echo ============================================================================
echo.
echo   Current Directory: %CD%
echo   Timestamp: %DATE% %TIME%
echo.
echo ============================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or higher from https://www.python.org/
    echo.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo [ERROR] .env file not found
    echo.
    echo Please create a .env file with your configuration.
    echo Copy .env.example to .env and fill in your API keys.
    echo.
    pause
    exit /b 1
)

REM Check if config file exists
if not exist "configs\box_trading.yaml" (
    echo [ERROR] Configuration file not found: configs\box_trading.yaml
    echo.
    echo Please ensure the box trading configuration file exists.
    echo.
    pause
    exit /b 1
)

echo [INFO] Starting Box Trading Bot...
echo [INFO] Press Ctrl+C to stop the bot
echo.
echo ============================================================================
echo.

REM Run the box trading bot
python runner\box_trading_runner.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo ============================================================================
    echo [ERROR] Bot exited with an error!
    echo ============================================================================
    echo.
    echo Check the log file for details: data\logs\box_trading_bot.log
    echo.
    echo Common issues:
    echo   - API keys not configured in .env
    echo   - Network connection issues
    echo   - Invalid configuration in box_trading.yaml
    echo   - Market is closed (bot will idle)
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ============================================================================
    echo [INFO] Bot stopped normally
    echo ============================================================================
    echo.
    echo Check the log file for session details: data\logs\box_trading_bot.log
    echo.
    pause
)
