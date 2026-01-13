@echo off
REM Start Dashboard from project root
cd /d %~dp0\..
python -m dashboard.app
