@echo off
title Investment Report GUI - N26 Portfolio Analyzer
cd /d "%~dp0"

:: Añadir rutas donde Claude Code suele instalarse
set PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\AppData\Roaming\npm;%PATH%

echo.
echo ========================================
echo   Investment Report GUI
echo   N26 Portfolio Analyzer
echo ========================================
echo.
python investment_report_gui.py
if errorlevel 1 (
    echo.
    echo ERROR: La aplicacion termino con error.
    pause
)
