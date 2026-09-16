@echo off
title Informe Inversiones 6m
cd /d "%~dp0"
set PYTHONUTF8=1
set PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\AppData\Roaming\npm;%PATH%

echo [1/3] Actualizando datos de mercado...
python update_universe_data.py --refresh >> logs\informe_semanal.log 2>&1

echo [2/3] Ejecutando analisis completo (horizonte 6m)...
python run_analysis_headless.py --horizon 6m >> logs\informe_semanal.log 2>&1

echo [3/3] Generando informe con tracking...
python weekly_report.py --open >> logs\informe_semanal.log 2>&1

echo Listo. Informe en la carpeta "informes\".
