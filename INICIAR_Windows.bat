@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Plataforma PTB-XL (la primera vez instala lo necesario, ~1 min)
where python >nul 2>nul
if errorlevel 1 (
  echo Falta Python 3. Instalalo desde https://www.python.org/downloads/ marcando "Add python.exe to PATH" y vuelve a abrir este archivo.
  pause
  exit /b 1
)
if not exist data\senales_norm_f16.npy (
  echo Falta data\senales_norm_f16.npy; descargando 390 MB...
  python scripts\descargar_datos.py
  if errorlevel 1 ( echo No se pudo descargar. & pause & exit /b 1 )
)
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\pip install -q -r requirements.txt
  if errorlevel 1 ( echo Fallo la instalacion. & pause & exit /b 1 )
)
for /f %%p in ('.venv\Scripts\python scripts\_puerto_libre.py 8010') do set PUERTO=%%p
echo ==============================================
echo  Plataforma en: http://localhost:%PUERTO%
echo  Cierra esta ventana para detenerla.
echo ==============================================
start "" cmd /c "timeout /t 4 >nul & start http://localhost:%PUERTO%"
.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port %PUERTO%
