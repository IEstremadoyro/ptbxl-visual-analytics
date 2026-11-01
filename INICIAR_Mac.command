#!/bin/bash
# Doble clic para abrir la plataforma PTB-XL en Mac (la primera vez instala lo necesario, ~1 min).
cd "$(dirname "$0")" || exit 1
PY=$(command -v python3)
if [ -z "$PY" ]; then echo "Falta Python 3. Instálalo desde https://www.python.org/downloads/ y vuelve a abrir este archivo."; read -p "Enter para cerrar"; exit 1; fi
if [ ! -f data/senales_norm_f16.npy ]; then
  echo "Falta data/senales_norm_f16.npy; descargándolo (390 MB)..."
  "$PY" scripts/descargar_datos.py || { echo "No se pudo descargar. Revisa tu conexión."; read -p "Enter para cerrar"; exit 1; }
fi
if [ ! -d .venv ]; then
  echo "Preparando el entorno (solo la primera vez)..."
  "$PY" -m venv .venv && ./.venv/bin/pip install -q -r requirements.txt || { echo "Falló la instalación."; read -p "Enter para cerrar"; exit 1; }
fi
PUERTO=$(./.venv/bin/python scripts/_puerto_libre.py 8010)
echo "=============================================="
echo " Plataforma en: http://localhost:$PUERTO"
echo " Cierra esta ventana (o Ctrl+C) para detenerla."
echo "=============================================="
( sleep 4; open "http://localhost:$PUERTO" ) &
./.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port "$PUERTO"
