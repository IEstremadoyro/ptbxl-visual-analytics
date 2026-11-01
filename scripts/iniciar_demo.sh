#!/bin/bash
# Plataforma completa para el video / demostración (solo en su Mac, sin contraseña).
# Busca solo un puerto libre desde 8010 (o use:  bash scripts/iniciar_demo.sh 8050)
cd "$(dirname "$0")/.." || exit 1
[ -d venv ] && source venv/bin/activate
PUERTO=$(python3 scripts/_puerto_libre.py "${1:-8010}")
echo "=============================================="
echo " Plataforma corriendo en: http://localhost:$PUERTO"
echo " Ctrl+C para cerrar."
echo "=============================================="
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port "$PUERTO"
