"""Descarga data/senales_norm_f16.npy (390 MB), el único archivo que no cabe en GitHub.

Uso (desde la carpeta del proyecto):
    python scripts/descargar_datos.py

Si cambias de lugar el archivo, edita la lista URLS. Se prueban en orden.
"""
import sys
import urllib.request
from pathlib import Path

URLS = [
    # 1) GitHub Releases del repositorio (ajusta usuario/repositorio si es otro)
    "https://github.com/IEstremadoyro/ptbxl-visual-analytics/releases/download/v1.0/senales_norm_f16.npy",
    # 2) Alternativa: enlace de descarga directa de Google Drive (pega aquí el tuyo)
    # "https://drive.google.com/uc?export=download&id=ID_DEL_ARCHIVO",
]
DESTINO = Path(__file__).resolve().parent.parent / "data" / "senales_norm_f16.npy"
TAM_ESPERADO = 389_856_128  # bytes


def barra(bloques, tam_bloque, total):
    hecho = bloques * tam_bloque
    if total > 0:
        pct = min(100, hecho * 100 // total)
        sys.stdout.write(f"\r  {hecho / 1e6:6.1f} MB de {total / 1e6:.1f} MB ({pct} %)")
        sys.stdout.flush()


def main():
    if DESTINO.exists() and DESTINO.stat().st_size == TAM_ESPERADO:
        print("Ya existe data/senales_norm_f16.npy con el tamaño correcto. No hay nada que descargar.")
        return 0
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        print("Descargando desde", url)
        try:
            urllib.request.urlretrieve(url, DESTINO, barra)
            print()
            if DESTINO.stat().st_size != TAM_ESPERADO:
                print(f"  Tamaño inesperado ({DESTINO.stat().st_size} bytes); se prueba la siguiente fuente.")
                continue
            print("Listo:", DESTINO)
            return 0
        except Exception as e:
            print("\n  No se pudo:", e)
    print("No se pudo descargar el archivo. Revisa el README (sección «Datos grandes»).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
