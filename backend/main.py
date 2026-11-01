"""
PTB-XL Visual Analytics v3 - backend FastAPI.

Archivos de datos (los genera el notebook v3.1; copiarlos a data/):
    ptbxl_predictivo_umap.csv, senales_norm_f16.npy, norm_stats.npz, cooccurrence.json
Frontend (static/): index.html, app.js, style.css

Ejecutar desde la carpeta del proyecto:
    python3 -m uvicorn backend.main:app --reload --reload-dir backend
(o  uvicorn main:app --reload  si main.py está en la raíz)

No necesita torch: predicciones, vecinos y señales ya vienen calculados.
Opcional: data/interpretabilidad.csv (lo genera scripts/generar_interpretabilidad.py) agrega la distancia a
cada vecino, el voto del kNN con tres distancias y los grupos de K-means y DBSCAN.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent   # carpeta de main.py (p. ej. backend/)
RAIZ = BASE.parent                       # carpeta del proyecto


def _primero(candidatos, marca):
    """Primera carpeta candidata que contiene el archivo `marca`."""
    for c in candidatos:
        if (c / marca).exists():
            return c
    raise FileNotFoundError(
        f"No encuentro '{marca}'. Busqué en:\n" + "\n".join(f"  - {c}" for c in candidatos)
    )


DATA = _primero([BASE / "data", RAIZ / "data", BASE, RAIZ], "ptbxl_predictivo_umap.csv")
STATIC = _primero(
    [BASE / "static", RAIZ / "static", RAIZ / "frontend", BASE / "frontend", RAIZ / "frontend" / "static"],
    "index.html",
)

print(f"[PTB-XL] datos:     {DATA}")
print(f"[PTB-XL] estáticos: {STATIC}")
for _f in ("app.js", "style.css"):
    if not (STATIC / _f).exists():
        print(f"[PTB-XL] AVISO: falta {STATIC / _f}. Sin ese archivo la página se ve sin estilos o sin datos.")
if not (DATA / "cooccurrence.json").exists():
    print(f"[PTB-XL] AVISO: falta {DATA / 'cooccurrence.json'}; la matriz de coocurrencia (vista B) quedará vacía.")

CLASES = ["NORM", "MI", "STTC", "CD", "HYP"]
CL2I = {c: i for i, c in enumerate(CLASES)}
SPLITS = ["train", "val", "test"]
DERIVACIONES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100

app = FastAPI(title="PTB-XL Visual Analytics")
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ---------------------------------------------------------------- datos
df = pd.read_csv(DATA / "ptbxl_predictivo_umap.csv")
SENALES = np.load(DATA / "senales_norm_f16.npy", mmap_mode="r")   # (N, 12, 1000) z-score
assert len(df) == SENALES.shape[0], "CSV y senales_norm_f16.npy desalineados"

if (DATA / "norm_stats.npz").exists():
    _s = np.load(DATA / "norm_stats.npz")
    MU, SD = _s["mu"].reshape(-1), _s["sd"].reshape(-1)
else:  # sin estadísticas se devuelve la señal en desviaciones estándar
    MU, SD = np.zeros(12), np.ones(12)

for _c in ("mi_subclase", "infarction_stadium1", "ruido"):   # columnas nuevas del notebook v3
    if _c not in df.columns:
        print(f"[PTB-XL] AVISO: el CSV no tiene '{_c}' (¿es el de la v2?). Se rellena vacío.")
        df[_c] = np.nan

LABEL = df["label"].map(CL2I).to_numpy(dtype=int)
PRED = df["prediccion"].map(CL2I).to_numpy(dtype=int)
SPLIT = df["split"].map({s: i for i, s in enumerate(SPLITS)}).to_numpy(dtype=int)
AGE = df["age"].fillna(0).clip(upper=90).to_numpy()   # PTB-XL codifica >89 años como 300
SEX = df["sex"].fillna(0).astype(int).to_numpy()      # 0 = hombre, 1 = mujer
PROBS = df[[f"p_{c}" for c in CLASES]].to_numpy()
POS = pd.Series(np.arange(len(df)), index=df["ecg_id"])  # ecg_id -> fila

SUBS = sorted(df["mi_subclase"].dropna().astype(str).unique().tolist())
SUB = df["mi_subclase"].astype("object").map(lambda v: SUBS.index(str(v)) if pd.notna(v) else -1).to_numpy(dtype=int)

_orden = ["Stadium I", "Stadium II", "Stadium II-III", "Stadium III", "unknown"]
_et = df["infarction_stadium1"].fillna("sin dato").astype(str)
ETAPAS = [e for e in _orden if e in set(_et)] + sorted(set(_et) - set(_orden))
ETAPA = _et.map({e: i for i, e in enumerate(ETAPAS)}).to_numpy(dtype=int)
RUIDO = df["ruido"].fillna(False).astype(str).str.lower().isin(["true", "1"]).astype(int).to_numpy()

# ---------------------------------------------------------------- interpretabilidad (opcional)
# distancia a los vecinos, voto del kNN con tres distancias y grupos de K-means / DBSCAN
HAY_INTERP = (DATA / "interpretabilidad.csv").exists()
if HAY_INTERP:
    _it = pd.read_csv(DATA / "interpretabilidad.csv").set_index("ecg_id").reindex(df["ecg_id"])
    DVEC = _it["dvec"].fillna(-1).to_numpy()
    KM = _it["km"].fillna(-1).astype(int).to_numpy()
    DB = _it["db"].fillna(-2).astype(int).to_numpy()
    KNN_COS = _it["knn_cos"].map(CL2I).fillna(-1).astype(int).to_numpy()
    _INFO = json.loads((DATA / "interpretabilidad.json").read_text(encoding="utf-8")) \
        if (DATA / "interpretabilidad.json").exists() else {}
    KM_DOM = [_INFO.get("km_dominante", {}).get(str(k), "?") for k in range(5)]
    KM_TAM = [_INFO.get("km_tamano_train", {}).get(str(k), 0) for k in range(5)]
else:
    print("[PTB-XL] AVISO: falta data/interpretabilidad.csv; la ficha no mostrará distancias, kNN ni grupos. "
          "Genéralo con: python scripts/generar_interpretabilidad.py")

_cache_puntos: dict[str, bytes] = {}


def fila(ecg_id: int) -> int:
    if ecg_id not in POS.index:
        raise HTTPException(404, f"ecg_id {ecg_id} no existe")
    return int(POS[ecg_id])


# ---------------------------------------------------------------- API
@app.get("/api/puntos")
def api_puntos(proy: str = Query("umap", pattern="^(umap|tsne)$")):
    """Todos los puntos en formato columnar (el filtrado se hace en el cliente)."""
    if proy not in _cache_puntos:
        cx, cy = f"{proy}_x", f"{proy}_y"
        if cx not in df.columns:
            raise HTTPException(404, f"El CSV no tiene la proyección {proy}")
        payload = {
            "n": len(df),
            "proy": proy,
            "clases": CLASES,
            "subs": SUBS,
            "etapas": ETAPAS,
            "id": df["ecg_id"].tolist(),
            "x": df[cx].round(3).tolist(),
            "y": df[cy].round(3).tolist(),
            "label": LABEL.tolist(),
            "pred": PRED.tolist(),
            "conf": df["confianza"].round(3).tolist(),
            "knn": df["knn_acuerdo"].round(2).tolist(),
            "age": AGE.tolist(),
            "sex": SEX.tolist(),
            "split": SPLIT.tolist(),
            "sub": SUB.tolist(),
            "etapa": ETAPA.tolist(),
            "ruido": RUIDO.tolist(),
        }
        if HAY_INTERP:
            payload.update({
                "dvec": np.round(DVEC, 5).tolist(),
                "km": KM.tolist(),
                "kcos": KNN_COS.tolist(),
                "km_dom": KM_DOM,
                "km_tam": KM_TAM,
            })
        _cache_puntos[proy] = json.dumps(payload, separators=(",", ":")).encode()
    return Response(_cache_puntos[proy], media_type="application/json")


@app.get("/api/coocurrencia")
def api_coocurrencia():
    """Matriz de coocurrencia calculada en el notebook con los datos completos."""
    ruta = DATA / "cooccurrence.json"
    if not ruta.exists():
        raise HTTPException(404, "Falta data/cooccurrence.json (lo genera el notebook v3.1)")
    return Response(ruta.read_bytes(), media_type="application/json")


@app.get("/api/paciente/{ecg_id}")
def api_paciente(ecg_id: int):
    i = fila(ecg_id)
    r = df.iloc[i]
    vec = [int(v) for v in str(r["vecinos"]).split(";") if v.strip().isdigit()]
    interp = None
    if HAY_INTERP:
        _f = _it.iloc[i]
        interp = {
            "dist_vecinos": [float(x) for x in str(_f["dist_vec"]).split(";") if x],
            "dvec": float(_f["dvec"]),
            "knn": {"coseno": _f["knn_cos"], "manhattan": _f["knn_l1"], "chebyshev": _f["knn_linf"]},
            "kmeans": {"grupo": int(_f["km"]), "dominante": KM_DOM[int(_f["km"])]},
            "dbscan": int(_f["db"]),
        }
    return {
        "ecg_id": int(r["ecg_id"]),
        "patient_id": int(r["patient_id"]),
        "edad": None if r["age"] > 89 else int(r["age"]),
        "sexo": "Mujer" if SEX[i] == 1 else "Hombre",
        "fold": int(r["strat_fold"]),
        "split": r["split"],
        "label": r["label"],
        "prediccion": r["prediccion"],
        "confianza": float(r["confianza"]),
        "probs": [round(float(p), 4) for p in PROBS[i]],
        "knn_acuerdo": float(r["knn_acuerdo"]),
        "frac_vecinos_MI": float(r["frac_vecinos_MI"]),
        "vecinos": vec,
        "subtipo": None if SUB[i] < 0 else SUBS[SUB[i]],
        "etapa": None if _et.iloc[i] == "sin dato" else _et.iloc[i],
        "interp": interp,
    }


@app.get("/api/senal/{ecg_id}")
def api_senal(ecg_id: int, derivacion: int = Query(0, ge=0, le=11)):
    """Señal de una derivación en mV (filtrada 0,5-40 Hz)."""
    i = fila(ecg_id)
    z = np.asarray(SENALES[i, derivacion], dtype=np.float32)
    mv = z * SD[derivacion] + MU[derivacion]
    return {"fs": FS, "derivacion": DERIVACIONES[derivacion], "mv": np.round(mv, 4).tolist()}


# ---------------------------------------------------------------- estáticos
@app.get("/")
def raiz():
    return FileResponse(STATIC / "index.html")


@app.get("/instrumento")
def instrumento():
    return FileResponse(STATIC / "instrumento.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

# ---------------------------------------------------------------- modo revisión (cardiólogo) y acceso con contraseña
# (el modo de revisión con cardiólogo no forma parte de esta entrega; si existe backend/revision.py se activa)
try:
    try:
        from backend.revision import instalar_revision
    except ImportError:  # si se ejecuta como `uvicorn main:app` desde dentro de backend/
        from revision import instalar_revision
    instalar_revision(app)
except ImportError:
    pass

try:
    try:
        from backend.comparar import instalar_comparar
    except ImportError:
        from comparar import instalar_comparar
    instalar_comparar(app, df, SENALES, MU, SD, fila)
except Exception as _e:   # la comparación es opcional
    print("[PTB-XL] AVISO: no se pudo activar la comparación de vecinos:", _e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)