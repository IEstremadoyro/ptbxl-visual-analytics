"""
Comparación de un ECG con sus 10 vecinos: latido medio de cada registro (12 derivaciones),
alineado en el pico R. Se usa para ver en paralelo al caso y a sus vecinos y para resaltar
dónde difieren. Es una ayuda visual; no demuestra la causa de un error.
"""
import numpy as np
from fastapi import APIRouter, HTTPException
from scipy.signal import find_peaks

PRE, POST = 25, 55            # muestras antes y después del pico R (100 Hz): 0,25 s y 0,55 s
router = APIRouter(prefix="/api")


def latido_medio(x):
    """x: (12, 1000) en mV. Mediana de los latidos alineados en el pico R, o None."""
    e = np.abs(np.diff(x, axis=1)).sum(0)
    e = np.convolve(e, np.ones(5) / 5, "same")
    pk, _ = find_peaks(e, distance=40, height=np.percentile(e, 90) * 0.5)
    pk = pk[(pk > PRE) & (pk < len(e) - POST)]
    if len(pk) < 3:
        return None, int(len(pk))
    return np.median(np.stack([x[:, p - PRE:p + POST] for p in pk]), 0), int(len(pk))


def instalar_comparar(app, df, SENALES, MU, SD, fila):
    @router.get("/comparar/{ecg_id}")
    def comparar(ecg_id: int):
        i = fila(ecg_id)
        r = df.iloc[i]
        vec = [int(v) for v in str(r["vecinos"]).split(";") if v.strip().isdigit()]
        ids = [int(ecg_id)] + vec
        out = []
        for k, e in enumerate(ids):
            j = fila(e)
            x = np.asarray(SENALES[j], dtype=np.float32) * SD[:, None] + MU[:, None]
            lat, n = latido_medio(x)
            out.append({
                "ecg_id": int(e), "k": k, "label": str(df.iloc[j]["label"]),
                "prediccion": str(df.iloc[j]["prediccion"]), "split": str(df.iloc[j]["split"]),
                "latidos": n,
                "beat": None if lat is None else np.round(lat, 3).tolist(),
            })
        return {"fs": 100, "pre": PRE, "post": POST, "registros": out}

    app.include_router(router)
