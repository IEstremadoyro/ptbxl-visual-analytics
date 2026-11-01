"""
Genera data/interpretabilidad.csv y data/interpretabilidad.json para la plataforma.

Qué calcula (todo sobre el resumen de 64 números del modelo CE+SupCon):
  - dist_vec : distancia coseno a cada uno de los 10 vecinos de la columna `vecinos` del CSV (mismo orden)
  - dvec     : promedio de esas 10 distancias
  - knn_cos / knn_l1 / knn_linf : clase que votan los 10 vecinos (de entrenamiento) con distancia
                coseno, Manhattan y Chebyshev. Para registros de entrenamiento se excluye el propio registro.
  - km       : grupo de K-means (k=5, ajustado con entrenamiento)
  - db       : grupo de DBSCAN (coseno, min_samples=10; -1 = ruido), ajustado con entrenamiento;
               los demás registros se asignan al punto núcleo más cercano si está a menos de eps.
Necesita torch y scikit-learn. Uso (desde la carpeta del proyecto):
    python scripts/generar_interpretabilidad.py
Opcional: --emb ruta.npz para reutilizar resúmenes ya calculados (E, P).
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

CL = ["NORM", "MI", "STTC", "CD", "HYP"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
ap.add_argument("--emb", default=None)
a = ap.parse_args()
D = Path(a.data)

df = pd.read_csv(D / "ptbxl_predictivo_umap.csv")
ids = df.ecg_id.to_numpy()


def embeddings():
    import torch, torch.nn as nn, torch.nn.functional as F

    def bloque(c_in, c_out, k):
        return nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2))

    class Atencion(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.score = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        def forward(self, h):
            h = h.permute(0, 2, 1)
            w = F.softmax(self.score(h), dim=1)
            return (w * h).sum(1), w.squeeze(-1)

    class Encoder(nn.Module):
        def __init__(self, n_leads=12, n_clases=5, d_emb=64):
            super().__init__()
            self.cnn = nn.Sequential(bloque(n_leads, 32, 7), bloque(32, 64, 5), bloque(64, 128, 5), bloque(128, 128, 3))
            self.atencion = Atencion(128); self.drop = nn.Dropout(0.3)
            self.proyeccion = nn.Linear(128, d_emb); self.clasificador = nn.Linear(128, n_clases)
        def forward(self, x):
            ctx, _ = self.atencion(self.cnn(x)); ctx = self.drop(ctx)
            return self.clasificador(ctx), F.normalize(self.proyeccion(ctx), dim=1), ctx

    m = Encoder()
    sd = torch.load(D / "modelo_CE_SupCon.pth", map_location="cpu")
    m.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd)
    m.eval()
    X = np.load(D / "senales_norm_f16.npy", mmap_mode="r")
    E, L = [], []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            lo, em, _ = m(torch.from_numpy(np.asarray(X[i:i + 256], dtype=np.float32)))
            E.append(em); L.append(lo)
    return torch.cat(E).numpy(), torch.softmax(torch.cat(L), 1).numpy()


if a.emb:
    z = np.load(a.emb); E, P = z["E"], z["P"]
else:
    E, P = embeddings()

# comprobación: las predicciones recalculadas deben coincidir con las del CSV
coincide = float((np.array(CL)[P.argmax(1)] == df.prediccion.to_numpy()).mean())
print(f"coincidencia de predicciones con el CSV: {coincide:.4f}")
assert coincide > 0.999, "El modelo y el CSV no corresponden"

y = df.label.map({c: i for i, c in enumerate(CL)}).to_numpy()
tr = (df.split == "train").to_numpy()
ytr = y[tr]; idtr = ids[tr]
pos = {int(e): i for i, e in enumerate(ids)}

# 1) distancia coseno a los vecinos de la columna `vecinos`
En = E / np.linalg.norm(E, axis=1, keepdims=True)
dist_vec, dvec = [], np.zeros(len(df))
for r, s in enumerate(df.vecinos.astype(str)):
    v = [int(t) for t in s.split(";") if t.strip().isdigit()]
    d = [1.0 - float(En[r] @ En[pos[t]]) for t in v]
    dist_vec.append(";".join(f"{x:.5f}" for x in d)); dvec[r] = np.mean(d) if d else np.nan

# 2) voto de los 10 vecinos con tres distancias
def voto(lbl):
    out = np.zeros(len(lbl), int)
    for i, l in enumerate(lbl):
        c = np.bincount(l, minlength=5); t = np.where(c == c.max())[0]
        out[i] = t[0] if len(t) == 1 else next(x for x in l if x in t)
    return out

votos = {}
for nombre, met in [("knn_cos", "cosine"), ("knn_l1", "manhattan"), ("knn_linf", "chebyshev")]:
    nn = NearestNeighbors(n_neighbors=11, metric=met, algorithm="brute").fit(E[tr])
    _, ix = nn.kneighbors(E)
    sel = np.zeros((len(df), 10), int)
    for r in range(len(df)):
        cand = [j for j in ix[r] if idtr[j] != ids[r]][:10]
        sel[r] = ytr[cand]
    votos[nombre] = np.array(CL)[voto(sel)]
    print(nombre, "listo")

# 3) agrupamientos (ajustados con entrenamiento)
km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(E[tr])
lab_km = km.predict(E)
dom = {int(c): CL[int(np.bincount(ytr[km.labels_ == c], minlength=5).argmax())] for c in range(5)}
tam = {int(c): int((km.labels_ == c).sum()) for c in range(5)}

nn_tr = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute").fit(E[tr])
dtr, _ = nn_tr.kneighbors(E[tr])
# eps: primer percentil (50, 70, 85) de la distancia al 10.º vecino que da al menos 2 grupos y menos de 35 % de ruido
elegido = None
for pc in (50, 70, 85):
    eps_c = float(np.percentile(dtr[:, -1], pc))
    db_c = DBSCAN(eps=eps_c, min_samples=10, metric="cosine", algorithm="brute", n_jobs=2).fit(E[tr])
    ncl = len(set(db_c.labels_)) - (1 if -1 in db_c.labels_ else 0)
    print(f"DBSCAN percentil {pc}: eps={eps_c:.4f} grupos={ncl} ruido={(db_c.labels_ == -1).mean():.2f}")
    if elegido is None and ncl >= 2 and (db_c.labels_ == -1).mean() < 0.35:
        elegido = (eps_c, db_c)
if elegido is None:
    elegido = (eps_c, db_c)
eps, db = elegido
core = db.core_sample_indices_
nn_core = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(E[tr][core])
dd, ii = nn_core.kneighbors(E)
lab_db = np.where(dd[:, 0] <= eps, db.labels_[core][ii[:, 0]], -1)
lab_db[tr] = db.labels_
print("DBSCAN eps", eps, "grupos", sorted(set(lab_db)))

out = pd.DataFrame({"ecg_id": ids, "dist_vec": dist_vec, "dvec": np.round(dvec, 5),
                    **votos, "km": lab_km, "db": lab_db})
out.to_csv(D / "interpretabilidad.csv", index=False)
json.dump({"km_dominante": dom, "km_tamano_train": tam, "dbscan_eps": eps, "k_vecinos": 10,
           "nota": "K-means k=5 y DBSCAN (coseno, min_samples=10) ajustados con entrenamiento; db=-1 es ruido"},
          open(D / "interpretabilidad.json", "w"), indent=1, ensure_ascii=False)
print("guardado", D / "interpretabilidad.csv", out.shape, dom)
