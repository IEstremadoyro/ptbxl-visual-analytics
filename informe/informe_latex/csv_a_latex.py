#!/usr/bin/env python3
"""Convierte comparacion_proyecciones.csv (lo genera el notebook v3.1) en tabla_proyecciones.tex.

Uso:  python3 csv_a_latex.py [comparacion_proyecciones.csv] [tabla_proyecciones.tex]
Deja tabla_proyecciones.tex junto a main.tex; el informe la incluye solo si existe.
"""
import csv
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "comparacion_proyecciones.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "tabla_proyecciones.tex"

COLS = [("knn_acc_2D", "kNN"), ("silhouette_2D", "Silh."), ("recall_MI_2D", "Recall MI"),
        ("auc_MI_2D", "AUC MI"), ("trustworthiness", "Trust.")]

def esc(t):
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("$", "\\$")]:
        t = t.replace(a, b)
    return t

def num(v):
    return f"{float(v):.3f}".replace(".", ",")

with open(src, encoding="utf-8") as f:
    filas = list(csv.DictReader(f))
clave = list(filas[0].keys())[0]                      # primera columna = nombre de la proyección
mejor = {c: max(float(r[c]) for r in filas) for c, _ in COLS}

out = [r"\begin{table}[t]", r"\centering",
       r"\caption{Métricas sobre el mapa 2D (fold de test). En negrita, el mejor valor de cada columna.}",
       r"\label{tab:proyecciones}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
       r"\begin{tabular}{l" + "c" * len(COLS) + "}", r"\toprule",
       "Proyección & " + " & ".join(n for _, n in COLS) + r" \\", r"\midrule"]
for r in filas:
    celdas = []
    for c, _ in COLS:
        t = num(r[c])
        celdas.append(r"\textbf{" + t + "}" if float(r[c]) == mejor[c] else t)
    out.append(esc(r[clave]) + " & " + " & ".join(celdas) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
with open(dst, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Escrito", dst)
