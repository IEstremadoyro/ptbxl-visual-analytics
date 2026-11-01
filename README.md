# PTB-XL Visual Analytics

Plataforma web para explorar, con ayuda visual, **dónde están los infartos de miocardio y en qué casos falla el modelo** en la base de datos de ECG PTB-XL. Es el trabajo final del curso *Ciencia de Datos y Big Data* (Maestría en Ciencias de la Computación, UNSA; docente: Dra. Ana María Cuadros Valdivia).

**Autores:** Ignacio Martín Estremadoyro Lam · Andree Eduardo Morales Pulcha

- Informe (LaTeX): [`informe/informe_latex/main.tex`](informe/informe_latex/main.tex) (también en [`informe/informe_latex.zip`](informe/informe_latex.zip), listo para subir a Overleaf).
- Video de la plataforma (≤ 10 min): **ENLACE_AL_VIDEO** ← pegar aquí el enlace (Drive o YouTube no listado).

## Qué hace

Una red neuronal 1D-CNN con atención clasifica cada ECG en 5 grupos (normal, infarto, cambios ST-T, trastornos de conducción, hipertrofia) y lo resume en 64 números. Con esos números se calculan los vecinos más parecidos de cada paciente (kNN, K = 10, distancia coseno) y se dibuja un mapa de 2 dimensiones (UMAP o t-SNE). La plataforma conecta siete vistas y tres ventanas de apoyo:

| Vista | Para qué sirve |
|---|---|
| A. Cohortes | Elegir el conjunto (Train/Val/Test), sexo, edad y ver la composición por diagnóstico |
| B. Coocurrencia | Cuántos registros tienen más de un diagnóstico |
| C. Mapa | Cada punto es un ECG; se colorea por diagnóstico real, predicción, acuerdo con vecinos o grupo de K-means; selección con lazo |
| D. Matriz de confusión | Dónde se confunde la red (clic en una celda resalta esos ECG en el mapa) |
| E. Errores del infarto por subgrupo | Subtipo (anterior/inferior), sexo, edad |
| F y G. Paciente y señal | Ficha con probabilidades y vecinos; señal de 12 derivaciones con superposición de vecinos |
| Comparar con los 10 vecinos | Latido del paciente frente a la plantilla (mediana de sus vecinos) |
| Patrón de los perdidos | Cómo se ven, en la señal, los infartos que la red no detecta |
| Pruebas de apoyo | Distancias alternativas, vecinos lejanos, agrupamientos y cambios en la señal |

Todo lo que se muestra corresponde al **conjunto de prueba (1.650 ECG)** salvo que se cambie el filtro. La clasificación de la matriz de confusión es la de la red; el kNN se usa solo para interpretar. No se usan SHAP ni LIME: la interpretación es visual.

Resultado principal (un solo entrenamiento y una sola partición de los datos): la red detecta 173 de los 256 infartos de prueba (recall 67,6 %) y pierde 83. Esos infartos perdidos tienen vecinos de otra clase y se parecen más al latido normal. Son descripciones del error, no explicaciones causales ni validación clínica.

## Ejecutar en tu computadora

Requisitos: Python 3.10 o superior (no hace falta GPU ni torch).

```bash
git clone https://github.com/IEstremadoyro/ptbxl-visual-analytics.git
cd ptbxl-visual-analytics

python3 -m venv venv
source venv/bin/activate            # en Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/descargar_datos.py   # baja data/senales_norm_f16.npy (390 MB), ver «Datos grandes»

python -m uvicorn backend.main:app --port 8000
```

Abrir <http://localhost:8000>. (En macOS/Linux también sirve `bash scripts/iniciar_demo.sh`, que busca un puerto libre.)

## Ejecutar en Google Colab

Abre [`colab/PTBXL_colab.ipynb`](colab/PTBXL_colab.ipynb) en Colab (Archivo → Abrir cuaderno → GitHub) y ejecuta las celdas en orden. Clona el repositorio, instala lo necesario, descarga los datos grandes, levanta la plataforma y la muestra dentro del cuaderno. Si la vista incrustada no carga, la última celda genera un enlace para abrirla en otra pestaña. Las celdas finales, opcionales, regeneran los resultados de interpretabilidad y del patrón de los perdidos.

## Datos grandes

GitHub no admite archivos de más de 100 MB, así que `data/senales_norm_f16.npy` (390 MB, señales ya normalizadas de los 16.244 ECG de una sola superclase) no está en el repositorio. Se baja de la sección *Releases* de este repositorio (archivo `senales_norm_f16.npy` de la versión `v1.0`) con `scripts/descargar_datos.py`, o a mano: descárgalo y colócalo en `data/`.

El resto de los datos pesa poco y sí está en `data/`:

| Archivo | Contenido |
|---|---|
| `ptbxl_predictivo_umap.csv` | Un registro por ECG: diagnóstico, partición, predicción de la red, probabilidades, vecinos, coordenadas UMAP y t-SNE |
| `interpretabilidad.csv/.json` | Distancias a los vecinos, voto del kNN con 3 distancias, grupos de K-means y DBSCAN |
| `cooccurrence.json` | Matriz de coocurrencia (vista B) |
| `norm_stats.npz` | Media y desviación de cada derivación (para volver las señales a su escala) |
| `modelo_CE_SupCon.pth`, `modelo_solo_CE.pth` | Pesos de la red entrenada |
| `metricas.json`, `comparacion_proyecciones.json`, `analisis_errores_MI.csv` | Métricas y comparaciones usadas en el informe |

Las señales originales vienen de PTB-XL (<https://physionet.org/content/ptb-xl/1.0.3/>, tasa de 100 Hz, `records100/`); no se incluyen aquí. La plataforma solo necesita las señales normalizadas.

## Regenerar resultados (opcional)

```bash
pip install -r requirements-scripts.txt
python scripts/generar_interpretabilidad.py   # vecinos, distancias, K-means y DBSCAN
python scripts/analisis_patron.py             # patrón de los infartos perdidos
```

El entrenamiento de la red se hizo en un cuaderno de Google Colab aparte; aquí están sus resultados, no el entrenamiento.

## Estructura

```text
backend/    API FastAPI (main.py, comparar.py)
frontend/   Página y vistas en D3.js (index.html, app.js, comparar.js, pruebas.js, estilos)
data/       Resultados del modelo y archivos livianos (ver arriba)
scripts/    Descarga de datos y regeneración de resultados
colab/      Cuaderno para ejecutar todo en Colab
informe/    Informe en LaTeX
```

## Base de datos y licencia

PTB-XL: Wagner et al., *Scientific Data* 7, 154 (2020), licencia CC BY 4.0. El código de este repositorio es material académico.
