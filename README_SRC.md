# GLOBEM — `src/` (Pasos A.1, A.2 y A.3 completados)

Motor Python extraído del notebook `globem_es.ipynb`. Todo el código sale del
notebook celda a celda; lo que se ha hecho es organizarlo en módulos por
responsabilidad, generalizar lo que el notebook repetía cuatro veces (una por
dimensión), y añadir tests.

## Los 8 módulos

| Módulo | Qué hace | Origen en el notebook |
|---|---|---|
| `data_loading.py` | Lee los 5 CSV, prepara la columna `day`, limpia el warmup (días 0-6) | Bloque 1 (celda 21) y Bloque 2 (secciones 2.5.x-II y IV) |
| `baseline.py` | Baseline rolling individual: shift 7, ventana 21, cobertura mínima 15 | Bloque 2, secciones 2.5.x-V (celdas 49, 80, 109, 138) |
| `zscore.py` | z-score intra-sujeto `(obs − media) / desviación` | Bloque 3, secciones 3.2 a 3.5 (celdas 155, 158, 161, 164) |
| `features.py` | Trunca las colas contaminadas, calcula rolling mean/std, winsoriza location al p95 | Bloque 4 (celdas 187, 192, 207) y sección 5.1 (celda 212) |
| `rule.py` | Las 4 capas: indicador diario → persistencia → dominios → convergencia 2/3 | Bloque 5, secciones 5.4 a 5.6 (celdas 225, 227, 230, 233) |
| `episodes.py` | Días-alerta → episodios (arranques de racha) | Sección 6.3.3 (celda 295) |
| `clinical_notice.py` | Traduce el z del día de arranque a lenguaje llano → el aviso al terapeuta | Sección 6.3.5 (celda 302) |
| `validation.py` | Ruido blanco, permutación temporal, CUSUM, EWMA, corroboración, frescura | Bloque 7 (celdas 312, 315, 319, 324) |

## Cómo se encadenan

```
data_loading  →  baseline  →  zscore  →  features  →  rule  →  episodes
                                                        ↓          ↓
                                                  validation  clinical_notice
```

## Uso

```python
from src.data_loading import load_and_prepare_all
from src.baseline import compute_all_baselines
from src.zscore import compute_all_zscores
from src.features import build_features
from src.rule import run_rule, count_alert_days, count_people_with_alerts
from src.episodes import build_episodes
from src.clinical_notice import build_clinical_notices

dfs, df_bdi, columnas = load_and_prepare_all("datasets")
baselines = compute_all_baselines(dfs, columnas)
zscores = compute_all_zscores(dfs, baselines, columnas)
df_rolling, df_std, umbral, n_recortados = build_features(zscores, dfs)

reglas = run_rule(df_rolling, threshold=2.0, persist=5)
df_episodios = build_episodes(reglas["convergence"])
avisos = build_clinical_notices(df_episodios, df_rolling, reglas["persistence"])
```

## Tests

```
python3 -m pytest tests/ -q
```

- **97 tests sintéticos** que no necesitan los CSV: comprueban cada pieza con
  datos construidos a propósito (6 personas, 2 de ellas con una desviación real
  sostenida en sueño y pasos).
- **12 tests contra los datos reales** (`test_real_data.py`), que verifican las
  cifras exactas del notebook: 186 días-alerta, 63 episodios, 56 personas,
  umbral de winsorización 24.07, 0 alertas sobre ruido blanco, 100% de
  corroboración CUSUM/EWMA, frescura 31/11/21. Se saltan solos si los CSV no
  están en `datasets/`.

**Total: 109 tests.**

## Diferencias respecto al notebook

Ninguna en resultados. Dos en la implementación, ambas deliberadas:

1. **El notebook repite el mismo código cuatro veces** (una por dimensión) para
   la columna `day`, el warmup, el baseline y el z-score. Aquí es una función
   que recibe la dimensión como parámetro.

2. **Corregido un fallo latente de alineación.** El notebook resta
   `df[col].values − baseline["mean"].values`, lo cual solo es correcto si el
   dataframe y el resultado del `groupby` tienen el mismo orden de filas.
   `groupby` ordena los participantes alfabéticamente; en los CSV del estudio
   los `pid` ya vienen ordenados y coinciden, pero con cualquier otro orden se
   restaría la observación de una persona contra el baseline de otra, en
   silencio. Aquí se alinea por índice de forma explícita y se preserva el
   orden con `sort=False`. Hay un test de regresión que lo blinda
   (`test_el_zscore_no_mezcla_participantes_si_el_orden_no_es_alfabetico`).

## Añadido después de la extracción (no venía del notebook)

- **A.2 — Semáforo señal/ruido**, en `validation.py`: `signal_noise_score` y
  `signal_noise_grid`. Convierte "alertas reales frente a alertas sobre ruido"
  en un número 0-10 para cada combinación de umbral y persistencia. Usa
  permutación temporal, no ruido blanco (el ruido blanco da cero alertas en
  todas las configuraciones y no distinguiría ninguna).

- **A.3 — Series por persona para la app**, en `features.py`:
  `build_trajectory`, `build_episode_window`, `count_active_domains` y
  `global_index_to_day`. Exportan la serie diaria de las cuatro señales, los
  días-alerta y los arranques de episodio. Sin ventana alimentan el panel
  izquierdo (92 días); con ventana, la gráfica pequeña del panel derecho, que
  es un recorte exacto de la misma serie.

## Pendiente

- **A.4** — Las tres vistas del dashboard: Historial, Pacientes, Resumen.
- **B** — `run_pipeline.py`: ejecuta el motor una vez y vuelca los JSON.
- **C/D** — Copiar los JSON a `Portafolio-VSC/projects/globem-app/data/`.
- **E** — `app.js`: reproductor, gráficas, mandos, pestañas.
- **F** — `Dockerfile` y despliegue.
