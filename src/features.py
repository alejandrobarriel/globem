"""
features.py

Convierte las cuatro Series de z-score en la materia prima de la regla:
  1. Une los z-scores en un unico DataFrame (df_z).
  2. Trunca cada serie por participante y dimension, cortando la cola donde
     los agregados de 7 dias estan incompletos.
  3. Calcula el rolling mean y el rolling std del z-score en ventana de 7 dias.
  4. Winsoriza location al percentil 95 para que su escala no domine.

Extraido del notebook globem_es.ipynb, Bloque 4 (secciones 4.1, 4.2 y 4.4,
celdas 187, 192, 207) y Bloque 5 (seccion 5.1, celda 212).

El resultado, df_z_roll_mean_w, es exactamente el objeto que recibe
rule.apply_rule y validation.
"""

import numpy as np
import pandas as pd


# Columna ':allday' (valor diario crudo) de cada dimension. Se usa solo para
# el truncamiento: sirve para saber que dias tienen los 7 valores diarios
# completos detras (notebook, celda 192).
ALLDAY_COLUMNS = {
    "df_sleep": "f_slp:fitbit_sleep_summary_rapids_sumdurationasleepmain:allday",
    "df_steps": "f_steps:fitbit_steps_intraday_rapids_sumsteps:allday",
    "df_location": "f_loc:phone_locations_barnett_rog:allday",
    "df_screen": "f_screen:phone_screen_rapids_sumdurationunlock:allday",
}

# Nombre de dimension que usa el sistema, por cada clave de dataframe.
DIMENSION_NAMES = {
    "df_sleep": "sleep",
    "df_steps": "steps",
    "df_location": "location",
    "df_screen": "screen",
}

DIMENSIONS = ["sleep", "steps", "location", "screen"]


def build_zscore_dataframe(zscores):
    """
    Une las cuatro Series de z-score (zscore.compute_all_zscores) en un unico
    DataFrame con MultiIndex (pid, idx) y una columna por dimension.

    Devuelve df_z, el mismo objeto que el notebook construye en la celda 187.
    """
    df_z = pd.concat(
        [
            zscores["df_sleep"],
            zscores["df_steps"],
            zscores["df_location"],
            zscores["df_screen"],
        ],
        axis=1,
        keys=DIMENSIONS,
    )
    return df_z


def find_last_valid_day(df, columna_allday):
    """
    Para cada participante de df, busca el ultimo dia que tiene los 7 valores
    diarios (':allday') completos en su ventana de 7 dias.

    Motivo (notebook, seccion 4.2): la feature primaria ':7dhist' es una suma
    de 7 dias. Cuando alguno de esos 7 valores diarios falta, el agregado se
    calcula con menos sumandos y queda artificialmente bajo; el z-score
    hereda ese sesgo y produce una caida hacia valores muy negativos al final
    del registro de muchos participantes. No es conducta: es un artefacto. Por
    eso se corta cada serie antes de esa zona.

    Devuelve una Series {pid: ultimo dia valido}.
    """
    df_sorted = df.sort_values(["pid", "day"]).copy()

    last_day_per_pid = {}
    for pid, group in df_sorted.groupby("pid"):
        has_data = group[columna_allday].notna()
        sum_7days = has_data.rolling(window=7, min_periods=7).sum()
        full_window = sum_7days == 7
        valid_days = group["day"][full_window]
        if len(valid_days) > 0:
            last_day_per_pid[pid] = valid_days.max()

    return pd.Series(last_day_per_pid)


def truncate_zscores(df_z, dfs):
    """
    Aplica el truncamiento por participante y dimension: pone a NaN todos los
    z-scores posteriores al ultimo dia con ventana ':allday' integra.

    El corte es propio de cada participante (cada uno pierde cobertura en un
    momento distinto) y propio de cada dimension (sueno, pasos, movilidad y
    pantalla pueden degradarse en dias distintos para la misma persona).

    Devuelve una copia truncada de df_z.
    """
    df_z_trunc = df_z.copy()

    last_valid_day = {}
    for nombre_df in dfs:
        dim = DIMENSION_NAMES[nombre_df]
        columna_allday = ALLDAY_COLUMNS[nombre_df]
        last_valid_day[dim] = find_last_valid_day(dfs[nombre_df], columna_allday)

    day_relative = df_z_trunc.groupby(level="pid").cumcount()
    pids_z = df_z_trunc.index.get_level_values("pid")
    pid_per_row = pd.Series(pids_z, index=df_z_trunc.index)

    for dim in DIMENSIONS:
        last_day = pid_per_row.map(last_valid_day[dim]).fillna(-1)
        mask_invalid = day_relative.values > last_day.values
        df_z_trunc.loc[mask_invalid, dim] = np.nan

    return df_z_trunc


def compute_rolling(df_z, window=7):
    """
    Calcula el rolling mean y el rolling std del z-score en ventana movil de
    'window' dias, participante a participante.

    El rolling mean recoge cuanto se aleja la persona de su patron de forma
    sostenida (suaviza la fluctuacion diaria); el rolling std cuantifica si
    ese nivel es estable o volatil.

    Nota (notebook, seccion 4.4): la ventana de 7 dias viene de la estructura
    de la señal (la feature primaria ya es una suma deslizante de 7 dias), y
    NO debe confundirse con la persistencia de la regla, que son 5 dias y es
    una decision distinta del Bloque 5.

    Devuelve (df_z_roll_mean, df_z_roll_std).
    """
    roll_mean_parts = []
    roll_std_parts = []

    for pid, group in df_z.groupby(level="pid"):
        roll_mean_parts.append(group.rolling(window=window, min_periods=window).mean())
        roll_std_parts.append(group.rolling(window=window, min_periods=window).std())

    df_z_roll_mean = pd.concat(roll_mean_parts).reindex(df_z.index)
    df_z_roll_std = pd.concat(roll_std_parts).reindex(df_z.index)

    return df_z_roll_mean, df_z_roll_std


def compute_winsor_threshold(df_z_roll_mean, dimension="location", percentile=0.95):
    """
    Calcula el umbral de recorte: el percentil 95 de los valores absolutos de
    la dimension indicada.

    Sobre los datos reales del notebook el umbral sale 24.07 (celda 213).
    """
    abs_values = []
    for v in df_z_roll_mean[dimension].values:
        if not np.isnan(v):
            abs_values.append(abs(v))

    abs_values.sort()
    n_vals = len(abs_values)
    position = int(n_vals * percentile)
    threshold = abs_values[position]

    return threshold


def winsorize_location(df_z_roll_mean, threshold=None):
    """
    Recorta la cola extrema de location: los valores por encima del umbral se
    fijan en el umbral, y los que estan por debajo de -umbral se fijan en
    -umbral. Los NaN se mantienen.

    Motivo (notebook, seccion 5.1): el z-score de location tiene una cola
    muchisimo mas larga que sleep, steps o screen. Sin recortarla, location
    dispararia señal mucho mas a menudo que las demas solo por su escala, y
    arrastraria la decision del sistema ella sola. Tras el recorte, las cuatro
    dimensiones quedan en el mismo rango y puede aplicarse un umbral comun.

    Si no se pasa threshold, se calcula del propio dataframe con
    compute_winsor_threshold.

    Devuelve (df_z_roll_mean_w, threshold, n_clipped).
    """
    if threshold is None:
        threshold = compute_winsor_threshold(df_z_roll_mean)

    df_z_roll_mean_w = df_z_roll_mean.copy()

    location = df_z_roll_mean_w["location"].values.copy()
    n_clipped = 0

    for i in range(len(location)):
        if np.isnan(location[i]):
            pass
        elif location[i] > threshold:
            location[i] = threshold
            n_clipped = n_clipped + 1
        elif location[i] < -threshold:
            location[i] = -threshold
            n_clipped = n_clipped + 1

    df_z_roll_mean_w["location"] = location

    return df_z_roll_mean_w, threshold, n_clipped


def build_features(zscores, dfs, window=7):
    """
    Pipeline completo del Bloque 4 + seccion 5.1, de las cuatro Series de
    z-score al objeto que recibe la regla:

      z-scores -> df_z -> truncamiento -> rolling mean/std -> winsorizacion

    Devuelve (df_z_roll_mean_w, df_z_roll_std, threshold, n_clipped).
    """
    df_z = build_zscore_dataframe(zscores)
    df_z = truncate_zscores(df_z, dfs)
    df_z_roll_mean, df_z_roll_std = compute_rolling(df_z, window)
    df_z_roll_mean_w, threshold, n_clipped = winsorize_location(df_z_roll_mean)

    return df_z_roll_mean_w, df_z_roll_std, threshold, n_clipped
