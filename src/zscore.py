"""
zscore.py

Calculo del z-score intra-sujeto: cuanto se aleja la observacion de un dia
concreto del baseline individual de esa persona en ese dia.

Extraido del notebook globem_es.ipynb, Bloque 3, secciones 3.2 a 3.5
(estructuralmente identico para sleep, steps, location y screen; el notebook
lo repite 4 veces, aqui se generaliza en una sola funcion; ver celdas 155,
158, 161, 164).
"""

import numpy as np
import pandas as pd


def compute_zscore(df, baseline, columna_valor):
    """
    Calcula el z-score intra-sujeto: (observacion - media del baseline) /
    desviacion tipica del baseline, para cada fila de df.

    df: DataFrame de una dimension (df_sleep, df_steps, df_location o
    df_screen), ya con la columna 'day' preparada.
    baseline: resultado de baseline.compute_baseline sobre esa misma
    dimension (tiene columnas 'mean' y 'std').
    columna_valor: nombre de la feature primaria de esa dimension (la misma
    que se uso para calcular el baseline).

    Devuelve una Series indexada igual que baseline (pid, fila original de
    df), con NaN en los dias donde el baseline no era calculable.

    El notebook (celda 155) resta directamente .values contra .values, lo que
    solo es correcto si baseline y df tienen exactamente el mismo orden de
    filas. Aqui se alinea de forma explicita: el segundo nivel del indice de
    baseline es la fila original de df, asi que se recupera cada observacion
    por su etiqueta y no por su posicion. El resultado es identico cuando el
    orden coincide, y correcto tambien cuando no.
    """
    filas_originales = baseline.index.get_level_values(1)
    observaciones = df[columna_valor].loc[filas_originales].values

    medias = baseline["mean"].values
    desviaciones = baseline["std"].values

    z = np.full(len(observaciones), np.nan)
    for i in range(len(observaciones)):
        if np.isnan(observaciones[i]) or np.isnan(medias[i]) or np.isnan(desviaciones[i]):
            continue
        if desviaciones[i] == 0:
            # Persona sin ninguna variabilidad en su ventana: el z-score no
            # esta definido (no hay escala contra la que medir la desviacion).
            continue
        z[i] = (observaciones[i] - medias[i]) / desviaciones[i]

    return pd.Series(z, index=baseline.index)


def compute_all_zscores(dfs, baselines, primary_columns):
    """
    Calcula el z-score de las 4 dimensiones a la vez.

    dfs: diccionario de data_loading.load_and_prepare_all.
    baselines: diccionario de baseline.compute_all_baselines.
    primary_columns: diccionario PRIMARY_COLUMNS de data_loading.py.

    Devuelve un diccionario con las mismas claves ("df_sleep", "df_steps",
    "df_location", "df_screen"), cada una con su Series de z-score (mismo
    resultado que z_sleep / z_steps / z_location / z_screen en el notebook).
    """
    zscores = {}
    for nombre_df in dfs:
        df = dfs[nombre_df]
        baseline = baselines[nombre_df]
        columna_valor = primary_columns[nombre_df]
        zscores[nombre_df] = compute_zscore(df, baseline, columna_valor)
    return zscores
