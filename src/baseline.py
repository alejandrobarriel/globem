"""
baseline.py

Calculo del baseline individual rolling (media y desviacion tipica de cada
participante, con buffer de 7 dias) para una dimension conductual.

Extraido del notebook globem_es.ipynb, Bloque 2, secciones 2.5.1-V a 2.5.4-IV
(identico para sleep, steps, location y screen, solo cambia la columna de
entrada; aqui se generaliza en una sola funcion, ver celdas 49, 80, 109, 138).
"""


def calcular_ventana_rolling(serie):
    """
    Aplica a una serie (los valores de UN participante, ya en orden temporal)
    el shift de 7 dias y la ventana rolling de 21 dias con cobertura minima
    de 15 valores no nulos, y devuelve media y desviacion tipica de cada
    ventana.

    Motivo del shift(7) (notebook, celda 47-48): la feature primaria es una
    suma deslizante de 7 dias, asi que el valor del dia d y el del dia d-1
    comparten 6 dias de datos crudos. Sin el shift, los ultimos dias de la
    ventana de 21 dias solaparian con la propia observacion del dia d y
    atenuarian artificialmente el z-score. Con el shift de 7 dias, el ultimo
    dia que entra en el baseline (d-8) ya no comparte datos crudos con la
    observacion (dias d-6 a d).

    Motivo de rolling(21, min_periods=15) (notebook, celda 48): ventana de 21
    dias previos, con al menos 15 valores no nulos (70% de cobertura) para
    que el baseline de ese dia se calcule; si no llega a 15, el resultado es
    NaN y ese dia queda como no evaluable en esa dimension.
    """
    serie_desplazada = serie.shift(7)
    ventana = serie_desplazada.rolling(window=21, min_periods=15)
    resultado = ventana.agg(["mean", "std"])
    return resultado


def compute_baseline(df, columna_valor, columna_pid="pid"):
    """
    Calcula el baseline rolling de columna_valor para cada participante de df
    por separado (groupby pid), aplicando calcular_ventana_rolling a la serie
    de cada uno.

    Devuelve un DataFrame con una fila por participante-dia y dos columnas,
    'mean' y 'std' (mismo resultado que baseline_sleep / baseline_steps /
    baseline_location / baseline_screen en el notebook).

    IMPORTANTE: se usa sort=False para que los participantes salgan en el
    mismo orden en que aparecen en df. Por defecto groupby los ordena
    alfabeticamente, y si ese orden no coincide con el del dataframe, el
    z-score restaria la observacion de una persona contra el baseline de
    otra. En los CSV del estudio los pid ya vienen ordenados y no se nota,
    pero el fallo seria silencioso en cualquier otro caso.
    """
    baseline = df.groupby(columna_pid, sort=False)[columna_valor].apply(calcular_ventana_rolling)
    return baseline


def compute_all_baselines(dfs, primary_columns, columna_pid="pid"):
    """
    Calcula el baseline rolling de las 4 dimensiones a la vez.

    dfs: diccionario devuelto por data_loading.load_and_prepare_all (claves
    "df_sleep", "df_steps", "df_location", "df_screen").
    primary_columns: diccionario PRIMARY_COLUMNS de data_loading.py.

    Devuelve un diccionario con las mismas claves, cada una con su baseline
    rolling (mean, std).
    """
    baselines = {}
    for nombre_df in dfs:
        df = dfs[nombre_df]
        columna_valor = primary_columns[nombre_df]
        baselines[nombre_df] = compute_baseline(df, columna_valor, columna_pid)
    return baselines
