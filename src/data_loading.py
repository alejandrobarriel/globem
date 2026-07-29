"""
data_loading.py

Carga de los cuatro sensores del proyecto (sleep, steps, location, screen) y de las
puntuaciones BDI-II finales, y preparacion de la columna 'day' (dia relativo desde el
inicio del registro de cada participante).

Extraido y generalizado del notebook globem_es.ipynb, Bloque 1 (celda de carga) y
Bloque 2, secciones 2.5.1-II a 2.5.4-IV (columna 'day', repetida de forma casi
identica para las 4 dimensiones; aqui se generaliza en una sola funcion).
"""

import pandas as pd
import numpy as np


# Feature primaria (raw, ventana 7dhist) de cada dimension.
# Mismos nombres que en el notebook (celdas 26, 66, 97, 126).
PRIMARY_COLUMNS = {
    "df_sleep": "f_slp:fitbit_sleep_summary_rapids_sumdurationasleepmain:7dhist",
    "df_steps": "f_steps:fitbit_steps_intraday_rapids_sumsteps:7dhist",
    "df_location": "f_loc:phone_locations_barnett_rog:7dhist",
    "df_screen": "f_screen:phone_screen_rapids_sumdurationunlock:7dhist",
}


def load_datasets(data_dir):
    """
    Lee los 4 CSV de sensores y el CSV de BDI-II final.

    data_dir: ruta a la carpeta datasets/ (contiene sleep.csv, steps.csv,
    location.csv, screen.csv, dep_endterm.csv).

    Devuelve (dfs, df_bdi), donde dfs es un diccionario con las mismas claves
    que PRIMARY_COLUMNS: "df_sleep", "df_steps", "df_location", "df_screen".
    """
    dfs = {
        "df_sleep": pd.read_csv(data_dir + "/sleep.csv"),
        "df_steps": pd.read_csv(data_dir + "/steps.csv"),
        "df_screen": pd.read_csv(data_dir + "/screen.csv"),
        "df_location": pd.read_csv(data_dir + "/location.csv"),
    }
    df_bdi = pd.read_csv(data_dir + "/dep_endterm.csv")
    return dfs, df_bdi


def prepare_day_column(df):
    """
    Añade la columna 'day': dia relativo de cada fila desde el inicio del
    registro de ese participante (0 = primer dia de ese pid).

    Mismo procedimiento para las 4 dimensiones (notebook, celdas 30, 34, 36,
    40, 42, repetidas identicas por sensor en las secciones 2.5.X-II):
      1. fecha minima por participante (con 'date' todavia en texto)
      2. conversion de 'date' y de esa fecha minima a tipo fecha real
      3. cada fila recibe la fecha de inicio de su participante
      4. dia relativo = fecha de la fila menos fecha de inicio, en dias enteros

    Modifica y devuelve el mismo df.
    """
    start_date = df.groupby("pid")["date"].min()

    df["date"] = pd.to_datetime(df["date"])
    start_date = pd.to_datetime(start_date)

    df["start_date"] = df["pid"].map(start_date)
    df["day"] = df["date"] - df["start_date"]
    df["day"] = df["day"].dt.days

    return df


def clean_warmup_days(df, column):
    """
    Pone a NaN los dias 0-6 de cada participante en la columna indicada.

    Motivo (notebook, celda 46 y seccion 2.5.1-IV): la columna ':7dhist' de
    cada feature no tiene NaN en los dias 0-6 como cabria esperar del warmup,
    sino sumas parciales acumuladas (dia 0 = suma de 1 dia real, dia 1 = suma
    de 2 dias, etc.), y si entran al rolling sesgan el baseline hacia abajo en
    el tramo inicial. En sleep, steps y location este filtro cambia valores
    reales; en screen los dias 0-6 ya eran NaN, así que no cambia nada.

    Modifica y devuelve el mismo df.
    """
    mask_warmup = df.groupby("pid").cumcount() <= 6
    df.loc[mask_warmup, column] = np.nan
    return df


def load_and_prepare_all(data_dir):
    """
    Carga los 4 sensores y el BDI, y aplica a cada sensor prepare_day_column
    y clean_warmup_days sobre su feature primaria (PRIMARY_COLUMNS).

    Devuelve (dfs, df_bdi, PRIMARY_COLUMNS) listos para pasar a baseline.py.
    """
    dfs, df_bdi = load_datasets(data_dir)

    for nombre_df in dfs:
        df = dfs[nombre_df]
        df = prepare_day_column(df)
        columna_primaria = PRIMARY_COLUMNS[nombre_df]
        df = clean_warmup_days(df, columna_primaria)
        dfs[nombre_df] = df

    return dfs, df_bdi, PRIMARY_COLUMNS
