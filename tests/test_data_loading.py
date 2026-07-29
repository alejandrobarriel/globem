"""
test_data_loading.py

Comprueba la preparacion de la columna 'day' y la limpieza del warmup.
"""

import numpy as np
import pandas as pd

from src.data_loading import prepare_day_column, clean_warmup_days, PRIMARY_COLUMNS

from conftest import COLUMNA, PERSONAS


def test_columna_day_empieza_en_cero_por_persona(dfs_sinteticos):
    """Cada participante empieza en su dia 0, no en un dia global."""
    df = dfs_sinteticos["df_sleep"]
    for pid in PERSONAS:
        dias = df[df["pid"] == pid]["day"]
        assert dias.min() == 0
        assert dias.max() == 91


def test_columna_day_es_entera(dfs_sinteticos):
    """El dia relativo tiene que ser un entero, no un timedelta."""
    df = dfs_sinteticos["df_sleep"]
    assert df["day"].dtype == np.int64 or df["day"].dtype == np.int32


def test_warmup_pone_a_nan_los_dias_0_a_6(dfs_sinteticos):
    """
    Los dias 0-6 traen sumas parciales acumuladas, no sumas de 7 dias reales.
    Si entrasen al rolling sesgarian el baseline hacia abajo.
    """
    df = dfs_sinteticos["df_sleep"]
    for pid in PERSONAS:
        sub = df[df["pid"] == pid]
        warmup = sub[sub["day"] <= 6][COLUMNA]
        assert warmup.isna().all()


def test_warmup_no_toca_el_dia_7(dfs_sinteticos):
    """El dia 7 es el primero con suma de 7 dias completa: no se limpia."""
    df = dfs_sinteticos["df_sleep"]
    for pid in PERSONAS:
        sub = df[df["pid"] == pid]
        dia_7 = sub[sub["day"] == 7][COLUMNA]
        assert dia_7.notna().all()


def test_fechas_distintas_por_participante():
    """
    La fecha de inicio se calcula por participante, no de forma global: dos
    personas que empiezan en dias distintos deben tener ambas su dia 0.
    """
    filas = []
    for pid, inicio in [("A", "2018-04-03"), ("B", "2018-05-10")]:
        for dia in range(10):
            fecha = pd.Timestamp(inicio) + pd.Timedelta(days=dia)
            filas.append({"pid": pid, "date": fecha.strftime("%Y-%m-%d"), "v": 1.0})

    df = prepare_day_column(pd.DataFrame(filas))

    for pid in ["A", "B"]:
        dias = df[df["pid"] == pid]["day"]
        assert dias.min() == 0
        assert dias.max() == 9


def test_hay_una_columna_primaria_por_dimension():
    """Las cuatro dimensiones tienen que tener su feature primaria definida."""
    assert set(PRIMARY_COLUMNS.keys()) == {
        "df_sleep", "df_steps", "df_location", "df_screen"
    }
    for columna in PRIMARY_COLUMNS.values():
        assert columna.endswith(":7dhist")
