"""
test_baseline.py

Comprueba el baseline rolling con buffer: la ventana correcta, el criterio de
cobertura del 70% y el primer dia evaluable.
"""

import numpy as np
import pandas as pd

from src.baseline import compute_baseline, calcular_ventana_rolling

from conftest import COLUMNA, PERSONAS


def test_primer_dia_evaluable_es_el_28(dfs_sinteticos):
    """
    Tras limpiar los dias 0-6, los primeros 15 valores limpios son los dias
    7-21, y la ventana [d-28, d-8] los contiene justo cuando d=28, que es
    cuando se cumple min_periods=15.
    """
    df = dfs_sinteticos["df_sleep"]
    baseline = compute_baseline(df, COLUMNA)

    for pid in PERSONAS:
        serie = baseline.loc[pid]["mean"].reset_index(drop=True)
        assert serie.first_valid_index() == 28


def test_ventana_de_deteccion_es_de_64_dias(dfs_sinteticos):
    """De los 92 dias, son evaluables los dias 28 a 91: 64 dias."""
    df = dfs_sinteticos["df_sleep"]
    baseline = compute_baseline(df, COLUMNA)

    for pid in PERSONAS:
        serie = baseline.loc[pid]["mean"]
        assert serie.notna().sum() == 64


def test_baseline_devuelve_media_y_desviacion(dfs_sinteticos):
    df = dfs_sinteticos["df_sleep"]
    baseline = compute_baseline(df, COLUMNA)
    assert list(baseline.columns) == ["mean", "std"]


def test_buffer_de_7_dias_evita_el_solape():
    """
    El baseline del dia d no debe incluir el valor del propio dia d ni los 7
    anteriores: la feature es una suma deslizante de 7 dias y solaparia.

    Con una serie donde los primeros 30 dias valen 100 y a partir de ahi
    valen 999, la media del baseline en el dia 30 debe seguir siendo 100:
    el salto todavia no ha entrado en la ventana [d-28, d-8].
    """
    valores = []
    for i in range(40):
        if i < 30:
            valores.append(100.0)
        else:
            valores.append(999.0)

    df = pd.DataFrame({
        "pid": ["A"] * 40,
        COLUMNA: valores,
    })

    baseline = compute_baseline(df, COLUMNA)
    media_dia_30 = baseline.loc["A"]["mean"].reset_index(drop=True)[30]

    assert media_dia_30 == 100.0


def test_cobertura_insuficiente_deja_nan():
    """
    Con menos de 15 valores no nulos en la ventana de 21 dias, el baseline de
    ese dia no se calcula: ese dia queda no evaluable en esa dimension.
    """
    valores = [np.nan] * 40
    for i in range(0, 10):
        valores[i] = 100.0

    df = pd.DataFrame({
        "pid": ["A"] * 40,
        COLUMNA: valores,
    })

    baseline = compute_baseline(df, COLUMNA)
    assert baseline["mean"].isna().all()


def test_calcular_ventana_rolling_sobre_una_serie():
    """La funcion de ventana opera sobre la serie de UNA persona."""
    serie = pd.Series([10.0] * 40)
    resultado = calcular_ventana_rolling(serie)

    assert list(resultado.columns) == ["mean", "std"]
    assert resultado["mean"].first_valid_index() == 21
