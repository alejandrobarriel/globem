"""
test_features.py

Comprueba el rolling mean/std del z-score y la winsorizacion de location.
"""

import numpy as np
import pandas as pd

from src.features import (
    build_zscore_dataframe,
    compute_rolling,
    compute_winsor_threshold,
    winsorize_location,
    DIMENSIONS,
)


def test_el_dataframe_de_z_tiene_las_cuatro_dimensiones(zscores_sinteticos):
    df_z = build_zscore_dataframe(zscores_sinteticos)
    assert list(df_z.columns) == DIMENSIONS


def test_rolling_necesita_la_ventana_completa(zscores_sinteticos):
    """
    Con min_periods igual al tamaño de ventana, los primeros 6 dias de cada
    persona no producen rolling mean.
    """
    df_z = build_zscore_dataframe(zscores_sinteticos)
    df_mean, df_std = compute_rolling(df_z, window=7)

    primer_z = df_z.loc["P1"]["sleep"].reset_index(drop=True).first_valid_index()
    primer_roll = df_mean.loc["P1"]["sleep"].reset_index(drop=True).first_valid_index()

    assert primer_roll == primer_z + 6


def test_rolling_suaviza_la_fluctuacion_diaria(zscores_sinteticos):
    """
    El rolling mean debe tener menos dispersion que la serie original: es lo
    que hace que un pico de un dia no arrastre la decision.
    """
    df_z = build_zscore_dataframe(zscores_sinteticos)
    df_mean, df_std = compute_rolling(df_z)

    for dim in DIMENSIONS:
        original = df_z[dim].dropna()
        suavizado = df_mean[dim].dropna()
        assert suavizado.std() < original.std()


def test_winsorizacion_recorta_por_arriba_y_por_abajo():
    """
    Los valores por encima del umbral se fijan en el umbral, y los que estan
    por debajo de -umbral se fijan en -umbral.
    """
    indice = pd.MultiIndex.from_product([["A"], range(10)], names=["pid", "idx"])
    df = pd.DataFrame({
        "sleep": [0.0] * 10,
        "steps": [0.0] * 10,
        "location": [0.0, 1.0, 2.0, 3.0, 50.0, -50.0, -3.0, -2.0, -1.0, 0.0],
        "screen": [0.0] * 10,
    }, index=indice)

    df_w, umbral, n_recortados = winsorize_location(df, threshold=3.0)

    valores = df_w["location"].values
    assert valores.max() <= 3.0
    assert valores.min() >= -3.0
    assert n_recortados == 2


def test_winsorizacion_no_toca_las_otras_dimensiones():
    indice = pd.MultiIndex.from_product([["A"], range(5)], names=["pid", "idx"])
    df = pd.DataFrame({
        "sleep": [100.0] * 5,
        "steps": [-100.0] * 5,
        "location": [50.0] * 5,
        "screen": [100.0] * 5,
    }, index=indice)

    df_w, umbral, n_recortados = winsorize_location(df, threshold=1.0)

    assert (df_w["sleep"] == 100.0).all()
    assert (df_w["steps"] == -100.0).all()
    assert (df_w["screen"] == 100.0).all()


def test_winsorizacion_conserva_los_nan():
    indice = pd.MultiIndex.from_product([["A"], range(4)], names=["pid", "idx"])
    df = pd.DataFrame({
        "sleep": [0.0] * 4,
        "steps": [0.0] * 4,
        "location": [1.0, np.nan, 99.0, np.nan],
        "screen": [0.0] * 4,
    }, index=indice)

    df_w, umbral, n_recortados = winsorize_location(df, threshold=5.0)

    assert np.isnan(df_w["location"].values[1])
    assert np.isnan(df_w["location"].values[3])


def test_umbral_es_el_percentil_95_de_los_absolutos():
    indice = pd.MultiIndex.from_product([["A"], range(100)], names=["pid", "idx"])
    valores = list(range(100))
    df = pd.DataFrame({
        "sleep": [0.0] * 100,
        "steps": [0.0] * 100,
        "location": [float(v) for v in valores],
        "screen": [0.0] * 100,
    }, index=indice)

    umbral = compute_winsor_threshold(df, "location", 0.95)
    assert umbral == 95.0
