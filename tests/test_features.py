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


# ---------------------------------------------------------------------------
# Series por persona para la aplicacion (pieza A.3)
# ---------------------------------------------------------------------------

from src.features import (
    build_trajectory,
    build_episode_window,
    count_active_domains,
    global_index_to_day,
)
from src.rule import run_rule


def test_el_panel_izquierdo_recibe_todos_los_dias(rolling_sintetico):
    """Sin ventana, la trayectoria devuelve el registro completo."""
    r = run_rule(rolling_sintetico)
    tr = build_trajectory(rolling_sintetico, r["convergence"], "P1")

    assert tr["n_days"] == 92
    assert len(tr["days"]) == 92
    assert tr["days"][0] == 0
    assert tr["days"][-1] == 91


def test_los_dias_se_cuentan_desde_cero_en_cada_persona(rolling_sintetico):
    """
    El indice interno es global (P2 empieza donde acaba P1), pero la app
    necesita "dia 64 de 92" para cada persona por separado.
    """
    r = run_rule(rolling_sintetico)
    for pid in ["P1", "P2", "P3"]:
        tr = build_trajectory(rolling_sintetico, r["convergence"], pid)
        assert tr["days"][0] == 0


def test_los_dias_sin_dato_salen_como_nulo_no_como_cero(rolling_sintetico):
    """
    Un cero fingiria una observacion que no existe. Los primeros dias no
    tienen baseline calculable y deben salir vacios.
    """
    r = run_rule(rolling_sintetico)
    tr = build_trajectory(rolling_sintetico, r["convergence"], "P1")

    assert tr["series"]["sleep"][0] is None


def test_la_ventana_es_un_trozo_exacto_de_la_serie_completa(rolling_sintetico):
    """
    Clave del diseño: las dos graficas de la app salen de la MISMA serie. La
    del panel derecho es un recorte, no un calculo distinto.
    """
    r = run_rule(rolling_sintetico)
    completa = build_trajectory(rolling_sintetico, r["convergence"], "P1")
    ventana = build_trajectory(rolling_sintetico, r["convergence"], "P1", 50, 60)

    for dim in ["sleep", "steps", "location", "screen"]:
        trozo = completa["series"][dim][50:61]
        assert trozo == ventana["series"][dim]


def test_la_ventana_devuelve_los_dias_pedidos(rolling_sintetico):
    r = run_rule(rolling_sintetico)
    ventana = build_trajectory(rolling_sintetico, r["convergence"], "P1", 50, 60)

    assert ventana["days"] == list(range(50, 61))
    assert ventana["day_from"] == 50
    assert ventana["day_to"] == 60


def test_la_ventana_no_se_sale_por_los_extremos(rolling_sintetico):
    """
    Si el episodio esta al principio o al final del registro, la ventana se
    recorta contra el limite en vez de pedir dias que no existen.
    """
    r = run_rule(rolling_sintetico)

    inicio = build_trajectory(rolling_sintetico, r["convergence"], "P1", -10, 5)
    assert inicio["day_from"] == 0

    final = build_trajectory(rolling_sintetico, r["convergence"], "P1", 85, 200)
    assert final["day_to"] == 91


def test_la_ventana_del_episodio_se_centra_en_el_arranque(rolling_sintetico):
    r = run_rule(rolling_sintetico)
    w = build_episode_window(rolling_sintetico, r["convergence"], "P1", 60, margen=7)

    assert w["day_from"] == 53
    assert w["day_to"] == 67


def test_los_dias_alerta_de_la_ventana_estan_dentro_de_ella(rolling_sintetico):
    r = run_rule(rolling_sintetico)
    w = build_trajectory(rolling_sintetico, r["convergence"], "P1", 50, 60)

    for dia in w["alert_days"]:
        assert 50 <= dia <= 60


def test_se_cuentan_los_dominios_activos(rolling_sintetico):
    """El '3 dominios' de la tarjeta de aviso."""
    r = run_rule(rolling_sintetico)
    conv = r["convergence"]

    dias_alerta = []
    for (pid, idx), v in conv["convergence"].items():
        if v == 1.0:
            dias_alerta.append((pid, idx))

    if len(dias_alerta) > 0:
        pid, idx = dias_alerta[0]
        n = count_active_domains(r["pillars"], pid, idx)
        # La regla exige convergencia 2 de 3, asi que un dia-alerta tiene 2 o 3
        assert n in (2, 3)


def test_el_indice_global_se_traduce_a_dia_relativo(rolling_sintetico):
    r = run_rule(rolling_sintetico)
    conv = r["convergence"]

    primer_indice = conv.loc["P2"].index.min()
    assert global_index_to_day(conv, "P2", primer_indice) == 0
    assert global_index_to_day(conv, "P2", primer_indice + 30) == 30
