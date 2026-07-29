"""
test_real_data.py

Verificacion contra los CSV reales del estudio: comprueba que el src/
reproduce EXACTAMENTE las cifras ya verificadas en el notebook.

Estos son los tests que de verdad garantizan que la conversion del notebook a
modulos no rompio nada. Se saltan solos si los CSV no estan en datasets/, para
que la suite pueda correr igualmente en un entorno sin los datos.

Cifras de referencia (notebook, cierres de los Bloques 5, 6 y 7):
  - 155 participantes, 92 dias cada uno
  - BDI: 139 valores validos (51 con depresion, 88 sin)
  - umbral de winsorizacion de location: 24.07
  - 186 dias-alerta sobre 5415 dias evaluables (3.4%)
  - 63 episodios en 56 personas (49 con uno, 7 con dos, 0 con tres o mas)
  - 63 avisos clinicos, 16 con alguna dimension sin medir
  - ruido blanco: 0 dias-alerta
  - CUSUM/EWMA corroboran el 100% de los episodios
  - frescura: 31 en curso, 11 mismo dia, 21 ya terminados (todos con 1 dia)
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data_loading import load_and_prepare_all
from src.baseline import compute_all_baselines
from src.zscore import compute_all_zscores
from src.features import build_features
from src.rule import run_rule, count_alert_days, count_people_with_alerts
from src.episodes import build_episodes, episodes_per_person
from src.clinical_notice import build_clinical_notices, count_incomplete_notices
from src.validation import (
    simulate_white_noise,
    compute_cusum_ewma_alarms,
    corroboration_rate,
    freshness,
)


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")

ARCHIVOS = ["sleep.csv", "steps.csv", "location.csv", "screen.csv", "dep_endterm.csv"]


def hay_datos_reales():
    for archivo in ARCHIVOS:
        if not os.path.exists(os.path.join(DATA_DIR, archivo)):
            return False
    return True


necesita_datos = pytest.mark.skipif(
    not hay_datos_reales(),
    reason="Faltan los CSV del estudio en datasets/",
)


@pytest.fixture(scope="module")
def pipeline_real():
    """Ejecuta el pipeline completo una sola vez para todos los tests."""
    dfs, df_bdi, primary_columns = load_and_prepare_all(DATA_DIR)
    baselines = compute_all_baselines(dfs, primary_columns)
    zscores = compute_all_zscores(dfs, baselines, primary_columns)
    df_rolling, df_roll_std, threshold, n_clipped = build_features(zscores, dfs)
    resultado = run_rule(df_rolling, threshold=2.0, persist=5)

    return {
        "dfs": dfs,
        "df_bdi": df_bdi,
        "df_rolling": df_rolling,
        "winsor_threshold": threshold,
        "n_clipped": n_clipped,
        "reglas": resultado,
    }


@necesita_datos
def test_la_cohorte_tiene_155_participantes_y_92_dias(pipeline_real):
    for nombre, df in pipeline_real["dfs"].items():
        assert df["pid"].nunique() == 155
        assert df["day"].min() == 0
        assert df["day"].max() == 91


@necesita_datos
def test_el_bdi_tiene_139_validos_51_con_depresion(pipeline_real):
    df_bdi = pipeline_real["df_bdi"]
    validos = df_bdi[df_bdi["BDI2"].notna()]

    assert len(validos) == 139
    assert (validos["dep"] == True).sum() == 51
    assert (validos["dep"] == False).sum() == 88


@necesita_datos
def test_el_umbral_de_winsorizacion_es_24_07(pipeline_real):
    assert round(pipeline_real["winsor_threshold"], 2) == 24.07


@necesita_datos
def test_la_regla_produce_186_dias_alerta(pipeline_real):
    df_conv = pipeline_real["reglas"]["convergence"]
    assert count_alert_days(df_conv) == 186


@necesita_datos
def test_hay_5415_dias_evaluables(pipeline_real):
    df_conv = pipeline_real["reglas"]["convergence"]
    assert df_conv["convergence"].notna().sum() == 5415


@necesita_datos
def test_las_alertas_afectan_a_56_personas(pipeline_real):
    df_conv = pipeline_real["reglas"]["convergence"]
    assert count_people_with_alerts(df_conv) == 56


@necesita_datos
def test_hay_63_episodios(pipeline_real):
    df_conv = pipeline_real["reglas"]["convergence"]
    df_eps = build_episodes(df_conv)
    assert len(df_eps) == 63


@necesita_datos
def test_la_distribucion_de_episodios_por_persona(pipeline_real):
    """49 personas con un episodio, 7 con dos, ninguna con tres o mas."""
    df_conv = pipeline_real["reglas"]["convergence"]
    df_eps = build_episodes(df_conv)
    dist = episodes_per_person(df_eps)

    assert dist["con_1"] == 49
    assert dist["con_2"] == 7
    assert dist["con_3_o_mas"] == 0
    assert dist["personas_totales"] == 56


@necesita_datos
def test_hay_63_avisos_16_incompletos(pipeline_real):
    df_conv = pipeline_real["reglas"]["convergence"]
    df_eps = build_episodes(df_conv)
    avisos = build_clinical_notices(
        df_eps,
        pipeline_real["df_rolling"],
        pipeline_real["reglas"]["persistence"],
    )

    assert len(avisos) == 63
    assert count_incomplete_notices(avisos) == 16


@necesita_datos
def test_el_ruido_blanco_no_produce_ni_una_alerta(pipeline_real):
    """
    La cifra mas contundente del Bloque 7: cero alertas sobre puro azar,
    frente a 186 sobre datos reales.
    """
    dias, episodios = simulate_white_noise(pipeline_real["df_rolling"], n_reps=10, seed=0)

    assert dias.max() == 0
    assert episodios.max() == 0


@necesita_datos
def test_cusum_y_ewma_corroboran_el_100_por_cien(pipeline_real):
    df_conv = pipeline_real["reglas"]["convergence"]
    df_eps = build_episodes(df_conv)

    cusum, ewma = compute_cusum_ewma_alarms(pipeline_real["df_rolling"])
    corr = corroboration_rate(df_eps, cusum, ewma)

    assert corr["n_episodes"] == 63
    assert corr["match_any"] == 63


@necesita_datos
def test_la_frescura_del_aviso(pipeline_real):
    """31 en curso, 11 el mismo dia, 21 ya terminados con 1 dia de retraso."""
    df_conv = pipeline_real["reglas"]["convergence"]
    fresco = freshness(df_conv)

    assert fresco["n_episodes"] == 63
    assert fresco["ongoing"] == 31
    assert fresco["same_day"] == 11
    assert fresco["already_ended"] == 21

    for retraso in fresco["days_since_end"]:
        assert retraso == 1
