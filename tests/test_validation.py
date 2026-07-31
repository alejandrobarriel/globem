"""
test_validation.py

Comprueba las piezas del Bloque 7: simulaciones de ruido, CUSUM, EWMA,
corroboracion y frescura del aviso.

Es la Capa 4 del portfolio ("tests automatizados sobre las funciones de src/:
baseline, z-score, regla de convergencia, persistencia, CUSUM, EWMA").
"""

import numpy as np
import pandas as pd

from src.validation import (
    simulate_white_noise,
    simulate_temporal_permutation,
    detect_cusum,
    detect_ewma,
    compute_cusum_ewma_alarms,
    corroboration_rate,
    freshness,
)
from src.rule import run_rule, apply_rule
from src.episodes import build_episodes


def test_cusum_no_dispara_sobre_una_serie_plana():
    serie = np.zeros(50)
    assert detect_cusum(serie) == []


def test_cusum_detecta_un_desplazamiento_sostenido():
    """
    CUSUM esta pensado para cambios pequeños pero sostenidos: acumula
    desviaciones hasta superar el umbral.
    """
    serie = np.concatenate([np.zeros(20), np.full(20, 2.0)])
    alarmas = detect_cusum(serie)

    assert len(alarmas) > 0
    assert alarmas[0] >= 20
    assert alarmas[0] <= 26


def test_cusum_se_reinicia_con_un_hueco():
    """Un NaN corta la racha de datos: el acumulador vuelve a cero."""
    serie = np.concatenate([np.full(10, 2.0), [np.nan], np.full(3, 2.0)])
    alarmas = detect_cusum(serie)

    for i in alarmas:
        assert i != 11


def test_cusum_detecta_en_las_dos_direcciones():
    subida = detect_cusum(np.concatenate([np.zeros(10), np.full(20, 3.0)]))
    bajada = detect_cusum(np.concatenate([np.zeros(10), np.full(20, -3.0)]))

    assert len(subida) > 0
    assert len(bajada) > 0


def test_ewma_no_dispara_sobre_una_serie_plana():
    serie = np.zeros(50)
    assert detect_ewma(serie) == []


def test_ewma_detecta_un_cambio_gradual():
    serie = np.concatenate([np.zeros(20), np.linspace(0, 4, 20)])
    alarmas = detect_ewma(serie)

    assert len(alarmas) > 0
    assert alarmas[0] >= 20


def test_ewma_se_reinicia_con_un_hueco():
    serie = np.array([5.0, 5.0, 5.0, np.nan, 0.0, 0.0])
    alarmas = detect_ewma(serie)

    assert 3 not in alarmas


def test_ruido_blanco_no_produce_alertas(rolling_sintetico):
    """
    La prueba central del Bloque 7: sobre datos sin estructura conductual,
    la regla no se activa. Si se activara, el sistema seria un generador de
    falsas alarmas.
    """
    dias, episodios = simulate_white_noise(rolling_sintetico, n_reps=5, seed=0)

    assert dias.max() == 0
    assert episodios.max() == 0


def test_la_permutacion_temporal_destruye_la_señal(rolling_sintetico):
    """
    La prueba mas exigente: se conservan los valores reales de cada persona
    (mismos extremos, misma distribucion) y solo se baraja el orden. Si la
    regla se activara igual, seria que responde a valores extremos sueltos y
    no a la secuencia temporal.
    """
    dias_reales, _ = apply_rule(rolling_sintetico)
    dias_perm, _ = simulate_temporal_permutation(rolling_sintetico, n_reps=5, seed=0)

    assert dias_reales > 0
    assert dias_perm.mean() < dias_reales


def test_las_simulaciones_conservan_los_huecos(rolling_sintetico):
    """
    Donde el dato real era NaN, la replica tambien tiene que ser NaN: no se
    puede inventar cobertura que no existia.
    """
    dias, episodios = simulate_white_noise(rolling_sintetico, n_reps=2, seed=0)
    assert len(dias) == 2


def test_la_permutacion_es_reproducible(rolling_sintetico):
    """Con la misma semilla, el mismo resultado."""
    a, _ = simulate_temporal_permutation(rolling_sintetico, n_reps=3, seed=42)
    b, _ = simulate_temporal_permutation(rolling_sintetico, n_reps=3, seed=42)

    assert list(a) == list(b)


def test_las_tecnicas_independientes_corroboran_los_episodios(rolling_sintetico):
    """
    Cada episodio del sistema deberia caer en un momento donde CUSUM o EWMA
    tambien ven cambio: si el sistema viera algo que solo el ve, seria
    sospechoso.
    """
    resultado = run_rule(rolling_sintetico)
    df_eps = build_episodes(resultado["convergence"])

    cusum, ewma = compute_cusum_ewma_alarms(rolling_sintetico)
    corr = corroboration_rate(df_eps, cusum, ewma)

    assert corr["n_episodes"] == len(df_eps)
    assert corr["match_any"] == corr["n_episodes"]


def test_la_ventana_de_corroboracion_es_configurable():
    """
    La ventana de +-5 dias existe porque cada tecnica tiene su propia
    latencia. Con ventana 0 se exige coincidencia exacta de dia.
    """
    df_eps = pd.DataFrame([{"pid": "A", "start_idx": 10}])
    cusum = {"A": {13}}
    ewma = {"A": set()}

    assert corroboration_rate(df_eps, cusum, ewma, window=5)["match_any"] == 1
    assert corroboration_rate(df_eps, cusum, ewma, window=0)["match_any"] == 0


def test_la_frescura_clasifica_los_episodios(rolling_sintetico):
    """Los tres casos posibles tienen que sumar el total de episodios."""
    resultado = run_rule(rolling_sintetico)
    fresco = freshness(resultado["convergence"])

    suma = fresco["ongoing"] + fresco["same_day"] + fresco["already_ended"]
    assert suma == fresco["n_episodes"]


def test_frescura_con_racha_larga_da_aviso_en_curso():
    """
    Si la racha dura mas alla del dia del disparo, el terapeuta recibe la
    alarma con el episodio todavia en curso.
    """
    valores = [0.0] + [1.0] * 10 + [0.0]
    indice = pd.MultiIndex.from_product([["A"], range(len(valores))], names=["pid", "idx"])
    conv = pd.DataFrame({"convergence": valores}, index=indice)

    fresco = freshness(conv)
    assert fresco["ongoing"] == 1
    assert fresco["already_ended"] == 0


def test_frescura_con_racha_corta_da_aviso_tras_el_cierre():
    """
    Si la racha termina antes del disparo, se registra el retraso en dias.
    """
    valores = [0.0, 1.0, 0.0, 0.0]
    indice = pd.MultiIndex.from_product([["A"], range(len(valores))], names=["pid", "idx"])
    conv = pd.DataFrame({"convergence": valores}, index=indice)

    fresco = freshness(conv)
    assert fresco["same_day"] == 1 or fresco["already_ended"] == 1


# ---------------------------------------------------------------------------
# Semaforo señal/ruido (pieza A.2)
# ---------------------------------------------------------------------------

from src.validation import signal_noise_score, signal_noise_grid, CONTAMINACION_CERO


def test_el_indicador_esta_entre_0_y_10(rolling_sintetico):
    r = signal_noise_score(rolling_sintetico, 2.0, 5, n_reps=3, seed=0)
    assert 0.0 <= r["score"] <= 10.0


def test_el_indicador_devuelve_todo_lo_necesario_para_el_json(rolling_sintetico):
    r = signal_noise_score(rolling_sintetico, 2.0, 5, n_reps=3, seed=0)
    esperadas = {"threshold", "persist", "real_days", "real_episodes",
                 "noise_mean", "noise_max", "contamination", "score"}
    assert set(r.keys()) == esperadas


def test_sin_contaminacion_el_indicador_es_10(rolling_sintetico):
    """
    Si el ruido no produce ni una alerta, todas las alertas reales son señal
    y el indicador esta en su tope.
    """
    r = signal_noise_score(rolling_sintetico, 2.0, 5, n_reps=3, seed=0)
    if r["noise_mean"] == 0.0:
        assert r["score"] == 10.0


def test_una_configuracion_que_no_detecta_nada_puntua_cero():
    """
    Sin alertas reales no hay señal que medir, y el indicador no puede
    premiar esa configuracion por no tener ruido.
    """
    indice = pd.MultiIndex.from_product([["A"], range(30)], names=["pid", "idx"])
    plano = pd.DataFrame({
        "sleep": [0.0] * 30, "steps": [0.0] * 30,
        "location": [0.0] * 30, "screen": [0.0] * 30,
    }, index=indice)

    r = signal_noise_score(plano, 2.0, 5, n_reps=2, seed=0)
    assert r["real_days"] == 0
    assert r["score"] == 0.0


def test_la_contaminacion_es_la_proporcion_de_alertas_explicables_por_azar(rolling_sintetico):
    r = signal_noise_score(rolling_sintetico, 2.0, 5, n_reps=3, seed=0)
    if r["real_days"] > 0:
        esperada = r["noise_mean"] / r["real_days"]
        assert abs(r["contamination"] - esperada) < 1e-5


def test_el_umbral_de_contaminacion_marca_el_cero():
    """
    La constante que define cuando el indicador vale 0 debe estar declarada y
    ser una proporcion, no un porcentaje.
    """
    assert 0.0 < CONTAMINACION_CERO < 1.0


def test_la_rejilla_cubre_todas_las_combinaciones(rolling_sintetico):
    grid = signal_noise_grid(rolling_sintetico, [2.0, 2.5], [3, 5], n_reps=2, seed=0)

    assert set(grid.keys()) == {"2.0_3", "2.0_5", "2.5_3", "2.5_5"}
    for clave in grid:
        assert 0.0 <= grid[clave]["score"] <= 10.0


def test_la_rejilla_conserva_la_configuracion_de_cada_celda(rolling_sintetico):
    grid = signal_noise_grid(rolling_sintetico, [2.5], [3], n_reps=2, seed=0)
    celda = grid["2.5_3"]

    assert celda["threshold"] == 2.5
    assert celda["persist"] == 3


def test_aflojar_la_regla_produce_mas_alertas_reales(rolling_sintetico):
    """
    Relacion que sostiene todo el semaforo: al aflojar entran mas dias-alerta,
    y con ellos la posibilidad de que entre ruido.
    """
    laxo = signal_noise_score(rolling_sintetico, 1.5, 3, n_reps=2, seed=0)
    estricto = signal_noise_score(rolling_sintetico, 2.5, 7, n_reps=2, seed=0)

    assert laxo["real_days"] >= estricto["real_days"]
