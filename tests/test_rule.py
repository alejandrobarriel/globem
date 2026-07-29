"""
test_rule.py

Comprueba las cuatro capas de la regla: indicador diario, persistencia,
composicion de dominios y convergencia 2/3.
"""

import numpy as np
import pandas as pd

from src.rule import (
    build_daily_signal,
    build_persistence,
    build_pillars,
    build_convergence,
    run_rule,
    count_alert_days,
    count_people_with_alerts,
    apply_rule,
)
from src.episodes import build_episodes

from conftest import PERSONAS_CON_DESVIACION


def hacer_rolling(sleep, steps, location, screen, pid="A"):
    """Construye un df_rolling minimo con las series que se le pasen."""
    n = len(sleep)
    indice = pd.MultiIndex.from_product([[pid], range(n)], names=["pid", "idx"])
    return pd.DataFrame({
        "sleep": sleep,
        "steps": steps,
        "location": location,
        "screen": screen,
    }, index=indice)


def test_indicador_diario_marca_por_encima_del_umbral():
    """Marca 1 en valor absoluto: tanto por arriba como por abajo."""
    df = hacer_rolling(
        sleep=[0.0, 2.5, -2.5, 1.9, -1.9],
        steps=[0.0] * 5,
        location=[0.0] * 5,
        screen=[0.0] * 5,
    )
    señal = build_daily_signal(df, threshold=2.0)
    assert list(señal["sleep"].values) == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_indicador_diario_conserva_los_nan():
    df = hacer_rolling(
        sleep=[3.0, np.nan, 3.0],
        steps=[0.0] * 3,
        location=[0.0] * 3,
        screen=[0.0] * 3,
    )
    señal = build_daily_signal(df, threshold=2.0)
    assert np.isnan(señal["sleep"].values[1])


def test_persistencia_exige_cinco_dias_seguidos():
    """
    Con cinco dias consecutivos de señal, el quinto dia (indice 4) es el
    primero que marca persistencia.
    """
    señal = hacer_rolling(
        sleep=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        steps=[0.0] * 7,
        location=[0.0] * 7,
        screen=[0.0] * 7,
    )
    pers = build_persistence(señal, persist=5)
    assert list(pers["sleep"].values) == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_persistencia_descarta_una_racha_de_cuatro():
    """Cuatro dias seguidos no bastan: la desviacion no se considera sostenida."""
    señal = hacer_rolling(
        sleep=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        steps=[0.0] * 6,
        location=[0.0] * 6,
        screen=[0.0] * 6,
    )
    pers = build_persistence(señal, persist=5)
    assert not (pers["sleep"] == 1.0).any()


def test_persistencia_con_un_nan_en_la_ventana_da_nan():
    """
    Si falta un dia dentro de la ventana no se puede saber si la racha se
    sostuvo: el resultado es NaN, no 0.
    """
    señal = hacer_rolling(
        sleep=[1.0, 1.0, np.nan, 1.0, 1.0, 1.0],
        steps=[0.0] * 6,
        location=[0.0] * 6,
        screen=[0.0] * 6,
    )
    pers = build_persistence(señal, persist=5)
    assert np.isnan(pers["sleep"].values[4])
    assert np.isnan(pers["sleep"].values[5])


def test_persistencia_es_configurable():
    """La app permite explorar otras persistencias desde el panel izquierdo."""
    señal = hacer_rolling(
        sleep=[1.0, 1.0, 1.0, 0.0],
        steps=[0.0] * 4,
        location=[0.0] * 4,
        screen=[0.0] * 4,
    )
    pers_3 = build_persistence(señal, persist=3)
    assert pers_3["sleep"].values[2] == 1.0

    pers_5 = build_persistence(señal, persist=5)
    assert not (pers_5["sleep"] == 1.0).any()


def test_activacion_es_steps_o_location():
    """El dominio de activacion marca si CUALQUIERA de las dos lo sostuvo."""
    pers = hacer_rolling(
        sleep=[0.0] * 4,
        steps=[1.0, 0.0, 0.0, 1.0],
        location=[0.0, 1.0, 0.0, 1.0],
        screen=[0.0] * 4,
    )
    pilares = build_pillars(pers)
    assert list(pilares["activation"].values) == [1.0, 1.0, 0.0, 1.0]


def test_activacion_usa_la_señal_disponible_si_falta_la_otra():
    pers = hacer_rolling(
        sleep=[0.0] * 3,
        steps=[np.nan, 1.0, np.nan],
        location=[1.0, np.nan, np.nan],
        screen=[0.0] * 3,
    )
    pilares = build_pillars(pers)
    assert pilares["activation"].values[0] == 1.0
    assert pilares["activation"].values[1] == 1.0
    assert np.isnan(pilares["activation"].values[2])


def test_sueno_y_uso_pasivo_pasan_directos():
    pers = hacer_rolling(
        sleep=[1.0, 0.0],
        steps=[0.0, 0.0],
        location=[0.0, 0.0],
        screen=[0.0, 1.0],
    )
    pilares = build_pillars(pers)
    assert list(pilares["sleep"].values) == [1.0, 0.0]
    assert list(pilares["passive_use"].values) == [0.0, 1.0]


def test_convergencia_exige_dos_dominios():
    """Un solo dominio activo no basta; dos si."""
    indice = pd.MultiIndex.from_product([["A"], range(4)], names=["pid", "idx"])
    pilares = pd.DataFrame({
        "sleep": [1.0, 1.0, 0.0, 1.0],
        "activation": [0.0, 1.0, 0.0, 1.0],
        "passive_use": [0.0, 0.0, 0.0, 1.0],
    }, index=indice)

    conv = build_convergence(pilares)
    assert list(conv["convergence"].values) == [0.0, 1.0, 0.0, 1.0]


def test_convergencia_con_menos_de_dos_dominios_evaluables_es_nan():
    """
    Si ese dia solo hay un dominio evaluable, el dia no es evaluable para la
    regla. NaN no es lo mismo que 0: es que no se sabe.
    """
    indice = pd.MultiIndex.from_product([["A"], range(2)], names=["pid", "idx"])
    pilares = pd.DataFrame({
        "sleep": [1.0, 1.0],
        "activation": [np.nan, 1.0],
        "passive_use": [np.nan, np.nan],
    }, index=indice)

    conv = build_convergence(pilares)
    assert np.isnan(conv["convergence"].values[0])
    assert conv["convergence"].values[1] == 1.0


def test_la_regla_detecta_la_desviacion_sintetica(rolling_sintetico):
    """
    Solo las dos personas con desviacion real sostenida deben producir
    dias-alerta. Las otras cuatro, que son ruido, no.
    """
    resultado = run_rule(rolling_sintetico, threshold=2.0, persist=5)
    df_conv = resultado["convergence"]

    personas_con_alerta = set()
    for (pid, idx), v in df_conv["convergence"].items():
        if not np.isnan(v) and v == 1.0:
            personas_con_alerta.add(pid)

    assert personas_con_alerta == set(PERSONAS_CON_DESVIACION)


def test_run_rule_devuelve_las_cuatro_capas(rolling_sintetico):
    resultado = run_rule(rolling_sintetico)
    assert set(resultado.keys()) == {
        "daily_signal", "persistence", "pillars", "convergence"
    }


def test_apply_rule_coincide_con_las_capas(rolling_sintetico):
    """El atajo tiene que dar lo mismo que recorrer las capas a mano."""
    resultado = run_rule(rolling_sintetico)
    dias = count_alert_days(resultado["convergence"])
    episodios = len(build_episodes(resultado["convergence"]))

    n_dias, n_episodios = apply_rule(rolling_sintetico)

    assert n_dias == dias
    assert n_episodios == episodios


def test_umbral_mas_alto_produce_menos_alertas(rolling_sintetico):
    """
    Es la relacion que sostiene el semaforo de la app: aflojar el umbral
    produce mas alertas, apretarlo produce menos.
    """
    dias_laxo, _ = apply_rule(rolling_sintetico, threshold=1.5)
    dias_estricto, _ = apply_rule(rolling_sintetico, threshold=3.0)

    assert dias_laxo >= dias_estricto
