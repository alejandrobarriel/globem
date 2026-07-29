"""
test_episodes.py

Comprueba la traduccion de dias-alerta a episodios: deteccion de arranques,
separacion de rachas y cierre de rachas abiertas.
"""

import numpy as np
import pandas as pd

from src.episodes import (
    build_episodes,
    build_episodes_with_end,
    count_episodes,
    episodes_per_person,
)


def hacer_convergencia(valores, pid="A"):
    indice = pd.MultiIndex.from_product([[pid], range(len(valores))], names=["pid", "idx"])
    return pd.DataFrame({"convergence": valores}, index=indice)


def test_una_racha_continua_es_un_solo_episodio():
    """Diez dias seguidos de alerta son UN episodio, no diez notificaciones."""
    conv = hacer_convergencia([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    df_eps = build_episodes(conv)

    assert len(df_eps) == 1
    assert df_eps.loc[0, "start_idx"] == 1


def test_dos_rachas_separadas_son_dos_episodios():
    """
    El caso que motiva partir de la serie completa: dias-alerta en 1,2,3 y
    6,7 son dos rachas porque los dias 4 y 5 no tuvieron alerta.
    """
    conv = hacer_convergencia([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    df_eps = build_episodes(conv)

    assert len(df_eps) == 2
    assert list(df_eps["start_idx"]) == [1, 6]


def test_un_nan_tambien_rompe_la_racha():
    """Un dia no evaluable corta la racha igual que un dia sin alerta."""
    conv = hacer_convergencia([1.0, 1.0, np.nan, 1.0, 1.0])
    df_eps = build_episodes(conv)

    assert len(df_eps) == 2


def test_racha_que_empieza_en_el_primer_dia():
    conv = hacer_convergencia([1.0, 1.0, 0.0])
    df_eps = build_episodes(conv)

    assert len(df_eps) == 1
    assert df_eps.loc[0, "start_idx"] == 0


def test_racha_que_llega_hasta_el_ultimo_dia():
    """Una racha abierta al final del registro tambien cuenta como episodio."""
    conv = hacer_convergencia([0.0, 1.0, 1.0, 1.0])
    df_eps = build_episodes(conv)

    assert len(df_eps) == 1


def test_episodios_con_fin_guardan_el_ultimo_dia():
    conv = hacer_convergencia([0.0, 1.0, 1.0, 1.0, 0.0])
    df_eps = build_episodes_with_end(conv)

    assert len(df_eps) == 1
    assert df_eps.loc[0, "start_idx"] == 1
    assert df_eps.loc[0, "end_idx"] == 3


def test_episodios_con_fin_cierran_la_racha_abierta():
    """Si la racha llega al ultimo dia, end_idx es ese ultimo dia."""
    conv = hacer_convergencia([0.0, 1.0, 1.0])
    df_eps = build_episodes_with_end(conv)

    assert df_eps.loc[0, "end_idx"] == 2


def test_los_dos_constructores_detectan_los_mismos_episodios():
    conv = hacer_convergencia([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    df_sin_fin = build_episodes(conv)
    df_con_fin = build_episodes_with_end(conv)

    assert len(df_sin_fin) == len(df_con_fin)
    assert list(df_sin_fin["start_idx"]) == list(df_con_fin["start_idx"])


def test_episodios_de_varias_personas():
    filas = []
    for pid, valores in [("A", [1.0, 1.0, 0.0]), ("B", [0.0, 1.0, 1.0]), ("C", [0.0, 0.0, 0.0])]:
        for idx, v in enumerate(valores):
            filas.append({"pid": pid, "idx": idx, "convergence": v})

    conv = pd.DataFrame(filas).set_index(["pid", "idx"])
    df_eps = build_episodes(conv)

    assert len(df_eps) == 2
    assert set(df_eps["pid"]) == {"A", "B"}


def test_count_episodes_coincide_con_build_episodes():
    conv = hacer_convergencia([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert count_episodes(conv) == len(build_episodes(conv))


def test_distribucion_de_episodios_por_persona():
    filas = []
    for pid, valores in [
        ("A", [1.0, 0.0, 1.0]),   # 2 episodios
        ("B", [1.0, 1.0, 1.0]),   # 1 episodio
    ]:
        for idx, v in enumerate(valores):
            filas.append({"pid": pid, "idx": idx, "convergence": v})

    conv = pd.DataFrame(filas).set_index(["pid", "idx"])
    df_eps = build_episodes(conv)
    dist = episodes_per_person(df_eps)

    assert dist["con_1"] == 1
    assert dist["con_2"] == 1
    assert dist["personas_totales"] == 2


def test_sin_alertas_no_hay_episodios():
    conv = hacer_convergencia([0.0, 0.0, np.nan, 0.0])
    assert count_episodes(conv) == 0
