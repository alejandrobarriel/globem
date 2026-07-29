"""
test_zscore.py

Comprueba el z-score intra-sujeto: la formula, el caracter intra-sujeto y la
propagacion de los dias no evaluables.
"""

import numpy as np
import pandas as pd

from src.baseline import compute_baseline
from src.zscore import compute_zscore

from conftest import COLUMNA, PERSONAS, PERSONAS_CON_DESVIACION


def test_zscore_aplica_la_formula(dfs_sinteticos):
    """z = (observacion - media del baseline) / desviacion del baseline."""
    df = dfs_sinteticos["df_sleep"]
    baseline = compute_baseline(df, COLUMNA)
    z = compute_zscore(df, baseline, COLUMNA)

    esperado = (df[COLUMNA].values - baseline["mean"].values) / baseline["std"].values

    valido = ~np.isnan(esperado)
    assert np.allclose(z.values[valido], esperado[valido])


def test_zscore_es_nan_donde_el_baseline_no_es_calculable(dfs_sinteticos):
    """Si no hay baseline ese dia, tampoco hay z-score."""
    df = dfs_sinteticos["df_sleep"]
    baseline = compute_baseline(df, COLUMNA)
    z = compute_zscore(df, baseline, COLUMNA)

    sin_baseline = baseline["mean"].isna().values
    assert z.values[sin_baseline].size == 0 or np.isnan(z.values[sin_baseline]).all()


def test_el_zscore_no_mezcla_participantes_si_el_orden_no_es_alfabetico():
    """
    Test de regresion. groupby ordena los participantes alfabeticamente por
    defecto, pero el dataframe puede venir en otro orden. Si el z-score
    restara por posicion, la observacion de una persona se compararia contra
    el baseline de otra, en silencio.

    Aqui 'zeta' aparece primero en el dataframe pero es el ultimo
    alfabeticamente. Como cada persona tiene un valor constante, su z-score
    solo puede ser 0 o NaN: cualquier otro valor significa que se han
    cruzado los datos.
    """
    filas = []
    for pid, nivel in [("zeta", 100.0), ("alfa", 9000.0)]:
        for dia in range(40):
            filas.append({"pid": pid, COLUMNA: nivel})

    df = pd.DataFrame(filas)
    baseline = compute_baseline(df, COLUMNA)
    z = compute_zscore(df, baseline, COLUMNA)

    for valor in z.dropna().values:
        assert abs(valor) < 1e-6


def test_zscore_es_intra_sujeto():
    """
    Dos personas con niveles absolutos muy distintos pero la misma
    variabilidad relativa deben producir z-scores parecidos: el sistema
    compara a cada persona consigo misma, no entre personas.
    """
    filas = []
    np.random.seed(11)
    for pid, nivel in [("dorm_poco", 200.0), ("dorm_mucho", 3000.0)]:
        ruido = np.random.normal(0, 1, 60)
        for dia in range(60):
            # misma variabilidad RELATIVA: 5% del nivel de cada persona
            valor = nivel + ruido[dia] * (nivel * 0.05)
            filas.append({"pid": pid, COLUMNA: valor})

    df = pd.DataFrame(filas)
    baseline = compute_baseline(df, COLUMNA)
    z = compute_zscore(df, baseline, COLUMNA)

    z_bajo = z.loc["dorm_poco"].dropna()
    z_alto = z.loc["dorm_mucho"].dropna()

    # Las dos distribuciones de z deben moverse en un rango comparable
    assert abs(z_bajo.std() - z_alto.std()) < 1.0


def racha_mas_larga_por_debajo(serie, umbral=-2.0):
    """Cuenta el tramo mas largo de dias seguidos por debajo del umbral."""
    mas_larga = 0
    actual = 0
    for v in serie.values:
        if not np.isnan(v) and v < umbral:
            actual = actual + 1
            if actual > mas_larga:
                mas_larga = actual
        else:
            actual = 0
    return mas_larga


def test_la_desviacion_sintetica_produce_una_racha_sostenida(zscores_sinteticos):
    """
    Lo que distingue una desviacion real del ruido no es que aparezca algun
    dia por debajo de -2 (el ruido tambien produce dias asi de vez en cuando),
    sino que esos dias se SOSTENGAN. Las personas con la caida deben producir
    una racha larga; las que son ruido, no.
    """
    z_sleep = zscores_sinteticos["df_sleep"]

    for pid in PERSONAS_CON_DESVIACION:
        serie = z_sleep.loc[pid]
        assert racha_mas_larga_por_debajo(serie) >= 5

    for pid in PERSONAS:
        if pid in PERSONAS_CON_DESVIACION:
            continue
        serie = z_sleep.loc[pid]
        assert racha_mas_larga_por_debajo(serie) < 5
