"""
conftest.py

Datos sinteticos compartidos por todos los tests.

Se construye una cohorte pequeña (6 personas, 92 dias, 4 dimensiones) con una
propiedad conocida de antemano: P1 y P2 tienen una desviacion sostenida real
en sueno y pasos entre los dias 50 y 70; las otras cuatro personas son ruido.
Eso permite comprobar que el sistema detecta a quien debe y no a quien no,
sin depender de los CSV reales del estudio.

Los tests que SI necesitan los CSV reales (para comprobar las cifras
verificadas del notebook: 186 dias-alerta, 63 episodios, 56 personas) estan
en test_real_data.py y se saltan solos si los CSV no estan presentes.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loading import prepare_day_column, clean_warmup_days
from src.baseline import compute_baseline
from src.zscore import compute_zscore
from src.features import build_zscore_dataframe, compute_rolling, winsorize_location


COLUMNA = "valor:7dhist"
PERSONAS = ["P1", "P2", "P3", "P4", "P5", "P6"]
PERSONAS_CON_DESVIACION = ["P1", "P2"]
DIA_INICIO_DESVIACION = 50
DIA_FIN_DESVIACION = 70


def construir_dimension(con_desviacion, semilla):
    """
    Construye el dataframe de una dimension: 6 personas x 92 dias, con la
    estructura minima que espera data_loading (columnas 'pid', 'date' y la
    feature primaria).
    """
    np.random.seed(semilla)
    filas = []
    for pid in PERSONAS:
        for dia in range(92):
            fecha = pd.Timestamp("2018-04-03") + pd.Timedelta(days=dia)
            valor = 1000 + np.random.normal(0, 50)
            if con_desviacion and pid in PERSONAS_CON_DESVIACION:
                if DIA_INICIO_DESVIACION <= dia <= DIA_FIN_DESVIACION:
                    valor = valor - 400
            filas.append({
                "pid": pid,
                "date": fecha.strftime("%Y-%m-%d"),
                COLUMNA: valor,
            })

    df = pd.DataFrame(filas)
    df = prepare_day_column(df)
    df = clean_warmup_days(df, COLUMNA)
    return df


@pytest.fixture
def dfs_sinteticos():
    """Las cuatro dimensiones. Solo sueno y pasos llevan la desviacion."""
    return {
        "df_sleep": construir_dimension(True, 1),
        "df_steps": construir_dimension(True, 2),
        "df_location": construir_dimension(False, 3),
        "df_screen": construir_dimension(False, 4),
    }


@pytest.fixture
def zscores_sinteticos(dfs_sinteticos):
    """El z-score de cada dimension."""
    zscores = {}
    for nombre in dfs_sinteticos:
        df = dfs_sinteticos[nombre]
        baseline = compute_baseline(df, COLUMNA)
        zscores[nombre] = compute_zscore(df, baseline, COLUMNA)
    return zscores


@pytest.fixture
def rolling_sintetico(zscores_sinteticos):
    """El rolling mean winsorizado: la entrada de la regla."""
    df_z = build_zscore_dataframe(zscores_sinteticos)
    df_roll_mean, df_roll_std = compute_rolling(df_z)
    df_w, threshold, n_clipped = winsorize_location(df_roll_mean)
    return df_w
