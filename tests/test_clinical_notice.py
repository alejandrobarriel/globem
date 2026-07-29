"""
test_clinical_notice.py

Comprueba la traduccion del z-score a lenguaje llano y la construccion del
aviso al terapeuta.
"""

import numpy as np
import pandas as pd

from src.clinical_notice import (
    describe_z,
    compute_z_activation,
    build_clinical_notices,
    count_incomplete_notices,
)


def test_los_siete_niveles_de_descripcion():
    assert describe_z(3.0) == "mucho mas alto de lo habitual"
    assert describe_z(1.5) == "claramente mas alto de lo habitual"
    assert describe_z(0.7) == "ligeramente por encima de lo habitual"
    assert describe_z(0.0) == "aproximadamente como lo habitual"
    assert describe_z(-0.7) == "ligeramente por debajo de lo habitual"
    assert describe_z(-1.5) == "claramente mas bajo de lo habitual"
    assert describe_z(-3.0) == "mucho mas bajo de lo habitual"


def test_dimension_no_medida():
    """
    Si esa dimension no estaba medida ese dia, se dice explicitamente. El
    aviso se entrega igual: la deteccion es real porque la convergencia se
    cumplio con las otras dimensiones.
    """
    assert describe_z(np.nan) == "no medido ese dia"


def test_los_limites_exactos_de_cada_nivel():
    """Los umbrales son cerrados por arriba y abiertos por abajo."""
    assert describe_z(2.0) == "mucho mas alto de lo habitual"
    assert describe_z(1.999) == "claramente mas alto de lo habitual"
    assert describe_z(-2.0) == "mucho mas bajo de lo habitual"
    assert describe_z(-1.999) == "claramente mas bajo de lo habitual"


def test_activacion_toma_la_señal_que_sostuvo_la_desviacion():
    """Si solo pasos sostuvo la racha, se describe pasos."""
    z = compute_z_activation(z_steps=-3.0, z_location=0.2, pers_steps=1.0, pers_location=0.0)
    assert z == -3.0

    z = compute_z_activation(z_steps=0.2, z_location=-3.0, pers_steps=0.0, pers_location=1.0)
    assert z == -3.0


def test_activacion_toma_la_mayor_si_las_dos_sostuvieron():
    z = compute_z_activation(z_steps=-2.5, z_location=-4.0, pers_steps=1.0, pers_location=1.0)
    assert z == -4.0


def test_activacion_es_nan_si_no_hay_ninguna_señal():
    z = compute_z_activation(np.nan, np.nan, 0.0, 0.0)
    assert np.isnan(z)


def test_activacion_usa_la_disponible_si_falta_una():
    z = compute_z_activation(z_steps=np.nan, z_location=-2.0, pers_steps=0.0, pers_location=0.0)
    assert z == -2.0


def test_el_aviso_tiene_una_fila_por_episodio():
    indice = pd.MultiIndex.from_product([["A"], range(5)], names=["pid", "idx"])
    df_rolling = pd.DataFrame({
        "sleep": [-3.0] * 5,
        "steps": [-2.5] * 5,
        "location": [0.1] * 5,
        "screen": [0.2] * 5,
    }, index=indice)
    df_pers = pd.DataFrame({
        "sleep": [1.0] * 5,
        "steps": [1.0] * 5,
        "location": [0.0] * 5,
        "screen": [0.0] * 5,
    }, index=indice)

    df_eps = pd.DataFrame([{"pid": "A", "start_idx": 2}])
    avisos = build_clinical_notices(df_eps, df_rolling, df_pers)

    assert len(avisos) == 1
    assert list(avisos.columns) == ["persona", "sueno", "actividad_fisica", "uso_movil"]


def test_el_aviso_describe_el_dia_de_arranque():
    """
    El aviso mira el dia de arranque, no un promedio del episodio: el
    terapeuta necesita informacion del momento del aviso.
    """
    indice = pd.MultiIndex.from_product([["A"], range(4)], names=["pid", "idx"])
    df_rolling = pd.DataFrame({
        "sleep": [0.0, 0.0, -3.0, 0.0],   # solo el dia 2 esta desviado
        "steps": [0.0] * 4,
        "location": [0.0] * 4,
        "screen": [0.0] * 4,
    }, index=indice)
    df_pers = pd.DataFrame({
        "sleep": [1.0] * 4,
        "steps": [0.0] * 4,
        "location": [0.0] * 4,
        "screen": [0.0] * 4,
    }, index=indice)

    df_eps = pd.DataFrame([{"pid": "A", "start_idx": 2}])
    avisos = build_clinical_notices(df_eps, df_rolling, df_pers)

    assert avisos.loc[0, "sueno"] == "mucho mas bajo de lo habitual"


def test_el_aviso_se_entrega_aunque_falte_una_dimension():
    indice = pd.MultiIndex.from_product([["A"], range(3)], names=["pid", "idx"])
    df_rolling = pd.DataFrame({
        "sleep": [-3.0] * 3,
        "steps": [np.nan] * 3,
        "location": [np.nan] * 3,
        "screen": [-2.5] * 3,
    }, index=indice)
    df_pers = pd.DataFrame({
        "sleep": [1.0] * 3,
        "steps": [np.nan] * 3,
        "location": [np.nan] * 3,
        "screen": [1.0] * 3,
    }, index=indice)

    df_eps = pd.DataFrame([{"pid": "A", "start_idx": 1}])
    avisos = build_clinical_notices(df_eps, df_rolling, df_pers)

    assert len(avisos) == 1
    assert avisos.loc[0, "actividad_fisica"] == "no medido ese dia"
    assert avisos.loc[0, "sueno"] == "mucho mas bajo de lo habitual"
    assert count_incomplete_notices(avisos) == 1


def test_el_aviso_no_lleva_etiqueta_de_tipo():
    """
    Decision de la seccion 6.3.5: el aviso lleva solo la descripcion del dia,
    sin el tipo del agrupamiento, porque dentro de un mismo tipo conviven
    dias con desviaciones de signo distinto.
    """
    indice = pd.MultiIndex.from_product([["A"], range(2)], names=["pid", "idx"])
    df_rolling = pd.DataFrame({
        "sleep": [-3.0] * 2, "steps": [-3.0] * 2,
        "location": [0.0] * 2, "screen": [0.0] * 2,
    }, index=indice)
    df_pers = pd.DataFrame({
        "sleep": [1.0] * 2, "steps": [1.0] * 2,
        "location": [0.0] * 2, "screen": [0.0] * 2,
    }, index=indice)

    df_eps = pd.DataFrame([{"pid": "A", "start_idx": 0}])
    avisos = build_clinical_notices(df_eps, df_rolling, df_pers)

    assert "tipo" not in avisos.columns
    assert "type" not in avisos.columns
