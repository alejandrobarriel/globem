"""
clinical_notice.py

Construccion del aviso que el sistema entrega al terapeuta: para cada
episodio, la descripcion en lenguaje llano de lo que le paso a esa persona el
dia en que se confirmo la desviacion.

Extraido del notebook globem_es.ipynb, seccion 6.3.5 (celda 302).

Decision metodologica (notebook, seccion 6.3.5): el aviso contiene UNICAMENTE
la descripcion del patron observado el dia de arranque, sin la etiqueta de
tipo del agrupamiento. El tipo es una construccion geometrica util para
describir el conjunto de episodios, pero no mejora la lectura clinica de un
caso individual: dentro de un mismo tipo conviven dias con desviaciones de
signo distinto, porque el agrupamiento responde a proximidad en el espacio de
las tres dimensiones, no a coincidencia de signos. Entregar solo la
informacion del dia es la opcion mas limpia y honesta.

Sobre los datos reales con 2.0/5: 63 avisos, de los cuales 16 tienen alguna
dimension sin medir el dia de arranque. Esos avisos se entregan igualmente,
con la marca explicita de que dimension falta: la deteccion es real porque la
regla de convergencia se cumplio, solo se renuncia a describir esa dimension.
"""

import numpy as np
import pandas as pd


def describe_z(z):
    """
    Traduce un valor z a una descripcion verbal en siete niveles, mas el caso
    de dimension no medida.

    Los umbrales estan en la propia escala z (notebook, celda 302).
    """
    if np.isnan(z):
        return "no medido ese dia"
    if z >= 2:
        return "mucho mas alto de lo habitual"
    elif z >= 1:
        return "claramente mas alto de lo habitual"
    elif z >= 0.5:
        return "ligeramente por encima de lo habitual"
    elif z > -0.5:
        return "aproximadamente como lo habitual"
    elif z > -1:
        return "ligeramente por debajo de lo habitual"
    elif z > -2:
        return "claramente mas bajo de lo habitual"
    else:
        return "mucho mas bajo de lo habitual"


def compute_z_activation(z_steps, z_location, pers_steps, pers_location):
    """
    Calcula el z de activacion del dia, siguiendo la misma logica con la que
    la regla compuso el dominio (steps OR location).

    Si las dos señales sostuvieron la desviacion, se toma la de mayor
    magnitud. Si solo una la sostuvo, se toma esa. Si ninguna la sostuvo, se
    toma la de mayor magnitud entre las disponibles. Si no hay ninguna
    disponible, NaN.

    Extraido del notebook, celda 302.
    """
    if pers_steps == 1.0 and pers_location == 1.0:
        if not np.isnan(z_steps) and (np.isnan(z_location) or abs(z_steps) >= abs(z_location)):
            return z_steps
        else:
            return z_location
    elif pers_steps == 1.0:
        return z_steps
    elif pers_location == 1.0:
        return z_location
    else:
        if np.isnan(z_steps) and np.isnan(z_location):
            return np.nan
        elif np.isnan(z_steps):
            return z_location
        elif np.isnan(z_location):
            return z_steps
        else:
            if abs(z_steps) >= abs(z_location):
                return z_steps
            else:
                return z_location


def build_clinical_notices(df_episodes, df_rolling, df_persistence):
    """
    Construye un aviso por episodio, mirando los valores del dia de arranque.

    df_episodes: DataFrame de episodes.build_episodes ('pid', 'start_idx').
    df_rolling: df_z_roll_mean_w (los valores z suavizados y winsorizados).
    df_persistence: el indicador de persistencia de rule.build_persistence,
    necesario para saber que señal sostuvo la desviacion en activacion.

    Devuelve un DataFrame con una fila por episodio y cuatro columnas:
    'persona', 'sueno', 'actividad_fisica', 'uso_movil'.
    """
    notices = []

    for i in range(len(df_episodes)):
        episode_pid = df_episodes.loc[i, "pid"]
        start_idx = df_episodes.loc[i, "start_idx"]

        z_sleep = df_rolling.loc[(episode_pid, start_idx), "sleep"]
        z_steps = df_rolling.loc[(episode_pid, start_idx), "steps"]
        z_location = df_rolling.loc[(episode_pid, start_idx), "location"]
        z_screen = df_rolling.loc[(episode_pid, start_idx), "screen"]

        pers_steps = df_persistence.loc[(episode_pid, start_idx), "steps"]
        pers_location = df_persistence.loc[(episode_pid, start_idx), "location"]

        z_activation = compute_z_activation(z_steps, z_location, pers_steps, pers_location)

        notice = {
            "persona": episode_pid,
            "sueno": describe_z(z_sleep),
            "actividad_fisica": describe_z(z_activation),
            "uso_movil": describe_z(z_screen),
        }
        notices.append(notice)

    return pd.DataFrame(notices)


def count_incomplete_notices(df_clinical_notices):
    """
    Cuenta cuantos avisos tienen alguna dimension sin medir el dia de arranque.
    Sobre los datos reales con 2.0/5 son 16 de 63.
    """
    n_incomplete = 0
    for i in range(len(df_clinical_notices)):
        row = df_clinical_notices.iloc[i]
        if ("no medido" in row["sueno"]
                or "no medido" in row["actividad_fisica"]
                or "no medido" in row["uso_movil"]):
            n_incomplete = n_incomplete + 1
    return n_incomplete
