"""
episodes.py

Traduccion de dias-alerta a episodios: un episodio es el bloque continuo de
dias-alerta consecutivos que corresponde a una misma desviacion sostenida.

Extraido del notebook globem_es.ipynb, seccion 6.3.3 (celda 295).

Por que importa la distincion (notebook, seccion 6.3.3): si una racha dura
diez dias, cada uno de esos diez dias es un dia-alerta, pero tratar cada dia
como una notificacion independiente saturaria al terapeuta con avisos
repetidos sobre el mismo cuadro. El momento que interesa es el arranque: el
dia en que el sistema confirma por primera vez que la desviacion se ha vuelto
persistente y multidimensional. Lo que viene despues es continuacion, no
deteccion nueva.

Sobre los datos reales con 2.0/5: los 186 dias-alerta se reducen a 63
episodios repartidos entre 56 personas (49 con un unico episodio, 7 con dos,
ninguna con tres o mas).
"""

import pandas as pd


def build_episodes(df_convergence):
    """
    Recorre persona por persona la serie de convergencia diaria y registra
    cada dia en que arranca una racha nueva.

    Un arranque es un dia con convergencia=1 cuyo dia previo NO tenia
    convergencia=1. Este criterio captura tanto el primer episodio de una
    persona como cualquier episodio posterior tras una pausa.

    Nota (notebook, seccion 6.3.3): se parte de df_convergence, que tiene la
    serie temporal COMPLETA de cada persona (dias con alerta, sin alerta y no
    evaluables), no de una lista de dias-alerta. Con solo la lista de dias que
    dispararon no se puede distinguir una racha continua de dos rachas
    separadas por una pausa.

    Devuelve un DataFrame con columnas 'pid' y 'start_idx'.
    """
    all_pids = df_convergence.index.get_level_values("pid").unique()

    episodes = []

    for pid in all_pids:
        person_sub = df_convergence.loc[pid].sort_index()
        conv_values = person_sub["convergence"].values
        indices = person_sub.index.values

        in_episode = False

        for i in range(len(conv_values)):
            current_value = conv_values[i]

            if current_value == 1.0 and not in_episode:
                in_episode = True
                start_idx = indices[i]
                episodes.append({"pid": pid, "start_idx": start_idx})
            elif current_value != 1.0 and in_episode:
                in_episode = False

    return pd.DataFrame(episodes)


def build_episodes_with_end(df_convergence):
    """
    Igual que build_episodes, pero guarda tambien el ultimo dia de cada racha.

    Necesario para medir la frescura del aviso (validation.freshness) y para
    dibujar la ventana del episodio en la mini-grafica de la app.

    Devuelve un DataFrame con columnas 'pid', 'start_idx' y 'end_idx'.
    """
    all_pids = df_convergence.index.get_level_values("pid").unique()

    episodes = []

    for pid in all_pids:
        person_sub = df_convergence.loc[pid].sort_index()
        conv_values = person_sub["convergence"].values
        indices = person_sub.index.values

        in_episode = False
        start_idx = None

        for i in range(len(conv_values)):
            if conv_values[i] == 1.0 and not in_episode:
                in_episode = True
                start_idx = indices[i]
            elif conv_values[i] != 1.0 and in_episode:
                in_episode = False
                episodes.append({
                    "pid": pid,
                    "start_idx": start_idx,
                    "end_idx": indices[i - 1],
                })

        # Si la racha llega hasta el ultimo dia del participante, cerrarla ahi
        if in_episode:
            episodes.append({
                "pid": pid,
                "start_idx": start_idx,
                "end_idx": indices[-1],
            })

    return pd.DataFrame(episodes)


def count_episodes(df_convergence):
    """
    Numero de episodios (arranques de racha).
    Sobre los datos reales con 2.0/5 son 63.
    """
    df_episodes = build_episodes(df_convergence)
    return len(df_episodes)


def episodes_per_person(df_episodes):
    """
    Distribucion de episodios por persona. Devuelve un diccionario con
    'con_1', 'con_2', 'con_3_o_mas' y 'personas_totales'.
    """
    if len(df_episodes) == 0:
        return {"con_1": 0, "con_2": 0, "con_3_o_mas": 0, "personas_totales": 0}

    conteo = df_episodes.groupby("pid").size()

    con_1 = 0
    con_2 = 0
    con_3_o_mas = 0
    for n in conteo.values:
        if n == 1:
            con_1 = con_1 + 1
        elif n == 2:
            con_2 = con_2 + 1
        else:
            con_3_o_mas = con_3_o_mas + 1

    return {
        "con_1": con_1,
        "con_2": con_2,
        "con_3_o_mas": con_3_o_mas,
        "personas_totales": len(conteo),
    }
