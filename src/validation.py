"""
validation.py

Validacion de que las alertas del sistema son señal real y no ruido, y de
que llegan a tiempo. Extraido del notebook globem_es.ipynb, Bloque 7.

Sirve dos propositos (sección 26.4 de la consolidación):
  1. Alimenta el semaforo señal/ruido de la Parte 2 de la app (A.2): compara
     los dias-alerta reales contra los dias-alerta sobre ruido, para
     cualquier configuracion de umbral.
  2. Permite que los tests de pytest prueben CUSUM/EWMA de verdad (Capa 4
     del portfolio), no solo mockeados.

Contiene:
  - simulate_white_noise / simulate_temporal_permutation (celdas 312, 315):
    cuantas alertas produce la regla sobre datos sin estructura conductual.
  - detect_cusum / detect_ewma (celda 319): dos detectores de cambio
    independientes de la regla del sistema.
  - corroboration_rate (celda 319): cuantos de los episodios del sistema
    tienen respaldo de CUSUM o EWMA en una ventana de +-5 dias.
  - freshness (celda 324): cuando llega la alarma al terapeuta respecto al
    cierre real de cada episodio.
"""

import numpy as np
import pandas as pd

from .rule import apply_rule
from .episodes import build_episodes_with_end


def simulate_white_noise(df_rolling, n_reps=100, seed=0):
    """
    Genera n_reps replicas de ruido blanco N(0,1): para cada persona y
    dimension, sustituye cada valor no-NaN por un valor aleatorio de una
    normal estandar (media 0, desviacion 1), conservando los NaN en su
    sitio. Aplica apply_rule a cada replica.

    Devuelve (days_arr, eps_arr): arrays de longitud n_reps con el numero
    de dias-alerta y de episodios de cada replica.

    Sobre los datos reales del notebook, las 100 replicas dan siempre 0
    dias-alerta (celda 313): la regla no se activa sobre puro azar.
    """
    np.random.seed(seed)

    white_days = []
    white_eps = []

    for rep in range(n_reps):
        df_sim = df_rolling.copy()
        for dim in ["sleep", "steps", "location", "screen"]:
            v_real = df_rolling[dim].values
            v_sim = []
            for i in range(len(v_real)):
                if np.isnan(v_real[i]):
                    v_sim.append(np.nan)
                else:
                    v_sim.append(np.random.normal(0, 1))
            df_sim[dim] = v_sim
        n_d, n_e = apply_rule(df_sim)
        white_days.append(n_d)
        white_eps.append(n_e)

    return np.array(white_days), np.array(white_eps)


def simulate_temporal_permutation(df_rolling, n_reps=100, seed=0):
    """
    Genera n_reps replicas de permutacion temporal: para cada persona y
    dimension, conserva los valores reales que esa persona produjo (mismos
    extremos, misma distribucion) pero baraja el orden en que aparecen.
    Aplica apply_rule a cada replica.

    Devuelve (days_arr, eps_arr): arrays de longitud n_reps con el numero
    de dias-alerta y de episodios de cada replica.

    Sobre los datos reales del notebook, la media es 0.1 dias-alerta por
    replica, maximo 2 (celda 316): sin la secuencia temporal real, la señal
    practicamente desaparece.
    """
    rng = np.random.default_rng(seed)

    perm_days = []
    perm_eps = []

    for rep in range(n_reps):
        df_sim = df_rolling.copy()
        for dim in ["sleep", "steps", "location", "screen"]:
            new_values = []
            for pid, sub in df_rolling.groupby(level="pid", sort=False):
                values = sub[dim].values.copy()
                mask_no_nan = ~np.isnan(values)
                non_nan_values = values[mask_no_nan]
                shuffled_values = rng.permutation(non_nan_values)
                values[mask_no_nan] = shuffled_values
                new_values.extend(values.tolist())
            df_sim[dim] = new_values
        n_d, n_e = apply_rule(df_sim)
        perm_days.append(n_d)
        perm_eps.append(n_e)

    return np.array(perm_days), np.array(perm_eps)


def detect_cusum(series, k=0.5, h=4.0):
    """
    CUSUM bilateral con holgura k y umbral h. Acumula desviaciones respecto
    a 0; cuando la suma acumulada (positiva o negativa) supera el umbral h,
    marca alarma y reinicia el acumulador. Un NaN en la serie reinicia el
    acumulador (la racha de datos se corta ahi).

    Devuelve la lista de posiciones (indices dentro de 'series') donde
    dispara alarma.
    """
    n = len(series)
    s_pos = 0.0
    s_neg = 0.0
    alarms = []
    for i in range(n):
        v = series[i]
        if np.isnan(v):
            s_pos = 0.0
            s_neg = 0.0
            continue
        s_pos = max(0.0, s_pos + v - k)
        s_neg = min(0.0, s_neg + v + k)
        if s_pos > h or s_neg < -h:
            alarms.append(i)
            s_pos = 0.0
            s_neg = 0.0
    return alarms


def detect_ewma(series, lam=0.2, L=2.5):
    """
    EWMA (media movil con peso exponencial decreciente). lam controla cuanto
    pesa el valor mas reciente; L*sigma_ewma es el umbral de alarma. Un NaN
    reinicia el promedio (se re-arranca con el siguiente valor valido).

    Devuelve la lista de posiciones (indices dentro de 'series') donde
    dispara alarma.
    """
    n = len(series)
    ewma = 0.0
    sigma_ewma = np.sqrt(lam / (2 - lam))
    alarms = []
    initialized = False
    for i in range(n):
        v = series[i]
        if np.isnan(v):
            ewma = 0.0
            initialized = False
            continue
        if not initialized:
            ewma = v
            initialized = True
        else:
            ewma = lam * v + (1 - lam) * ewma
        if abs(ewma) > L * sigma_ewma:
            alarms.append(i)
    return alarms


def compute_cusum_ewma_alarms(df_rolling):
    """
    Aplica detect_cusum y detect_ewma a cada persona y cada una de las 4
    dimensiones de df_rolling. Devuelve (cusum_alarms, ewma_alarms): dos
    diccionarios {pid: set(idx originales con alarma)}, uniendo las alarmas
    de las 4 dimensiones para cada persona.
    """
    all_pids = df_rolling.index.get_level_values("pid").unique()

    cusum_alarms = {}
    ewma_alarms = {}

    for pid in all_pids:
        sub = df_rolling.loc[pid].sort_index()
        indices = sub.index.values

        cusum_set = set()
        ewma_set = set()

        for dim in ["sleep", "steps", "location", "screen"]:
            series = sub[dim].values
            a_c = detect_cusum(series)
            a_e = detect_ewma(series)
            for i in a_c:
                cusum_set.add(indices[i])
            for i in a_e:
                ewma_set.add(indices[i])

        cusum_alarms[pid] = cusum_set
        ewma_alarms[pid] = ewma_set

    return cusum_alarms, ewma_alarms


def corroboration_rate(df_episodes, cusum_alarms, ewma_alarms, window=5):
    """
    Para cada episodio de df_episodes (columnas 'pid', 'start_idx'), comprueba
    si hay alguna alarma de CUSUM o de EWMA para esa persona dentro de una
    ventana de +-window dias alrededor de start_idx.

    Devuelve un diccionario con match_cusum, match_ewma, match_any (numero de
    episodios con coincidencia de cada tipo) y n_episodes (total).

    Sobre los datos reales del notebook, match_any/n_episodes da 100%
    (63/63): cada episodio del sistema esta corroborado por al menos una
    tecnica independiente.
    """
    match_cusum = 0
    match_ewma = 0
    match_any = 0

    for i in range(len(df_episodes)):
        episode_pid = df_episodes.loc[i, "pid"]
        start_idx = df_episodes.loc[i, "start_idx"]

        window_low = start_idx - window
        window_high = start_idx + window

        has_c = False
        has_e = False
        for ix in cusum_alarms.get(episode_pid, set()):
            if window_low <= ix <= window_high:
                has_c = True
                break
        for ix in ewma_alarms.get(episode_pid, set()):
            if window_low <= ix <= window_high:
                has_e = True
                break

        if has_c:
            match_cusum = match_cusum + 1
        if has_e:
            match_ewma = match_ewma + 1
        if has_c or has_e:
            match_any = match_any + 1

    n_eps = len(df_episodes)
    return {
        "match_cusum": match_cusum,
        "match_ewma": match_ewma,
        "match_any": match_any,
        "n_episodes": n_eps,
    }


def freshness(df_convergence):
    """
    Para cada episodio (reconstruido con build_episodes_with_end), mide
    cuando llega la alarma al terapeuta (el dia siguiente al start_idx)
    respecto al final real de la racha:
      - 'ongoing': la racha sigue activa cuando llega la alarma.
      - 'same_day': la racha termina el mismo dia en que llega la alarma.
      - 'already_ended': la racha ya habia terminado; se guarda cuantos
        dias de retraso hay en 'days_since_end'.

    Devuelve un diccionario con ongoing, same_day, already_ended,
    days_since_end (lista) y n_episodes.

    Sobre los datos reales del notebook: 31 ongoing, 11 same_day, 21
    already_ended, todos con exactamente 1 dia de retraso (celda 325).
    """
    df_episodes_end = build_episodes_with_end(df_convergence)

    ongoing = 0
    same_day = 0
    already_ended = 0
    days_since_end = []

    for k in range(len(df_episodes_end)):
        pid = df_episodes_end.loc[k, "pid"]
        sub = df_convergence.loc[pid].sort_index()
        indices = sub.index.values

        start_pos = np.where(indices == df_episodes_end.loc[k, "start_idx"])[0][0]
        end_pos = np.where(indices == df_episodes_end.loc[k, "end_idx"])[0][0]
        fire_pos = start_pos + 1

        if end_pos > fire_pos:
            ongoing = ongoing + 1
        elif end_pos == fire_pos:
            same_day = same_day + 1
        else:
            already_ended = already_ended + 1
            days_since_end.append(fire_pos - end_pos)

    return {
        "ongoing": ongoing,
        "same_day": same_day,
        "already_ended": already_ended,
        "days_since_end": days_since_end,
        "n_episodes": len(df_episodes_end),
    }
