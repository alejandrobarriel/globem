"""
rule.py

La regla operativa del sistema, en las cuatro capas del Bloque 5:

  5.4  indicador diario      -> 1 si |rolling mean| supera el umbral
  5.4  persistencia          -> 1 solo si los N dias seguidos hasta hoy son 1
  5.5  composicion dominios  -> sueno, activacion (steps OR location), uso pasivo
  5.6  convergencia 2/3      -> 1 si al menos 2 dominios estan activos a la vez

Extraido del notebook globem_es.ipynb, celdas 225 (indicador diario), 227
(persistencia 5 dias), 230 (dominios) y 233 (convergencia). La funcion
apply_rule de la celda 309 hace exactamente lo mismo en un solo paso; aqui se
separa en capas para poder reutilizar los objetos intermedios (sobre todo
df_persistence_5d, que necesita clinical_notice.py).

Verificado contra los datos reales: umbral 2.0 y persistencia 5 producen 186
dias-alerta, 63 episodios y 56 personas.
"""

import numpy as np
import pandas as pd


DIMENSIONS = ["sleep", "steps", "location", "screen"]

# Punto de operacion decidido en la seccion 5.3 por criterio de uso
# (credibilidad ante el terapeuta), no por coincidencia con el BDI.
DEFAULT_THRESHOLD = 2.0
DEFAULT_PERSIST = 5


def build_daily_signal(df_rolling, threshold=DEFAULT_THRESHOLD):
    """
    Capa 1 (seccion 5.4): indicador diario.

    Marca 1 cuando el rolling mean de una dimension supera el umbral en valor
    absoluto (la persona se aleja de su patron habitual ese dia), 0 cuando no,
    y mantiene NaN donde no habia dato.

    Devuelve un DataFrame con las mismas columnas que df_rolling.
    """
    df_daily_signal = df_rolling.copy()

    for dim in DIMENSIONS:
        values = df_daily_signal[dim].values.copy()
        for i in range(len(values)):
            if not np.isnan(values[i]):
                if abs(values[i]) > threshold:
                    values[i] = 1.0
                else:
                    values[i] = 0.0
        df_daily_signal[dim] = values

    return df_daily_signal


def build_persistence(df_daily_signal, persist=DEFAULT_PERSIST):
    """
    Capa 2 (seccion 5.4): indicador de persistencia.

    Un dia marca 1 solo si los 'persist' dias consecutivos que terminan en el
    tuvieron todos señal diaria activa. Una desviacion aislada de un dia no
    basta para que el sistema la considere sostenida.

    Si algun dia de la ventana es NaN, el resultado es NaN (no se puede saber
    si la racha se sostuvo). Los primeros dias de cada persona, que no tienen
    ventana completa detras, quedan a 0.0 si hay dato y a NaN si no lo hay.

    persist es variable: 5 es el valor del sistema, pero la app permite
    explorar otros valores desde el panel izquierdo.
    """
    df_persistence = df_daily_signal.copy()
    all_pids = df_daily_signal.index.get_level_values("pid").unique()

    for dim in DIMENSIONS:
        persistence_values = []

        for pid in all_pids:
            pid_data = df_daily_signal.loc[pid][dim].values
            n = len(pid_data)
            pid_persistence = []

            for i in range(n):
                if i < persist - 1:
                    # No hay dias suficientes detras para formar la ventana
                    if np.isnan(pid_data[i]):
                        pid_persistence.append(np.nan)
                    else:
                        pid_persistence.append(0.0)
                else:
                    window = pid_data[i - (persist - 1): i + 1]

                    has_nan = False
                    for v in window:
                        if np.isnan(v):
                            has_nan = True
                            break

                    if has_nan:
                        pid_persistence.append(np.nan)
                    else:
                        all_active = True
                        for v in window:
                            if v != 1.0:
                                all_active = False
                                break
                        if all_active:
                            pid_persistence.append(1.0)
                        else:
                            pid_persistence.append(0.0)

            persistence_values.extend(pid_persistence)

        df_persistence[dim] = persistence_values

    return df_persistence


def build_pillars(df_persistence):
    """
    Capa 3 (seccion 5.5): composicion de dominios.

    Agrupa los cuatro indicadores de persistencia en los tres dominios sobre
    los que opera la regla:
      - sueno: solo sleep
      - activacion: steps OR location (marca si se sostuvo en cualquiera)
      - uso pasivo: solo screen

    Si una de las dos señales de activacion falta, se usa la otra; si faltan
    las dos, el dominio queda NaN ese dia.
    """
    sl = df_persistence["sleep"].values
    st = df_persistence["steps"].values
    lo = df_persistence["location"].values
    sc = df_persistence["screen"].values

    act = np.full(len(sl), np.nan)
    for i in range(len(sl)):
        if np.isnan(st[i]) and np.isnan(lo[i]):
            act[i] = np.nan
        elif np.isnan(st[i]):
            act[i] = lo[i]
        elif np.isnan(lo[i]):
            act[i] = st[i]
        else:
            if st[i] == 1.0 or lo[i] == 1.0:
                act[i] = 1.0
            else:
                act[i] = 0.0

    df_pillars = pd.DataFrame(
        {"sleep": sl, "activation": act, "passive_use": sc},
        index=df_persistence.index,
    )

    return df_pillars


def build_convergence(df_pillars):
    """
    Capa 4 (seccion 5.6): convergencia 2 de 3.

    Un dia cuenta como dia-alerta si al menos dos de los tres dominios estan
    activos a la vez. Si ese dia hay menos de dos dominios evaluables, el dia
    no es evaluable para la regla y queda NaN (no es un 0: es que no se sabe).

    Devuelve un DataFrame con una unica columna 'convergence'.
    """
    sl = df_pillars["sleep"].values
    act = df_pillars["activation"].values
    pu = df_pillars["passive_use"].values

    conv = np.full(len(sl), np.nan)
    for i in range(len(conv)):
        values = [sl[i], act[i], pu[i]]

        evaluable = 0
        active = 0
        for v in values:
            if not np.isnan(v):
                evaluable = evaluable + 1
                if v == 1.0:
                    active = active + 1

        if evaluable < 2:
            conv[i] = np.nan
        elif active >= 2:
            conv[i] = 1.0
        else:
            conv[i] = 0.0

    df_convergence = pd.DataFrame({"convergence": conv}, index=df_pillars.index)

    return df_convergence


def run_rule(df_rolling, threshold=DEFAULT_THRESHOLD, persist=DEFAULT_PERSIST):
    """
    Aplica las cuatro capas de la regla de una vez.

    Devuelve un diccionario con los cuatro objetos intermedios:
    'daily_signal', 'persistence', 'pillars' y 'convergence'. La app y los
    demas modulos toman de aqui lo que necesiten (clinical_notice.py necesita
    'persistence'; episodes.py necesita 'convergence').
    """
    df_daily_signal = build_daily_signal(df_rolling, threshold)
    df_persistence = build_persistence(df_daily_signal, persist)
    df_pillars = build_pillars(df_persistence)
    df_convergence = build_convergence(df_pillars)

    return {
        "daily_signal": df_daily_signal,
        "persistence": df_persistence,
        "pillars": df_pillars,
        "convergence": df_convergence,
    }


def count_alert_days(df_convergence):
    """
    Numero de dias-alerta: dias con convergencia activa.
    Sobre los datos reales con 2.0/5 son 186.
    """
    return int((df_convergence["convergence"] == 1.0).sum())


def count_people_with_alerts(df_convergence):
    """
    Numero de personas con al menos un dia-alerta.
    Sobre los datos reales con 2.0/5 son 56.
    """
    people = set()
    for (pid, idx), v in df_convergence["convergence"].items():
        if not np.isnan(v) and v == 1.0:
            people.add(pid)
    return len(people)


def apply_rule(df_rolling, threshold=DEFAULT_THRESHOLD, persist=DEFAULT_PERSIST):
    """
    Atajo que devuelve solo el recuento: (n_dias_alerta, n_episodios).

    Es la forma en que la usa validation.py para comparar datos reales contra
    ruido, y la que alimenta el semaforo señal/ruido de la app.

    Sobre los datos reales con 2.0/5: (186, 63).
    """
    from .episodes import count_episodes

    result = run_rule(df_rolling, threshold, persist)
    df_convergence = result["convergence"]

    n_days = count_alert_days(df_convergence)
    n_episodes = count_episodes(df_convergence)

    return n_days, n_episodes
