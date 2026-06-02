import pandas as pd
import numpy as np

def limpiar_y_rankear_jugadores(df, top_n):

    df_clean = df.dropna().copy()

    df_clean["kda"] = (
        (df_clean["kills"] + df_clean["assists"])
        / df_clean["deaths"].clip(lower=1)
    )

    resultado = (
        df_clean[["jugador", "kda", "partidas_jugadas"]]
        .sort_values("kda", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return resultado
