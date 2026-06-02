import pandas as pd

def normalizar_por_categoria(df):

    df_resultado = df.copy()

    max_por_categoria = (
        df_resultado
        .groupby("categoria")["precio"]
        .transform("max")
    )

    df_resultado["precio_relativo"] = (
        df_resultado["precio"] / max_por_categoria
    )

    return df_resultado
