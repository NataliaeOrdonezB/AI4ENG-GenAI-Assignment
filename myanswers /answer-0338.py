import pandas as pd

def generar_ranking_estudiantes(df):

    df_clean = df[
        (df['nota'] >= 0) &
        (df['nota'] <= 5)
    ].copy()

    df_clean = df_clean.sort_values(
        ['grupo', 'nota'],
        ascending=[True, False]
    )

    df_clean['ranking'] = (
        df_clean.groupby('grupo')
        .cumcount() + 1
    )

    return df_clean
