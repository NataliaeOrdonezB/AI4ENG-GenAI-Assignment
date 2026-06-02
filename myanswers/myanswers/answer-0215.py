import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

def analizar_defectos(df, test_size, pca_components):

    df_clean = df.dropna(subset=["defect_type"]).copy()

    df_clean["pressure"] = df_clean["pressure"].apply(
        lambda x: max(x, 0)
    )

    normalizer = Normalizer(norm="l2")

    X_scaled = normalizer.fit_transform(
        df_clean[["vibration", "temperature", "pressure"]]
    )

    y = df_clean["defect_type"].to_numpy()

    pca = PCA(n_components=pca_components)

    X_pca = pca.fit_transform(X_scaled)

    X_pca_train, X_pca_test, y_train, y_test = train_test_split(
        X_pca,
        y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )

    pca_model = f"PCA(n_components={pca_components}) (simulado)"

    return (
        X_pca_train,
        X_pca_test,
        y_train,
        y_test,
        pca_model
    )
