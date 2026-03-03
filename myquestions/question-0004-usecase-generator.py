import numpy as np
from sklearn.cluster import KMeans
import random

def generar_caso_de_uso_segmentar_y_calcular_distancias():
    """
    Genera datos aleatorios y calcula las distancias esperadas a los centroides.
    """
    # 1. Generar datos aleatorios (X)
    n_samples = random.randint(40, 60)
    n_features = 3
    n_clusters = random.randint(2, 4)
    X = np.random.rand(n_samples, n_features)
    
    # 2. Lógica de solución esperada
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    
    # Calcular distancias manualmente para el output esperado
    distancias = []
    for i in range(len(X)):
        punto = X[i]
        centroide_asignado = centroids[labels[i]]
        # Distancia euclidiana: raiz de la suma de los cuadrados de las diferencias
        d = np.linalg.norm(punto - centroide_asignado)
        distancias.append(d)
    
    input_data = {
        'X': X,
        'n_clusters': n_clusters
    }
    output_data = (labels, np.array(distancias))
    
    return input_data, output_data

if __name__ == "__main__":
    inp, out = generar_caso_de_uso_segmentar_y_calcular_distancias()
    print("Caso 0004 generado con éxito.")
    print(f"Número de etiquetas generadas: {len(out[0])}")
    print(f"Ejemplo de primera distancia: {out[1][0]}")
