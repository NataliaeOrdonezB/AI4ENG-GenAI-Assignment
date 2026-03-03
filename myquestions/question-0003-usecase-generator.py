import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random

def generar_caso_de_uso_evaluar_clasificador_riesgo():
    """
    Genera datos sintéticos y calcula el resultado esperado para evaluar_clasificador_riesgo.
    """
    # 1. Generar datos aleatorios de juguete (X e y)
    n_samples = random.randint(120, 180)
    n_features = 5
    X = np.random.rand(n_samples, n_features)
    # Crear un y que dependa algo de X para que el modelo no sea totalmente al azar
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    
    # 2. Lógica interna (lo que debería hacer la función del compañero)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Matriz de confusión: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    
    # Aseguramos que la matriz sea 2x2 para extraer los datos
    if cm.shape == (2, 2):
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
    else:
        # Caso borde si solo predice una clase
        fp = 0
        fn = 0
        
    metriz_dict = {
        'falsos_positivos': fp,
        'falsos_negativos': fn
    }
    
    # El input debe ser un diccionario con los argumentos de la función
    input_data = {'X': X, 'y': y}
    # El output es la tupla (modelo, diccionario)
    output_data = (model, metriz_dict)
    
    return input_data, output_data

if __name__ == "__main__":
    inp, out = generar_caso_de_uso_evaluar_clasificador_riesgo()
    print("Caso 0003 (Sklearn) generado.")
    print(f"Diccionario de métricas esperado: {out[1]}")
