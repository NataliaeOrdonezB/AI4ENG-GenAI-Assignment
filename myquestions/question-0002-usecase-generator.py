import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_analizar_variacion_precios():
    """
    Genera un caso de prueba aleatorio para la función analizar_variacion_precios.
    """
    # 1. Crear datos aleatorios
    n_productos = random.randint(2, 4)
    productos = [f'PROD_{10+i}' for i in range(n_productos)]
    fechas = pd.date_range(start='2026-01-01', periods=10).strftime('%Y-%m-%d').tolist()
    
    data = []
    for p in productos:
        # Cada producto tiene un número de registros al azar entre 4 y 7
        n_regs = random.randint(4, 7)
        fechas_prod = random.sample(fechas, n_regs)
        for f in fechas_prod:
            data.append({
                'fecha': f,
                'producto_id': p,
                'precio': round(random.uniform(50.0, 150.0), 2)
            })
            
    df_input = pd.DataFrame(data)
    # Desordenar los datos inicialmente para probar que la función los ordene bien
    df_input = df_input.sample(frac=1).reset_index(drop=True)
    
    # 2. Calcular la solución esperada (Lógica interna)
    df_sol = df_input.copy()
    df_sol['fecha'] = pd.to_datetime(df_sol['fecha'])
    df_sol = df_sol.sort_values(['producto_id', 'fecha'])
    
    # Calculamos el cambio agrupando por producto
    df_sol['cambio_precio'] = df_sol.groupby('producto_id')['precio'].diff()
    df_sol['es_subida'] = df_sol['cambio_precio'] > 0
    
    # Quitamos nulos y reseteamos índice
    df_sol = df_sol.dropna(subset=['cambio_precio']).reset_index(drop=True)
    
    input_data = {'df': df_input}
    output_data = df_sol
    
    return input_data, output_data

if __name__ == "__main__":
    inp, out = generar_caso_de_uso_analizar_variacion_precios()
    print("Caso 0002 generado con éxito.")
    print(f"Columnas resultantes: {out.columns.tolist()}")
    print(out.head())
