import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_metricas_clientes():
    """
    Genera un caso de uso aleatorio para la función
    calcular_metricas_clientes(df)
    """

    # -----------------------------
    # 1. Generar datos aleatorios
    # -----------------------------
    
    n_clientes = random.randint(3, 6)
    n_categorias = random.randint(2, 4)
    n_transacciones = random.randint(15, 30)

    clientes = [f"C{i}" for i in range(1, n_clientes + 1)]
    categorias = [f"CAT{i}" for i in range(1, n_categorias + 1)]

    data = {
        "cliente_id": np.random.choice(clientes, size=n_transacciones),
        "categoria": np.random.choice(categorias, size=n_transacciones),
        "monto": np.round(np.random.uniform(10, 500, size=n_transacciones), 2)
    }

    df = pd.DataFrame(data)

    # -----------------------------
    # 2. Construir INPUT
    # -----------------------------

    input_data = {
        "df": df.copy()
    }

    # -----------------------------
    # 3. Calcular OUTPUT esperado
    # -----------------------------

    # Métricas por cliente
    metricas_cliente = df.groupby("cliente_id").agg(
        total_gastado_cliente=("monto", "sum"),
        promedio_gasto_cliente=("monto", "mean"),
        num_transacciones_cliente=("monto", "count")
    ).reset_index()

    # Métricas por cliente y categoría
    metricas_categoria = df.groupby(["cliente_id", "categoria"]).agg(
        total_categoria=("monto", "sum")
    ).reset_index()

    # Merge
    resultado = metricas_categoria.merge(
        metricas_cliente,
        on="cliente_id",
        how="left"
    )

    # Calcular porcentaje
    resultado["porcentaje_categoria"] = (
        resultado["total_categoria"] /
        resultado["total_gastado_cliente"]
    )

    # Ordenar
    resultado = resultado.sort_values(
        by=["cliente_id", "categoria"]
    ).reset_index(drop=True)

    output_data = resultado

    return input_data, output_data
