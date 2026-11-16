from typing import Iterator
import pandas as pd

# Columnas que representan el futuro/presente en el día t y NUNCA deben entrar en X
COLUMNAS_EXCLUIDAS = [
    "fecha_compra",
    "unidades_vendidas",
    "precio_promedio_diario",
    "flete_promedio_diario",
    "ingresos_totales_diarios",
]


def obtener_columnas_predictoras(df: pd.DataFrame) -> list[str]:
    """Devuelve la lista limpia de features rezagadas y de calendario (cero data leakage)."""
    return [col for col in df.columns if col not in COLUMNAS_EXCLUIDAS]


def generar_cortes_walk_forward(
    df: pd.DataFrame,
    n_cortes: int = 4,
    horizonte_dias: int = 28,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """
    Implementa validación cruzada temporal con ventana expansiva (Expanding Window).
    Reserva los últimos (n_cortes * horizonte_dias) para evaluar en bloques sucesivos.
    """
    df_ordenado = df.sort_values("fecha_compra").reset_index(drop=True)
    total_filas = len(df_ordenado)
    inicio_evaluacion = total_filas - (n_cortes * horizonte_dias)

    if inicio_evaluacion <= 180:
        raise ValueError(
            "La ventana inicial de entrenamiento es demasiado corta. Reduce n_cortes o horizonte_dias."
        )

    for indice_corte in range(n_cortes):
        fin_train = inicio_evaluacion + (indice_corte * horizonte_dias)
        fin_test = fin_train + horizonte_dias

        df_train = df_ordenado.iloc[:fin_train].copy()
        df_test = df_ordenado.iloc[fin_train:fin_test].copy()

        yield indice_corte + 1, df_train, df_test


if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR, TARGET_CATEGORY

    ruta_features = PROCESSED_DATA_DIR / f"dataset_features_{TARGET_CATEGORY}.csv"
    df_datos = pd.read_csv(ruta_features, parse_dates=["fecha_compra"])
    features = obtener_columnas_predictoras(df_datos)

    print(f"Total de variables predictoras (X): {len(features)}")
    print(f"Lista de features: {features}")
    print("-" * 75)

    for num_corte, train, test in generar_cortes_walk_forward(df_datos, n_cortes=4, horizonte_dias=28):
        print(
            f"Corte {num_corte} | "
            f"Train: {train['fecha_compra'].min().date()} -> {train['fecha_compra'].max().date()} ({len(train)} días) | "
            f"Test: {test['fecha_compra'].min().date()} -> {test['fecha_compra'].max().date()} ({len(test)} días)"
        )