from pathlib import Path
import pandas as pd
from src.config import (
    ORDERS_FILE,
    ITEMS_FILE,
    PRODUCTS_FILE,
    PROCESSED_DATA_DIR,
    START_DATE,
    END_DATE,
    TARGET_CATEGORY,
    VALID_ORDER_STATUS,
)


def cargar_tablas_crudas() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga las tres tablas relacionales de Olist desde data/raw."""
    tabla_ordenes = pd.read_csv(
        ORDERS_FILE,
        usecols=["order_id", "order_status", "order_purchase_timestamp"],
    )
    tabla_items_orden = pd.read_csv(
        ITEMS_FILE,
        usecols=["order_id", "order_item_id", "product_id", "price", "freight_value"],
    )
    tabla_productos = pd.read_csv(
        PRODUCTS_FILE,
        usecols=["product_id", "product_category_name"],
    )
    return tabla_ordenes, tabla_items_orden, tabla_productos


def construir_serie_diaria_categoria(
    categoria_objetivo: str = TARGET_CATEGORY,
) -> pd.DataFrame:
    """
    Cruza las tablas transaccionales, filtra por categoría y estado válido,
    y reindexa a una serie temporal diaria continua sin fechas faltantes.
    """
    tabla_ordenes, tabla_items_orden, tabla_productos = cargar_tablas_crudas()

    # 1. Filtrar órdenes con estado de demanda válida
    ordenes_validas = tabla_ordenes[
        tabla_ordenes["order_status"].isin(VALID_ORDER_STATUS)
    ].copy()

    # 2. Convertir timestamp a fecha diaria (piso del día)
    ordenes_validas["fecha_compra"] = pd.to_datetime(
        ordenes_validas["order_purchase_timestamp"]
    ).dt.floor("D")

    # 3. Unir ítems con productos para conocer la categoría de cada ítem vendido
    items_con_categoria = pd.merge(
        tabla_items_orden,
        tabla_productos,
        on="product_id",
        how="inner",
    )

    # 4. Filtrar únicamente la categoría objetivo
    items_categoria = items_con_categoria[
        items_con_categoria["product_category_name"] == categoria_objetivo
    ].copy()

    # 5. Unir con las órdenes válidas para obtener la fecha de cada venta
    transacciones_completas = pd.merge(
        items_categoria,
        ordenes_validas[["order_id", "fecha_compra"]],
        on="order_id",
        how="inner",
    )

    # 6. Agregación diaria: cada fila en order_items representa 1 unidad física vendida
    serie_agrupada = (
        transacciones_completas.groupby("fecha_compra")
        .agg(
            unidades_vendidas=("order_item_id", "count"),
            precio_promedio_diario=("price", "mean"),
            flete_promedio_diario=("freight_value", "mean"),
            ingresos_totales_diarios=("price", "sum"),
        )
        .reset_index()
    )

    # 7. Reindexación a calendario diario continuo entre START_DATE y END_DATE
    calendario_completo = pd.date_range(
        start=START_DATE, end=END_DATE, freq="D", name="fecha_compra"
    )
    serie_diaria = (
        serie_agrupada.set_index("fecha_compra")
        .reindex(calendario_completo)
        .reset_index()
    )

    # 8. Tratamiento riguroso de nulos en días sin transacciones
    # Cantidades e ingresos en días sin compras son estrictamente 0
    serie_diaria["unidades_vendidas"] = (
        serie_diaria["unidades_vendidas"].fillna(0).astype(int)
    )
    serie_diaria["ingresos_totales_diarios"] = serie_diaria[
        "ingresos_totales_diarios"
    ].fillna(0.0)

    # Precios y fletes se mantienen con el último valor observado (evita falsos precios 0)
    serie_diaria["precio_promedio_diario"] = (
        serie_diaria["precio_promedio_diario"].ffill().bfill()
    )
    serie_diaria["flete_promedio_diario"] = (
        serie_diaria["flete_promedio_diario"].ffill().bfill()
    )

    return serie_diaria


def guardar_serie_procesada(df_serie: pd.DataFrame, categoria: str = TARGET_CATEGORY) -> Path:
    """Guarda la serie temporal limpia en data/processed/."""
    ruta_salida = PROCESSED_DATA_DIR / f"serie_diaria_{categoria}.csv"
    df_serie.to_csv(ruta_salida, index=False)
    return ruta_salida


if __name__ == "__main__":
    print(f"Construyendo serie temporal para la categoría: {TARGET_CATEGORY}...")
    df_resultado = construir_serie_diaria_categoria(TARGET_CATEGORY)
    ruta = guardar_serie_procesada(df_resultado, TARGET_CATEGORY)

    print(f"Archivo guardado exitosamente en: {ruta}")
    print("-" * 50)
    print(f"Total de días en la serie: {len(df_resultado)}")
    print(f"Días con 0 ventas: {(df_resultado['unidades_vendidas'] == 0).sum()}")
    print(f"Promedio de unidades diarias: {df_resultado['unidades_vendidas'].mean():.2f}")
    print(f"Máximo de unidades en un día: {df_resultado['unidades_vendidas'].max()}")
    print(f"Precio promedio de la categoría (BRL): {df_resultado['precio_promedio_diario'].mean():.2f}")
    print("-" * 50)
    print(df_resultado.head())