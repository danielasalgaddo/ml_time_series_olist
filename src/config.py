from pathlib import Path

# 1. Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

# Crear directorios de salida si no existen
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Archivos crudos de entrada (Olist)
ORDERS_FILE = RAW_DATA_DIR / "olist_orders_dataset.csv"
ITEMS_FILE = RAW_DATA_DIR / "olist_order_items_dataset.csv"
PRODUCTS_FILE = RAW_DATA_DIR / "olist_products_dataset.csv"

# 3. Parámetros de construcción de la Serie Temporal
# Recorte estricto: evitamos 2016 (datos dispersos) y fines de agosto 2018 (corte artificial)
START_DATE = "2017-01-01"
END_DATE = "2018-08-15"

# Categoría objetivo (cama_mesa_banho es la de mayor volumen y continuidad en Olist)
TARGET_CATEGORY = "cama_mesa_banho"

# Estados de orden considerados como demanda efectiva
VALID_ORDER_STATUS = ["delivered", "shipped", "invoiced", "processing", "approved"]

# 4. Parámetros Económicos (Newsvendor / Pinball Loss)
# Basados en estructura de costos estimada sobre el ticket promedio de la categoría
COST_UNDERSTOCK = 20.0  # c_u: Margen perdido + penalización logística (BRL por unidad)
COST_OVERSTOCK = 5.0    # c_o: Costo de bodega + capital inmovilizado (BRL por unidad)

# Cuantil óptimo teórico: alpha* = c_u / (c_u + c_o) = 20 / 25 = 0.80
OPTIMAL_QUANTILE = COST_UNDERSTOCK / (COST_UNDERSTOCK + COST_OVERSTOCK)