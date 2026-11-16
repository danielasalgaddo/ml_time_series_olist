from pathlib import Path
import numpy as np
import pandas as pd
from src.config import PROCESSED_DATA_DIR, TARGET_CATEGORY


def agregar_variables_calendario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables deterministas de calendario, codificación cíclica
    y banderas de eventos comerciales extremos (Black Friday y Feriados clave).
    """
    df_feat = df.copy()
    fecha = pd.to_datetime(df_feat["fecha_compra"])

    # Componentes básicos de fecha
    dia_semana = fecha.dt.day_of_week        # 0 = Lunes, 6 = Domingo         # 0 = Lunes, 6 = Domingo
    dia_mes = fecha.dt.day                   # 1 a 31
    mes = fecha.dt.month                     # 1 a 12

    df_feat["es_fin_de_semana"] = (dia_semana >= 5).astype(int)
    df_feat["es_quincena_o_fin_mes"] = dia_mes.isin([1, 2, 15, 16, 30, 31]).astype(int)

    # Codificación cíclica (Seno / Coseno) para día de la semana y mes
    df_feat["dia_semana_sin"] = np.sin(2 * np.pi * dia_semana / 7.0)
    df_feat["dia_semana_cos"] = np.cos(2 * np.pi * dia_semana / 7.0)
    df_feat["mes_sin"] = np.sin(2 * np.pi * (mes - 1) / 12.0)
    df_feat["mes_cos"] = np.cos(2 * np.pi * (mes - 1) / 12.0)

    # Variable exógena crítica: Black Friday en Brasil (24 nov 2017 y fin de semana posterior)
    fechas_black_friday = pd.to_datetime(["2017-11-24", "2017-11-25", "2017-11-26"])
    df_feat["es_black_friday"] = fecha.isin(fechas_black_friday).astype(int)

    # Feriados nacionales fijos principales en Brasil (Navidad, Año Nuevo, Carnaval aprox, Tiradentes, etc.)
    feriados_brasil = pd.to_datetime([
        "2017-01-01", "2017-02-27", "2017-02-28", "2017-04-14", "2017-04-21",
        "2017-05-01", "2017-09-07", "2017-10-12", "2017-11-02", "2017-11-15", "2017-12-25",
        "2018-01-01", "2018-02-12", "2018-02-13", "2018-03-30", "2018-04-21",
        "2018-05-01", "2018-05-31",
    ])
    df_feat["es_feriado_nacional"] = fecha.isin(feriados_brasil).astype(int)

    return df_feat


def agregar_rezagos_y_ventanas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea rezagos (lags) y estadísticas de ventana móvil desplazadas estrictamente
    con .shift(1) para evitar cualquier fuga de datos (data leakage).
    """
    df_feat = df.copy()
    target = df_feat["unidades_vendidas"]

    # 1. Rezagos autorregresivos directos (1, 2, 3, 7, 14, 21 y 28 días)
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df_feat[f"unidades_lag_{lag}"] = target.shift(lag)

    # 2. Ventanas móviles sobre la serie desplazada 1 día hacia atrás (t-1)
    serie_pasada = target.shift(1)

    for ventana in [7, 14, 28]:
        df_feat[f"unidades_media_movil_{ventana}"] = serie_pasada.rolling(window=ventana).mean()
        df_feat[f"unidades_std_movil_{ventana}"] = serie_pasada.rolling(window=ventana).std()

    # 3. Cuantil móvil empírico reciente (percentil 80 de los últimos 14 y 28 días)
    # Excelente predictor base para Regresión Cuantílica
    df_feat["unidades_cuantil80_movil_14"] = serie_pasada.rolling(window=14).quantile(0.80)
    df_feat["unidades_cuantil80_movil_28"] = serie_pasada.rolling(window=28).quantile(0.80)

    # 4. Variables exógenas de precio y flete (estrictamente en t-1)
    df_feat["precio_promedio_lag_1"] = df_feat["precio_promedio_diario"].shift(1)
    df_feat["precio_media_movil_7"] = df_feat["precio_promedio_diario"].shift(1).rolling(7).mean()

    # Ratio de precio relativo (¿ayer estuvo más barato que el promedio de las últimas 4 semanas?)
    media_precio_28 = df_feat["precio_promedio_diario"].shift(1).rolling(28).mean()
    df_feat["ratio_precio_relativo_28"] = df_feat["precio_promedio_lag_1"] / media_precio_28

    return df_feat


def construir_matriz_features(categoria: str = TARGET_CATEGORY) -> pd.DataFrame:
    """
    Carga la serie limpia, genera todas las features, elimina las filas iniciales
    incompletas por la ventana de 28 días y guarda el dataset listo para modelar.
    """
    ruta_entrada = PROCESSED_DATA_DIR / f"serie_diaria_{categoria}.csv"
    df_serie = pd.read_csv(ruta_entrada, parse_dates=["fecha_compra"])

    df_con_calendario = agregar_variables_calendario(df_serie)
    df_completo = agregar_rezagos_y_ventanas(df_con_calendario)

    # Eliminamos los primeros 28 días que tienen NaN por los lags y limpiamos los 0s del arranque de 2017
    df_modelado = df_completo.dropna().reset_index(drop=True)

    ruta_salida = PROCESSED_DATA_DIR / f"dataset_features_{categoria}.csv"
    df_modelado.to_csv(ruta_salida, index=False)
    return df_modelado


if __name__ == "__main__":
    df_final = construir_matriz_features(TARGET_CATEGORY)
    fila_pico = df_final.loc[df_final["unidades_vendidas"].idxmax()]

    print(f"Matriz de features generada exitosamente.")
    print("-" * 55)
    print(f"Dimensiones finales (filas, columnas): {df_final.shape}")
    print(f"Rango de fechas efectivo: {df_final['fecha_compra'].min().date()} a {df_final['fecha_compra'].max().date()}")
    print(f"Día de máxima demanda: {fila_pico['fecha_compra'].date()} con {fila_pico['unidades_vendidas']} unidades (Black Friday = {fila_pico['es_black_friday']})")
    print(f"Nulos restantes en la matriz: {df_final.isna().sum().sum()}")
    print("-" * 55)