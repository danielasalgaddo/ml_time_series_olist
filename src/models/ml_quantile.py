import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from src.config import OPTIMAL_QUANTILE
from src.validation import obtener_columnas_predictoras


def _crear_estimador_lgbm(objetivo: str, alpha: float | None = None) -> LGBMRegressor:
    """
    Configura un regresor LightGBM optimizado para series temporales cortas.
    """
    parametros = {
        "objective": objetivo,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 12,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbose": -1,
    }
    if objetivo == "quantile" and alpha is not None:
        parametros["alpha"] = alpha

    return LGBMRegressor(**parametros)


def _calcular_pesos_temporales(n_muestras: int, decaimiento: float = 0.004) -> np.ndarray:
    """
    Genera pesos exponenciales crecientes para que el modelo priorice el régimen
    reciente de demanda frente a observaciones antiguas de 2017.
    """
    indices = np.arange(n_muestras, dtype=float)
    pesos = np.exp(decaimiento * (indices - n_muestras))
    return pesos / np.mean(pesos)


def entrenar_y_predecir_lightgbm(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cuantil_optimo: float = OPTIMAL_QUANTILE,
    dias_calibracion: int = 28,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """
    Entrena las variantes de LightGBM descomponiendo el nivel local
    (y_t = base_movil_14 + residuo_t) para hacer estacionario el objetivo
    y evitar el sesgo de subestimación de los árboles ante tendencias.
    """
    columnas_x = obtener_columnas_predictoras(df_train)
    X_train = df_train[columnas_x].reset_index(drop=True)
    X_test = df_test[columnas_x].reset_index(drop=True)

    # Nivel local conocido en t-1 (elimina la no-estacionariedad de nivel entre 2017 y 2018)
    base_train = df_train["unidades_media_movil_14"].astype(float).reset_index(drop=True)
    base_test = df_test["unidades_media_movil_14"].astype(float).reset_index(drop=True)

    y_train_abs = df_train["unidades_vendidas"].astype(float).reset_index(drop=True)
    # Target estacionario: cuánto sube o baja la demanda respecto a las últimas 2 semanas
    y_train_res = y_train_abs - base_train

    pesos_totales = _calcular_pesos_temporales(len(df_train))

    # 1. Modelo Simétrico L2 (Media sobre nivel local)
    modelo_media = _crear_estimador_lgbm(objetivo="regression")
    modelo_media.fit(X_train, y_train_res, sample_weight=pesos_totales)
    pred_media = np.maximum(0.0, base_test + modelo_media.predict(X_test))

    # 2. Modelo Simétrico L1 (Mediana, q = 0.50 sobre nivel local)
    modelo_mediana = _crear_estimador_lgbm(objetivo="quantile", alpha=0.50)
    modelo_mediana.fit(X_train, y_train_res, sample_weight=pesos_totales)
    pred_mediana = np.maximum(0.0, base_test + modelo_mediana.predict(X_test))

    # 3. Modelo Asimétrico Newsvendor (Cuantil óptimo q = 0.80 sobre nivel local)
    modelo_q80 = _crear_estimador_lgbm(objetivo="quantile", alpha=cuantil_optimo)
    modelo_q80.fit(X_train, y_train_res, sample_weight=pesos_totales)
    pred_q80 = np.maximum(0.0, base_test + modelo_q80.predict(X_test))

    # 4. Modelo LightGBM Q80 + Calibración Conforme Temporal (CQR)
    corte_cal = len(df_train) - dias_calibracion
    X_subtrain = X_train.iloc[:corte_cal]
    y_subtrain_res = y_train_res.iloc[:corte_cal]
    X_cal = X_train.iloc[corte_cal:]
    y_cal_abs = y_train_abs.iloc[corte_cal:].to_numpy()
    base_cal = base_train.iloc[corte_cal:].to_numpy()

    pesos_subtrain = _calcular_pesos_temporales(len(X_subtrain))
    modelo_cal = _crear_estimador_lgbm(objetivo="quantile", alpha=cuantil_optimo)
    modelo_cal.fit(X_subtrain, y_subtrain_res, sample_weight=pesos_subtrain)
    pred_cal_abs = base_cal + modelo_cal.predict(X_cal)

    residuos_calibracion = y_cal_abs - pred_cal_abs
    ajuste_conforme = float(np.quantile(residuos_calibracion, cuantil_optimo))
    pred_q80_cqr = np.maximum(0.0, pred_q80 + ajuste_conforme)

    df_importancia = pd.DataFrame({
        "variable": columnas_x,
        "importancia_splits": modelo_q80.feature_importances_,
    }).sort_values("importancia_splits", ascending=False).reset_index(drop=True)

    predicciones = {
        "lgbm_media_l2": pred_media.to_numpy(dtype=float),
        "lgbm_mediana_q50": pred_mediana.to_numpy(dtype=float),
        "lgbm_cuantil80": pred_q80.to_numpy(dtype=float),
        "lgbm_cuantil80_cqr": pred_q80_cqr.to_numpy(dtype=float),
    }
    return predicciones, df_importancia


if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR, TARGET_CATEGORY
    from src.validation import generar_cortes_walk_forward
    from src.evaluation import evaluar_predicciones
    from src.models.statistical import predecir_baseline_empirico, entrenar_y_predecir_sarimax

    ruta = PROCESSED_DATA_DIR / f"dataset_features_{TARGET_CATEGORY}.csv"
    df = pd.read_csv(ruta, parse_dates=["fecha_compra"])

    _, train_c1, test_c1 = next(generar_cortes_walk_forward(df, n_cortes=4, horizonte_dias=28))
    y_real_c1 = test_c1["unidades_vendidas"].to_numpy()

    pred_base = predecir_baseline_empirico(test_c1)
    preds_sarimax = entrenar_y_predecir_sarimax(train_c1, test_c1)
    preds_lgbm, df_imp = entrenar_y_predecir_lightgbm(train_c1, test_c1)

    resultados = [
        evaluar_predicciones(y_real_c1, pred_base, "1. Baseline Empírico (Rolling Q80)"),
        evaluar_predicciones(y_real_c1, preds_sarimax["sarimax_media"], "2. SARIMAX (Media Simétrica)"),
        evaluar_predicciones(y_real_c1, preds_sarimax["sarimax_cuantil80"], "3. SARIMAX (Cuantil 0.80 Paramétrico)"),
        evaluar_predicciones(y_real_c1, preds_lgbm["lgbm_media_l2"], "4. LightGBM (Media L2)"),
        evaluar_predicciones(y_real_c1, preds_lgbm["lgbm_mediana_q50"], "5. LightGBM (Mediana Q50)"),
        evaluar_predicciones(y_real_c1, preds_lgbm["lgbm_cuantil80"], "6. LightGBM (Pinball Loss Q80)"),
        evaluar_predicciones(y_real_c1, preds_lgbm["lgbm_cuantil80_cqr"], "7. LightGBM (Pinball Q80 + CQR)"),
    ]

    print("Prueba rápida en Corte 1 (28 días) con Ponderación Temporal y CQR:")
    print(pd.DataFrame(resultados).to_string(index=False))