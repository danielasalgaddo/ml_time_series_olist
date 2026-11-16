import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.config import OPTIMAL_QUANTILE

COLUMNAS_EXOGENAS_SARIMAX = [
    "es_fin_de_semana",
    "es_quincena_o_fin_mes",
    "es_black_friday",
    "es_feriado_nacional",
]


def predecir_baseline_empirico(
    df_test: pd.DataFrame,
    columna_base: str = "unidades_cuantil80_movil_14",
) -> np.ndarray:
    """
    Baseline heurístico de inventario: aprovisiona para mañana el cuantil 0.80
    empírico observado en los últimos 14 días (t-14 a t-1).
    """
    return df_test[columna_base].to_numpy(dtype=float)


def entrenar_y_predecir_sarimax(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cuantil: float = OPTIMAL_QUANTILE,
) -> dict[str, np.ndarray]:
    """
    Ajusta un modelo SARIMAX(1,0,1)x(1,0,1,7) sobre train y realiza pronósticos
    a 1 paso adelante (h=1) sobre test actualizando el filtro de Kalman día a día
    sin reestimar parámetros (refit=False), garantizando igualdad de condiciones con ML.

    Devuelve tanto la predicción puntual media como el cuantil paramétrico alpha.
    """
    # Asignar índice temporal diario continuo para que statsmodels valide la secuencia
    idx_train = pd.DatetimeIndex(df_train["fecha_compra"], freq="D")
    idx_test = pd.DatetimeIndex(df_test["fecha_compra"], freq="D")

    y_train = pd.Series(df_train["unidades_vendidas"].to_numpy(dtype=float), index=idx_train)
    exog_train = pd.DataFrame(
        df_train[COLUMNAS_EXOGENAS_SARIMAX].to_numpy(dtype=float),
        index=idx_train,
        columns=COLUMNAS_EXOGENAS_SARIMAX,
    )

    y_test = pd.Series(df_test["unidades_vendidas"].to_numpy(dtype=float), index=idx_test)
    exog_test = pd.DataFrame(
        df_test[COLUMNAS_EXOGENAS_SARIMAX].to_numpy(dtype=float),
        index=idx_test,
        columns=COLUMNAS_EXOGENAS_SARIMAX,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            endog=y_train,
            exog=exog_train,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        resultado_actual = modelo.fit(disp=False, maxiter=150)

    predicciones_media = []
    predicciones_cuantil = []

    # Límite superior 'mean_ci_upper' coincide con el cuantil unilateral `cuantil`
    # cuando alpha_bilateral = 2 * (1 - cuantil).
    alpha_bilateral = 2.0 * (1.0 - cuantil)

    for i in range(len(df_test)):
        exog_paso = exog_test.iloc[[i]]
        pronostico = resultado_actual.get_forecast(steps=1, exog=exog_paso)
        resumen = pronostico.summary_frame(alpha=alpha_bilateral)

        media_t = float(resumen["mean"].iloc[0])
        cuantil_t = float(resumen["mean_ci_upper"].iloc[0])

        predicciones_media.append(max(0.0, media_t))
        predicciones_cuantil.append(max(0.0, cuantil_t))

        # Incorporar la observación real del día i (que extiende el DatetimeIndex sin saltos)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resultado_actual = resultado_actual.append(
                endog=y_test.iloc[[i]],
                exog=exog_paso,
                refit=False,
            )

    return {
        "sarimax_media": np.array(predicciones_media, dtype=float),
        "sarimax_cuantil80": np.array(predicciones_cuantil, dtype=float),
    }