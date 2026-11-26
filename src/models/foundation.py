"""
Modelos fundacionales de series temporales en modo zero-shot (sin entrenamiento):
Chronos-2 (Amazon) y TimesFM 3 (Google Research).

Se evalúan bajo el mismo protocolo que el resto del pipeline: pronóstico a 1 paso
adelante (h=1) usando como contexto toda la demanda real observada hasta t-1 y las
mismas variables de calendario conocidas de antemano que usa SARIMAX.

Requieren dependencias opcionales: ver requirements-foundation.txt.
"""

import importlib.util
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import OPTIMAL_QUANTILE
from src.models.statistical import COLUMNAS_EXOGENAS_SARIMAX

CHRONOS2_REPO = "amazon/chronos-2"
TIMESFM3_REPO = "google/timesfm-3.0-pytorch"

# Mismas exógenas deterministas de calendario que SARIMAX (conocidas en t para t)
COLUMNAS_CALENDARIO = COLUMNAS_EXOGENAS_SARIMAX


def modelos_fundacionales_disponibles() -> bool:
    """Indica si están instaladas las dependencias opcionales (torch, chronos, timesfm)."""
    return all(
        importlib.util.find_spec(paquete) is not None
        for paquete in ("torch", "chronos", "timesfm3")
    )


def _dispositivo() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _cargar_chronos2():
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(CHRONOS2_REPO, device_map=_dispositivo())


@lru_cache(maxsize=1)
def _cargar_timesfm3():
    from timesfm3 import TimesFM3Forecaster

    return TimesFM3Forecaster.from_pretrained(
        TIMESFM3_REPO, device=_dispositivo(), per_core_batch_size=32
    )


def _preparar_serie(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    dias_calibracion: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Une train y test en una sola serie cronológica y devuelve los índices de los días
    a pronosticar: primero los `dias_calibracion` finales de train y luego todo test.
    Para cada índice t, el modelo solo recibe la demanda de los días anteriores a t.
    """
    df_total = pd.concat([df_train, df_test], ignore_index=True)
    demanda = df_total["unidades_vendidas"].to_numpy(dtype=np.float32)
    calendario = df_total[COLUMNAS_CALENDARIO].to_numpy(dtype=np.float32).T
    indices = list(range(len(df_train) - dias_calibracion, len(df_total)))
    return demanda, calendario, indices


def _cuantil_chronos2(
    demanda: np.ndarray,
    calendario: np.ndarray,
    indices: list[int],
    cuantil: float,
) -> np.ndarray:
    pipeline = _cargar_chronos2()
    entradas = [
        {
            "target": demanda[:t],
            "past_covariates": {
                nombre: calendario[j, :t] for j, nombre in enumerate(COLUMNAS_CALENDARIO)
            },
            "future_covariates": {
                nombre: calendario[j, t : t + 1] for j, nombre in enumerate(COLUMNAS_CALENDARIO)
            },
        }
        for t in indices
    ]
    cuantiles, _ = pipeline.predict_quantiles(
        entradas, prediction_length=1, quantile_levels=[cuantil], batch_size=64
    )
    return np.array([float(q.reshape(-1)[0]) for q in cuantiles], dtype=float)


def _cuantil_timesfm3(
    demanda: np.ndarray,
    calendario: np.ndarray,
    indices: list[int],
    cuantil: float,
) -> np.ndarray:
    modelo = _cargar_timesfm3()
    niveles = np.asarray(modelo.config.quantiles, dtype=float)
    idx_cuantil = int(np.argmin(np.abs(niveles - cuantil)))
    if not np.isclose(niveles[idx_cuantil], cuantil):
        raise ValueError(
            f"TimesFM 3 solo emite los cuantiles {niveles.tolist()}; no incluye {cuantil}."
        )

    salidas = modelo.predict_batch(
        contexts=[demanda[:t] for t in indices],
        horizon=1,
        # Covariables conocidas a futuro: historia + el valor del día a pronosticar
        past_future_covariates=[calendario[:, : t + 1] for t in indices],
        return_quantiles=True,
    )
    return np.array(
        [float(np.asarray(s.quantiles).reshape(-1, len(niveles))[0, idx_cuantil]) for s in salidas],
        dtype=float,
    )


def _separar_y_calibrar(
    pred_total: np.ndarray,
    df_train: pd.DataFrame,
    dias_calibracion: int,
    cuantil: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separa las predicciones de calibración y de test, y aplica el mismo ajuste
    conforme (CQR) que LightGBM: suma el cuantil `cuantil` de los residuos
    observados en los últimos `dias_calibracion` días de train.
    """
    pred_cal = pred_total[:dias_calibracion]
    pred_test = np.maximum(0.0, pred_total[dias_calibracion:])

    y_cal = df_train["unidades_vendidas"].to_numpy(dtype=float)[-dias_calibracion:]
    ajuste_conforme = float(np.quantile(y_cal - pred_cal, cuantil))
    pred_test_cqr = np.maximum(0.0, pred_test + ajuste_conforme)
    return pred_test, pred_test_cqr


def predecir_modelos_fundacionales(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cuantil: float = OPTIMAL_QUANTILE,
    dias_calibracion: int = 28,
) -> dict[str, np.ndarray]:
    """
    Pronostica el cuantil óptimo de inventario con Chronos-2 y TimesFM 3 en zero-shot,
    crudo y con calibración conforme temporal (CQR) sobre los últimos días de train.
    """
    demanda, calendario, indices = _preparar_serie(df_train, df_test, dias_calibracion)

    predicciones = {}
    for nombre, funcion in (("chronos2", _cuantil_chronos2), ("timesfm3", _cuantil_timesfm3)):
        pred_total = funcion(demanda, calendario, indices, cuantil)
        crudo, calibrado = _separar_y_calibrar(pred_total, df_train, dias_calibracion, cuantil)
        predicciones[f"{nombre}_cuantil80"] = crudo
        predicciones[f"{nombre}_cuantil80_cqr"] = calibrado
    return predicciones


if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR, TARGET_CATEGORY
    from src.evaluation import evaluar_predicciones
    from src.validation import generar_cortes_walk_forward

    ruta = PROCESSED_DATA_DIR / f"dataset_features_{TARGET_CATEGORY}.csv"
    df = pd.read_csv(ruta, parse_dates=["fecha_compra"])

    _, train_c1, test_c1 = next(generar_cortes_walk_forward(df, n_cortes=4, horizonte_dias=28))
    y_real_c1 = test_c1["unidades_vendidas"].to_numpy()

    preds = predecir_modelos_fundacionales(train_c1, test_c1)
    resultados = [evaluar_predicciones(y_real_c1, y_hat, nombre) for nombre, y_hat in preds.items()]

    print(f"Prueba rápida en Corte 1 (28 días) - dispositivo: {_dispositivo()}")
    print(pd.DataFrame(resultados).to_string(index=False))
