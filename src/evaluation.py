import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.config import COST_UNDERSTOCK, COST_OVERSTOCK, OPTIMAL_QUANTILE


def calcular_pinball_loss(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    cuantil: float = OPTIMAL_QUANTILE,
) -> float:
    """
    Calcula la pérdida cuantílica promedio (Pinball Loss):
    L_alpha(y, y_hat) = max(alpha * (y - y_hat), (alpha - 1) * (y - y_hat))
    """
    error = y_real - y_pred
    perdida_diaria = np.maximum(cuantil * error, (cuantil - 1.0) * error)
    return float(np.mean(perdida_diaria))


def calcular_costo_economico_inventario(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    costo_escasez: float = COST_UNDERSTOCK,
    costo_exceso: float = COST_OVERSTOCK,
) -> dict[str, float]:
    """
    Evalúa el impacto financiero real en BRL bajo la función de pérdida asimétrica.
    Redondea el stock aprovisionado a unidades físicas enteras no negativas.
    """
    stock_aprovisionado = np.maximum(0.0, np.round(y_pred))
    unidades_faltantes = np.maximum(0.0, y_real - stock_aprovisionado)
    unidades_sobrantes = np.maximum(0.0, stock_aprovisionado - y_real)

    costo_total_escasez = float(np.sum(unidades_faltantes * costo_escasez))
    costo_total_exceso = float(np.sum(unidades_sobrantes * costo_exceso))
    costo_total_brl = costo_total_escasez + costo_total_exceso

    # Nivel de servicio (Fill Rate diario): proporción de días sin quiebre de stock (y_pred >= y_real)
    tasa_cobertura_dias = float(np.mean(stock_aprovisionado >= y_real) * 100.0)

    return {
        "costo_total_brl": round(costo_total_brl, 2),
        "costo_diario_promedio_brl": round(costo_total_brl / len(y_real), 2),
        "costo_escasez_brl": round(costo_total_escasez, 2),
        "costo_exceso_brl": round(costo_total_exceso, 2),
        "unidades_faltantes_totales": int(np.sum(unidades_faltantes)),
        "unidades_sobrantes_totales": int(np.sum(unidades_sobrantes)),
        "cobertura_dias_pct": round(tasa_cobertura_dias, 2),
    }


def evaluar_predicciones(
    y_real: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    nombre_modelo: str,
) -> dict[str, float | str]:
    """
    Consolida métricas estadísticas simétricas (RMSE, MAE) y métricas asimétricas
    (Pinball Loss y Costo Económico en BRL) para comparar ambos paradigmas.
    """
    arr_real = np.asarray(y_real, dtype=float)
    arr_pred = np.maximum(0.0, np.asarray(y_pred, dtype=float))

    rmse = float(root_mean_squared_error(arr_real, arr_pred))
    mae = float(mean_absolute_error(arr_real, arr_pred))
    pinball = calcular_pinball_loss(arr_real, arr_pred, OPTIMAL_QUANTILE)
    costos = calcular_costo_economico_inventario(arr_real, arr_pred)

    return {
        "modelo": nombre_modelo,
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "Pinball_Loss_q80": round(pinball, 3),
        **costos,
    }


if __name__ == "__main__":
    # Simulación de prueba para verificar el teorema del cuantil crítico (Supuesto 1)
    rng = np.random.default_rng(42)
    demanda_simulada = rng.poisson(lam=20, size=100)

    # Predicción A: Media exacta (20 unidades) -> Optimiza RMSE
    pred_media = np.full_like(demanda_simulada, fill_value=20.0, dtype=float)

    # Predicción B: Cuantil 0.80 de una Poisson(20) (~24 unidades) -> Optimiza Costo Asimétrico
    pred_cuantil80 = np.full_like(demanda_simulada, fill_value=24.0, dtype=float)

    df_comparacion = pd.DataFrame([
        evaluar_predicciones(demanda_simulada, pred_media, "Predicción Media (Simétrica)"),
        evaluar_predicciones(demanda_simulada, pred_cuantil80, "Predicción Cuantil 0.80 (Asimétrica)"),
    ])
    print("Demostración empírica del Marco Teórico (Simulación Poisson):")
    print(df_comparacion.to_string(index=False))