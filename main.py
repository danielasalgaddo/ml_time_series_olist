import subprocess
import sys
from pathlib import Path


def _relanzar_en_entorno_virtual() -> None:
    """
    Si existe un entorno virtual .venv en el proyecto y main.py se ejecutó con otro
    intérprete (por ejemplo el Python global), relanza el script con el de .venv para
    usar las versiones fijadas y las dependencias de los modelos fundacionales.
    """
    dir_venv = Path(__file__).resolve().parent / ".venv"
    candidatos = [dir_venv / "Scripts" / "python.exe", dir_venv / "bin" / "python"]
    python_venv = next((p for p in candidatos if p.exists()), None)

    if python_venv is None or Path(sys.prefix).resolve() == dir_venv:
        return

    print(f"[Aviso] Relanzando con el entorno virtual del proyecto: {python_venv}", flush=True)
    sys.exit(subprocess.run([str(python_venv), *sys.argv]).returncode)


if __name__ == "__main__":
    _relanzar_en_entorno_virtual()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    COST_OVERSTOCK,
    COST_UNDERSTOCK,
    FIGURES_DIR,
    METRICS_DIR,
    OPTIMAL_QUANTILE,
    TARGET_CATEGORY,
)
from src.data_builder import construir_serie_diaria_categoria, guardar_serie_procesada
from src.evaluation import evaluar_predicciones
from src.features import construir_matriz_features
from src.models.foundation import modelos_fundacionales_disponibles, predecir_modelos_fundacionales
from src.models.ml_quantile import entrenar_y_predecir_lightgbm
from src.models.statistical import entrenar_y_predecir_sarimax, predecir_baseline_empirico
from src.validation import generar_cortes_walk_forward


def ejecutar_backtesting_walk_forward(
    df_features: pd.DataFrame,
    n_cortes: int = 4,
    horizonte_dias: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el protocolo Walk-Forward Expanding Window sobre los 4 cortes (112 días).
    Devuelve:
      1. df_predicciones_diarias: todas las predicciones día a día de los 112 días.
      2. df_metricas_por_corte: desempeño desglosado en cada ventana de 28 días.
      3. df_metricas_globales: evaluación consolidada sobre los 112 días completos.
    """
    registros_diarios = []
    metricas_por_corte = []
    importancias_acumuladas = []

    usar_fundacionales = modelos_fundacionales_disponibles()
    if not usar_fundacionales:
        print(
            "  [Aviso] Chronos-2 y TimesFM 3 omitidos: faltan dependencias opcionales "
            "(pip install -r requirements-foundation.txt)."
        )

    for num_corte, df_train, df_test in generar_cortes_walk_forward(
        df_features, n_cortes=n_cortes, horizonte_dias=horizonte_dias
    ):
        print(
            f"  -> Ejecutando Corte {num_corte}/{n_cortes} "
            f"(Test: {df_test['fecha_compra'].min().date()} a {df_test['fecha_compra'].max().date()})..."
        )
        y_real = df_test["unidades_vendidas"].to_numpy(dtype=float)

        # 1. Modelos Estadísticos y Baseline
        pred_base = predecir_baseline_empirico(df_test)
        preds_sarimax = entrenar_y_predecir_sarimax(df_train, df_test)

        # 2. Modelos Machine Learning (LightGBM L2, Q50, Q80 y Q80+CQR)
        preds_lgbm, df_imp = entrenar_y_predecir_lightgbm(df_train, df_test)
        df_imp["corte"] = num_corte
        importancias_acumuladas.append(df_imp)

        diccionario_modelos = {
            "1. Baseline Empírico (Rolling Q80)": pred_base,
            "2. SARIMAX (Media Simétrica)": preds_sarimax["sarimax_media"],
            "3. SARIMAX (Cuantil 0.80 Paramétrico)": preds_sarimax["sarimax_cuantil80"],
            "4. LightGBM (Media L2)": preds_lgbm["lgbm_media_l2"],
            "5. LightGBM (Mediana Q50)": preds_lgbm["lgbm_mediana_q50"],
            "6. LightGBM (Pinball Loss Q80)": preds_lgbm["lgbm_cuantil80"],
            "7. LightGBM (Pinball Q80 + CQR)": preds_lgbm["lgbm_cuantil80_cqr"],
        }

        # 3. Modelos Fundacionales Zero-shot (Chronos-2 y TimesFM 3), crudos y con CQR
        if usar_fundacionales:
            preds_fund = predecir_modelos_fundacionales(df_train, df_test)
            diccionario_modelos.update({
                "8. Chronos-2 (Zero-shot Q80)": preds_fund["chronos2_cuantil80"],
                "9. Chronos-2 (Zero-shot Q80 + CQR)": preds_fund["chronos2_cuantil80_cqr"],
                "10. TimesFM 3 (Zero-shot Q80)": preds_fund["timesfm3_cuantil80"],
                "11. TimesFM 3 (Zero-shot Q80 + CQR)": preds_fund["timesfm3_cuantil80_cqr"],
            })

        # Evaluar métricas del corte actual
        for nombre_mod, y_hat in diccionario_modelos.items():
            res = evaluar_predicciones(y_real, y_hat, nombre_mod)
            res["corte"] = num_corte
            metricas_por_corte.append(res)

        # Guardar predicciones diarias para el consolidado y los gráficos
        df_corte_diario = pd.DataFrame({
            "corte": num_corte,
            "fecha_compra": df_test["fecha_compra"].to_numpy(),
            "y_real": y_real,
            **diccionario_modelos,
        })
        registros_diarios.append(df_corte_diario)

    df_predicciones_diarias = pd.concat(registros_diarios, ignore_index=True)
    df_metricas_por_corte = pd.DataFrame(metricas_por_corte)

    # Evaluación consolidada sobre los 112 días combinados
    nombres_modelos = [col for col in df_predicciones_diarias.columns if col not in ["corte", "fecha_compra", "y_real"]]
    metricas_globales = []
    y_real_global = df_predicciones_diarias["y_real"].to_numpy()

    for nombre_mod in nombres_modelos:
        y_hat_global = df_predicciones_diarias[nombre_mod].to_numpy()
        metricas_globales.append(evaluar_predicciones(y_real_global, y_hat_global, nombre_mod))

    df_metricas_globales = pd.DataFrame(metricas_globales).sort_values("costo_total_brl").reset_index(drop=True)
    return df_predicciones_diarias, df_metricas_por_corte, df_metricas_globales


def generar_graficos_informe(
    df_predicciones: pd.DataFrame,
    df_global: pd.DataFrame,
) -> None:
    """Genera y guarda los 3 gráficos principales del proyecto en outputs/figures/."""
    sns.set_theme(style="whitegrid")

    # GRÁFICO 1: Serie Temporal Real vs Predicciones Simétricas y Asimétricas (112 días)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fechas = pd.to_datetime(df_predicciones["fecha_compra"])

    ax.plot(fechas, df_predicciones["y_real"], color="#1f2937", linewidth=1.8, label="Demanda Real (Unidades)", alpha=0.85)
    ax.plot(
        fechas,
        df_predicciones["5. LightGBM (Mediana Q50)"],
        color="#2563eb",
        linestyle="--",
        linewidth=1.6,
        label="LightGBM Mediana Q50 (Simétrico)",
    )
    ax.plot(
        fechas,
        df_predicciones["7. LightGBM (Pinball Q80 + CQR)"],
        color="#dc2626",
        linewidth=2.0,
        label="LightGBM Pinball Q80 + CQR (Stock Óptimo)",
    )
    ax.plot(
        fechas,
        df_predicciones["3. SARIMAX (Cuantil 0.80 Paramétrico)"],
        color="#059669",
        linestyle="-.",
        linewidth=1.6,
        label="SARIMAX Cuantil 0.80 (Paramétrico)",
    )

    ax.fill_between(
        fechas,
        df_predicciones["5. LightGBM (Mediana Q50)"],
        df_predicciones["7. LightGBM (Pinball Q80 + CQR)"],
        color="#fca5a5",
        alpha=0.30,
        label="Colchón de Protección Asimétrica (Q80 vs Q50)",
    )

    ax.set_title(
        f"Backtesting Walk-Forward (112 días): Demanda Real vs. Stock Aprovisionado ({TARGET_CATEGORY})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Fecha (Corte 1 a Corte 4 - 2018)")
    ax.set_ylabel("Unidades Diarias")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_serie_predicciones_q80.png", dpi=300)
    plt.close(fig)

    # GRÁFICO 2: El Espejismo del RMSE vs Costo Económico Total (BRL)
    fig, ax = plt.subplots(figsize=(10, 6))
    desplazamientos_etiquetas = {
        "8": ((-10, 2), "right"),
        "9": ((-10, 12), "right"),
        "10": ((10, -14), "left"),
        "11": ((10, -3), "left"),
    }
    for _, fila in df_global.iterrows():
        es_q80 = "Q80" in fila["modelo"] or "0.80" in fila["modelo"]
        color = "#dc2626" if es_q80 else "#2563eb"
        marcador = "D" if es_q80 else "o"
        ax.scatter(fila["RMSE"], fila["costo_total_brl"], color=color, s=130, marker=marcador, zorder=4)
        # Los modelos fundacionales quedan agrupados: se separan sus etiquetas a mano
        desplazamiento, alineacion = desplazamientos_etiquetas.get(
            fila["modelo"].split(".")[0], ((8, 5), "left")
        )
        ax.annotate(
            fila["modelo"],
            (fila["RMSE"], fila["costo_total_brl"]),
            xytext=desplazamiento,
            textcoords="offset points",
            ha=alineacion,
            va="center",
            fontsize=9.5,
            fontweight="bold" if es_q80 else "normal",
        )

    ax.set_title(
        f"Trade-off Metodológico: Error Estadístico Simétrico (RMSE) vs. Costo Real de Inventario (BRL)\n"
        f"(c_u = {COST_UNDERSTOCK} BRL, c_o = {COST_OVERSTOCK} BRL -> Cuantil Óptimo = {OPTIMAL_QUANTILE:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("RMSE (Menor es mejor estadísticamente para la media)")
    ax.set_ylabel("Costo Económico Total en 112 días (BRL - Menor es mejor para el negocio)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_tradeoff_rmse_vs_costo.png", dpi=300)
    plt.close(fig)

    # GRÁFICO 3: Desglose de Costos en BRL (Escasez vs Exceso)
    fig, ax = plt.subplots(figsize=(11, 6))
    df_ord = df_global.sort_values("costo_total_brl", ascending=True)
    y_pos = np.arange(len(df_ord))

    ax.barh(
        y_pos,
        df_ord["costo_escasez_brl"],
        color="#ef4444",
        label=f"Costo por Quiebre de Stock (c_u = {COST_UNDERSTOCK} BRL/u)",
    )
    ax.barh(
        y_pos,
        df_ord["costo_exceso_brl"],
        left=df_ord["costo_escasez_brl"],
        color="#3b82f6",
        label=f"Costo por Sobrestock (c_o = {COST_OVERSTOCK} BRL/u)",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_ord["modelo"], fontsize=10)
    ax.set_xlabel("Costo Acumulado en 112 días (Reales Brasileños - BRL)")
    ax.set_title("Descomposición del Costo Económico por Modelo (Quiebre de Stock vs. Almacenamiento)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=True)

    for i, total in enumerate(df_ord["costo_total_brl"]):
        ax.text(total + 100, i, f"{total:,.0f} BRL", va="center", fontsize=9.5, fontweight="bold")

    ax.set_xlim(0, df_ord["costo_total_brl"].max() * 1.22)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_desglose_costos_brl.png", dpi=300)
    plt.close(fig)


def main() -> None:
    print("=" * 80)
    print(f"PIPELINE DE FORECASTING CON PÉRDIDA ASIMÉTRICA - CATEGORÍA: {TARGET_CATEGORY}")
    print("=" * 80)

    print("\n[1/4] Construyendo serie temporal diaria desde tablas relacionales de Olist...")
    df_serie = construir_serie_diaria_categoria(TARGET_CATEGORY)
    guardar_serie_procesada(df_serie, TARGET_CATEGORY)

    print("[2/4] Generando matriz de características temporales y exógenas (cero data leakage)...")
    df_features = construir_matriz_features(TARGET_CATEGORY)

    print("[3/4] Ejecutando Backtesting Walk-Forward (4 cortes x 28 días = 112 días)...")
    df_pred_diarias, df_por_corte, df_global = ejecutar_backtesting_walk_forward(
        df_features, n_cortes=4, horizonte_dias=28
    )

    # Exportar resultados a CSV en outputs/metrics/
    df_pred_diarias.to_csv(METRICS_DIR / "predicciones_diarias_112dias.csv", index=False)
    df_por_corte.to_csv(METRICS_DIR / "metricas_por_corte.csv", index=False)
    df_global.to_csv(METRICS_DIR / "metricas_consolidadas_112dias.csv", index=False)

    print("[4/4] Generando gráficos de alta resolución en outputs/figures/...")
    generar_graficos_informe(df_pred_diarias, df_global)

    print("\n" + "=" * 80)
    print("RESULTADOS CONSOLIDADOS DEL BACKTESTING (112 DÍAS DE EVALUACIÓN - ORDENADOS POR COSTO BRL)")
    print("=" * 80)
    print(df_global.to_string(index=False))
    print("\nArchivos exportados en outputs/metrics/ y gráficos guardados en outputs/figures/.")


if __name__ == "__main__":
    main()