# Forecasting de Ventas con Función de Pérdida Asimétrica (Olist E-Commerce)

Proyecto final integrador de **Machine Learning (Series Temporales)**.
**Autores:** Daniela Salgado y Santiago Fernández

## 1. Planteamiento del Problema y Originalidad
En la gestión de inventarios para e-commerce (*fulfillment*), minimizar métricas estadísticas simétricas como **RMSE** o **MAE** es subóptimo desde el punto de vista económico: el costo de quedarse sin stock (*under-stocking*: margen perdido + penalización por demora) es sustancialmente superior al costo de almacenar una unidad extra (*over-stocking*: costo diario de bodega y capital inmovilizado).

Este proyecto conecta formalmente el **Problema del Vendedor de Periódicos (*Newsvendor Problem*)** con el entrenamiento directo mediante **Pinball Loss (Regresión Cuantílica)** y **Calibración Cuantílica Conforme (CQR)** sobre datos transaccionales reales del *Brazilian E-Commerce Public Dataset (Olist)* en la categoría líder `cama_mesa_banho`.

$$\alpha^* = \frac{c_u}{c_u + c_o} = \frac{20\text{ BRL}}{20\text{ BRL} + 5\text{ BRL}} = 0.80$$

## 2. Arquitectura del Proyecto (Sin Notebooks)
```text
ml_time_series_olist/
├── data/
│   ├── raw/                  # CSVs transaccionales crudos de Olist (no versionados)
│   └── processed/            # Serie diaria reindexada y matriz de features
├── outputs/
│   ├── figures/              # Gráficos en alta resolución (300 DPI)
│   └── metrics/              # Tablas CSV de resultados por corte y consolidados
├── src/
│   ├── config.py             # Hiperparámetros de negocio, rutas y cuantil crítico
│   ├── data_builder.py       # Cruce relacional, filtrado y reindexación diaria sin fugas
│   ├── features.py           # Lags, ventanas móviles desplazadas (.shift(1)) y calendario
│   ├── validation.py         # Esquema Walk-Forward Expanding Window (4 cortes x 28 días)
│   ├── evaluation.py         # Pinball Loss, RMSE, MAE y Costo Económico en BRL
│   └── models/
│       ├── statistical.py    # Baseline Empírico y SARIMAX(1,0,1)x(1,0,1,7) con estado Kalman h=1
│       └── ml_quantile.py    # LightGBM (L2, Q50, Q80) con estacionamiento local y CQR
├── main.py                   # Punto de entrada único del pipeline completo
└── requirements.txt          # Dependencias reproducibles