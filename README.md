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
│       ├── ml_quantile.py    # LightGBM (L2, Q50, Q80) con estacionamiento local y CQR
│       └── foundation.py     # Chronos-2 y TimesFM 3 en zero-shot (Q80 crudo y con CQR)
├── main.py                   # Punto de entrada único del pipeline completo
├── requirements.txt          # Dependencias reproducibles
└── requirements-foundation.txt  # Dependencias opcionales de los modelos fundacionales
```

## 3. Modelos Fundacionales (Opcional)
`main.py` agrega **Chronos-2** (Amazon) y **TimesFM 3** (Google Research) como modelos 8 a 11 si sus dependencias están instaladas; si no, los omite con un aviso y corre los 7 modelos originales.

- Se usan en **zero-shot** (sin entrenamiento), con el mismo protocolo h=1: el contexto es la demanda real hasta t-1 más las variables de calendario que usa SARIMAX.
- Se reporta el cuantil 0.80 crudo y con el mismo ajuste conforme (CQR) que LightGBM, calibrado en los últimos 28 días de train.
- Los pesos de TimesFM 3 tienen licencia no comercial. Olist es un dataset público: no se puede descartar que haya formado parte del preentrenamiento de estos modelos.

```text
pip install -r requirements-foundation.txt -c requirements.txt
```