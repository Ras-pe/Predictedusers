<div align="center">

# KKBOX Churn Prediction

**Descubre si un usuario dejará de usar la aplicación**

Un sistema de Machine Learning para predecir el **abandono de usuarios (churn)** de
una plataforma de streaming de música, entrenado con el dataset público de KKBOX.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Modelo](https://img.shields.io/badge/Modelo-Random%20Forest-6DB33F)](https://scikit-learn.org/stable/modules/ensemble.html)
[![Pruebas](https://img.shields.io/badge/Pruebas-14%20PASS%20%E2%80%A2%200%20FAIL-brightgreen)](output/pruebas_resumen.csv)
[![Estado](https://img.shields.io/badge/Estado-Desplegable-blue)](web_app/app.py)

</div>

---

## Contenido

- [Descripción del proyecto](#descripción-del-proyecto)
- [Características](#características)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Aplicación web](#aplicación-web)
  - [Pruebas y gráficas](#pruebas-y-gráficas)
  - [Deserializar el modelo](#deserializar-el-modelo)
- [Reentrenar el modelo](#reentrenar-el-modelo)
- [Resultados de las pruebas](#resultados-de-las-pruebas)
- [Documentación](#documentación)
- [Tecnologías](#tecnologías)

---

## Descripción del proyecto

El objetivo del proyecto es **detectar qué usuarios están en riesgo de dejar la
aplicación** (no renovar o cancelar su suscripción) para poder actuar a tiempo con
campañas de retención.

A partir del historial de **pagos, membresías y actividad de escucha** de cada
usuario, se entrena un **Random Forest Classifier** que estima la probabilidad de
churn. El modelo deserializado se integra en una aplicación web que clasifica a cada
usuario en riesgo **Bajo**, **Medio** o **Alto**.

## Características

- **Modelo Random Forest** de 100 árboles con `class_weight='balanced'` para
  compensar el desbalance de clases típico del churn.
- **36 características** numéricas por usuario: comportamiento de pago, actividad de
  escucha y datos demográficos.
- **Módulo de deserialización** (`src/modelo.py`) que valida y alinea las
  características automáticamente por nombre.
- **Suite de pruebas** (`src/Pruebas.py`) con 18 tests y **7 gráficas de análisis**
  generadas automáticamente en `output/`.
- **Aplicación web** en Flask para predecir churn en tiempo real.
- **Documento de análisis** (`Documento.md`) que explica cada gráfica y justifica la
  eficiencia del modelo.

## Estructura del repositorio

```text
Predictedusers/
├── data/                    # Datos crudos (NO se suben a Git, solo para reentrenar)
├── output/
│   ├── rf_churn_model.pkl   # Modelo entrenado (serializado con joblib)
│   ├── prueba_*.png         # Gráficas generadas por las pruebas
│   └── prueba_*.csv/json    # Reportes de métricas
├── src/
│   ├── analisis_kkbox.py    # Entrenamiento, EDA y selección de variables
│   ├── modelo.py            # Deserializador / wrapper del modelo (ModeloChurn)
│   ├── Pruebas.py           # Suite de pruebas + generación de gráficas
│   └── scala_originales/    # Scripts originales del labeller (referencia)
├── web_app/
│   ├── app.py               # API Flask /predict
│   └── templates/index.html # Interfaz web
├── Documento.md             # Análisis del modelo con gráficas
└── README.md
```

## Requisitos

- Python **3.10+**
- Paquetes: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`,
  `joblib` y `flask`

## Instalación

```bash
# 1) Clonar el repositorio
git clone <url-del-repositorio>
cd Predictedusers

# 2) Crear y activar un entorno virtual (opcional pero recomendado)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3) Instalar dependencias
pip install scikit-learn pandas numpy matplotlib seaborn joblib flask
```

## Uso

### Aplicación web

Inicia el servidor y abre `http://localhost:5000`:

```bash
python web_app/app.py
```

La API acepta las 36 características del usuario y devuelve la predicción:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"txn_count": 3, "actual_amount_paid_sum": 300, "membership_duration_days": 120}'
```

Respuesta:

```json
{
  "prediction": 1,
  "probability": 0.7814,
  "label": "Churn",
  "risk": "Alto"
}
```

### Pruebas y gráficas

Ejecuta los 18 tests y genera las 7 gráficas de análisis:

```bash
python src/Pruebas.py
```

Los resultados quedan en `output/`:

| Archivo                          | Contenido                                    |
|----------------------------------|----------------------------------------------|
| `prueba_01_distribucion_probabilidades.png` | Distribución de probabilidades predichas |
| `prueba_02_riesgo_por_grupo.png` | Usuarios por grupo de riesgo                 |
| `prueba_03_curva_umbral.png`     | Análisis de umbrales de decisión             |
| `prueba_04_importancia_features.png` | Importancia de las 20 mejores variables  |
| `prueba_05_sensibilidad_features.png` | Curvas de respuesta por característica   |
| `prueba_06_incertidumbre.png`    | Incertidumbre entre árboles del bosque       |
| `prueba_07_ranking_sensibilidad.png` | Ranking de impacto de características    |
| `pruebas_resumen.csv`            | Resultado de los 18 tests                    |
| `prueba_metricas.json`           | Métricas completas del análisis              |

### Deserializar el modelo

Usa el wrapper directamente desde cualquier script:

```python
import sys
sys.path.insert(0, 'src')
from modelo import ModeloChurn

modelo = ModeloChurn()  # carga output/rf_churn_model.pkl
usuario = {"membership_duration_days": 120, "days_since_last_txn": 45}
proba = modelo.probabilidad_churn(usuario)
print(f"Probabilidad de churn: {proba[0]:.2%}")
```

## Reentrenar el modelo

> **Nota:** para reentrenar, agrega la carpeta `data/` con los archivos fuente de
> KKBOX (`train_v2.csv.csv`, `members_v3.csv.csv`, `transactions_v2.csv.csv` y
> `user_logs_v2.csv.csv`) y ejecuta:

```bash
python src/analisis_kkbox.py
```

El script realiza la carga y limpieza, el análisis exploratorio (EDA), la selección
de variables y guarda el nuevo modelo en `output/rf_churn_model.pkl`, sobrescribiendo
el existente.

> **Nota:** los archivos de `data/` están en `.gitignore` por su tamaño, no se
> versionan.

## Resultados de las pruebas

La suite de pruebas sobre una población sintética reproducible (`seed = 42`) reporta:

| Métrica                  | Resultado                          |
|--------------------------|------------------------------------|
| Pruebas ejecutadas       | 18                                 |
| PASS                     | 14                                 |
| WARN                     | 4 (informativo, sin errores)       |
| FAIL                     | 0                                  |
| Desviación ante ruido 1% | 0.0011                             |
| Incertidumbre media      | 0.375                              |

Detalle completo en `output/pruebas_resumen.csv`.

## Documentación

El documento **[`Documento.md`](Documento.md)** describe el proyecto, explica cada
gráfica generada por las pruebas y justifica la eficiencia del modelo, incluyendo
sus limitaciones (calibración y validación con datos reales) y las recomendaciones
para llevarlo a producción.

## Tecnologías

| Tecnología     | Uso                                   |
|----------------|---------------------------------------|
| Python         | Lenguaje principal                    |
| scikit-learn   | Modelo Random Forest y métricas       |
| pandas / numpy | Manipulación y agregación de datos    |
| matplotlib / seaborn | Visualizaciones y gráficas     |
| joblib         | Serialización / deserialización del modelo |
| Flask          | Aplicación web de predicción          |

---

<div align="center">

**Proyecto académico de predicción de churn basado en la competencia KKBOX
Churn Prediction Challenge.**

</div>
