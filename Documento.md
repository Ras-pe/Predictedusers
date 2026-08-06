# Documento de Análisis del Modelo de Predicción de Churn (KKBOX)

> **Nota sobre las gráficas:** este documento tiene reservado un espacio para cada
> gráfica con el marcador `(nombre de la tabla.png)`. Una vez generadas las pruebas
> (con `python src/Pruebas.py`), reemplaza ese texto por el nombre real del archivo
> PNG que se guardó en la carpeta `output/`. La correspondencia se detalla en el
> [Anexo A](#anexo-a-correspondencia-de-placeholders).

---

## 1. Descripción del proyecto

Este proyecto es un **sistema de predicción de abandono de usuarios (churn)** para
una plataforma de streaming de música. El objetivo es **descubrir si un usuario
dejará de usar la aplicación** en el futuro próximo, analizando su comportamiento de
pago y de escucha.

### Pregunta de negocio
¿Qué usuarios están en riesgo de **dejar la aplicación** (cancelar o no renovar su
suscripción) para poder actuar a tiempo con campañas de retención?

### Qué hace el sistema
1. **Entrenamiento del modelo:** `src/analisis_kkbox.py` construye un Random Forest a
   partir del dataset público de churn de KKBOX (transacciones, membresías y logs de
   actividad).
2. **Despliegue del modelo:** `output/rf_churn_model.pkl` se deserializa mediante
   `src/modelo.py` y se consume desde la aplicación web (`web_app/app.py`).
3. **Pruebas y análisis:** `src/Pruebas.py` valida el modelo y genera las gráficas
   que analizan su comportamiento; este documento las explica.

### Variable objetivo
`is_churn`: vale **1 si el usuario abandonó la aplicación** y **0 si continuó
activo**. El modelo estima la probabilidad de que un usuario dado pertenezca al grupo
que deja el servicio, permitiendo clasificarlo en riesgo bajo, medio o alto.

---

## 2. Resumen ejecutivo

El modelo `output/rf_churn_model.pkl` es un **Random Forest Classifier** entrenado
para **descubrir si un usuario dejará de usar la aplicación** (churn, `is_churn`) en
la competencia de churn de KKBOX. Se deserializó correctamente con `joblib` (ver
`src/modelo.py`) y quedó validado: los **18 tests automáticos** ejecutados en
`src/Pruebas.py` terminaron con **14 PASS, 4 WARN y 0 FAIL**.

Las gráficas generadas permiten analizar tres dimensiones del modelo:

1. **Comportamiento global** — cómo distribuye las probabilidades de churn y cómo
   se reparten los usuarios por nivel de riesgo.
2. **Interpretabilidad** — qué características impulsan la predicción (importancia,
   sensibilidad e impacto).
3. **Robustez** — estabilidad ante ruido, incertidumbre entre árboles y respuesta
   ante valores extremos.

---

## 3. Descripción del modelo

### 3.1 ¿Qué es el modelo?

El modelo es un **Random Forest Classifier** (bosque aleatorio), un algoritmo de
aprendizaje supervisado que combina **100 árboles de decisión** independientes. Cada
árbol aprende a clasificar a los usuarios (¿churn o no?) con un subconjunto
aleatorio de datos y características, y la predicción final es el **promedio de las
probabilidades de todos los árboles**. Este enfoque reduce el sobreajuste frente a
un árbol único y es robusto ante variables correlacionadas y ruido, lo que lo hace
adecuado para datos de comportamiento como los de KKBOX.

### 3.2 ¿Qué recibe y qué devuelve?

- **Entrada:** 36 características numéricas que describen a un usuario (historial de
  pagos, actividad de escucha y datos demográficos).
- **Salida:** una **probabilidad de churn** entre 0 y 1 (probabilidad de que el
  usuario deje la aplicación) y una clase binaria: `0` = continúa activo, `1` =
  abandona.

### 3.3 Hiperparámetros del modelo

| Parámetro          | Valor                              | Efecto                                  |
|--------------------|------------------------------------|-----------------------------------------|
| Tipo de modelo     | `RandomForestClassifier`           | Ensamble de árboles de decisión         |
| Número de árboles  | 100                                | Mayor número = predicción más estable   |
| Profundidad máxima | 15                                 | Limita el sobreajuste                   |
| Criterio           | Gini                               | Medida de pureza de las divisiones      |
| `class_weight`     | `balanced`                         | Compensa el desbalance de clases        |
| `max_features`     | `sqrt`                             | Aleatoriza las variables por árbol      |
| `min_samples_leaf` | 10                                 | Evita hojas con muy pocos ejemplos      |
| `random_state`     | 42                                 | Reproducibilidad de los resultados      |
| Características    | 36 (numéricas, agregadas por `msno`) | Perfil completo del usuario           |

### 3.4 ¿Cómo se construyó y cómo se usa?

1. **Entrenamiento** — `src/analisis_kkbox.py` procesa los datasets de KKBOX
   (transacciones, membresías, logs de actividad), agrega las métricas por usuario
   (`msno`), imputa valores faltantes y entrena el bosque con validación cruzada.
   Las predicciones se calibran hacia la clase minoritaria con
   `class_weight='balanced'`, dado que el churn suele ser un evento poco frecuente.
2. **Serialización** — el modelo entrenado se guarda en `output/rf_churn_model.pkl`.
3. **Deserialización** — `src/modelo.py` lo carga con `joblib` y lo expone mediante
   la clase `ModeloChurn`, que valida la estructura y alinea las características por
   nombre para predecir.
4. **Uso en producción** — la aplicación web (`web_app/app.py`) consume el modelo
   para calcular la probabilidad de churn de cada usuario y clasificarlo en riesgo
   **Bajo** (< 0.30), **Medio** (0.30–0.70) o **Alto** (≥ 0.70).

---

## 4. Metodología de las pruebas

Como el dataset original no está disponible en el repositorio (la carpeta `data/`
no existe), las pruebas de `src/Pruebas.py` utilizan una **población sintética
reproducible** de 20 000 usuarios generada con `seed = 42`. Cada característica se
muestrea con una normal truncada centrada en los valores por defecto que usa la web
app, recortada a rangos plausibles definidos por `feature`.

> **Importante:** la población sintética NO es representativa de la población real
> de KKBOX. Las métricas absolutas (p. ej. tasa de churn) deben interpretarse como
> un ejercicio de robustez y análisis de comportamiento del modelo, no como
> estimaciones de la tasa real de cancelación.

---

## 5. Resultados de las pruebas (resumen)

| Resultado | Conteo |
|-----------|--------|
| PASS      | 14     |
| WARN      | 4      |
| FAIL      | 0      |

Los 4 avisos (WARN) son **informativos** y se explican a continuación:

- **T10 (95.5 % de churn a umbral 0.5):** el modelo concentra sus predicciones en
  el rango alto de probabilidad. Es un hallazgo de calibración, no un error.
- **T15/T16 (puntos de corte 0.79 / 0.75):** para clasificar solo al 10 % o 20 % de
  la población como churn, el umbral debe subir a ~0.75–0.79.
- **T18 (incertidumbre media 0.375):** la dispersión entre los 100 árboles es alta,
  lo que sugiere que las probabilidades altas son poco "seguras" en términos de
  consenso entre árboles.

Detalle completo: `output/pruebas_resumen.csv` y `output/prueba_metricas.json`.

---

## 6. Análisis de las gráficas

### 6.1 Distribución de probabilidades de churn

![Histograma de probabilidades de churn predichas](nombre de la tabla.png)

**Explicación:** esta gráfica muestra la distribución de la probabilidad de churn
predicha por el modelo sobre la población sintética, con líneas de referencia en los
umbrales de riesgo medio (0.30), decisión (0.50) y riesgo alto (0.70), además de la
media de la distribución.

**Interpretación:** la masa de probabilidades se concentra entre ~0.6 y ~0.95. El
modelo **apenas emite probabilidades bajas** (menos del 0.1 % de la población queda
por debajo de 0.30). Esto significa que el modelo, tal como fue entrenado
(`class_weight='balanced'`, sin recalibración), tiende a **sobreestimar el riesgo** y
a comprimir sus salidas en un rango estrecho. Es una señal clara de que conviene
recalibrar las probabilidades (p. ej. con `CalibratedClassifierCV`) o reajustar los
umbrales operativos antes de usar la probabilidad como puntaje de riesgo directo.

---

### 6.2 Distribución de usuarios por grupo de riesgo

![Barras de grupos de riesgo Bajo/Medio/Alto](nombre de la tabla.png)

**Explicación:** los usuarios se clasifican en tres grupos según los umbrales de la
web app: **Bajo** (< 0.30), **Medio** (0.30–0.70) y **Alto** (≥ 0.70).

**Interpretación:** en la población sintética el reparto fue aproximadamente
0.06 % / 62.1 % / 37.8 % (Bajo/Medio/Alto). El grupo "Bajo" es prácticamente
inexistente, lo que confirma la observación de la sección anterior: con los umbrales
actuales, casi todos los usuarios quedarían en riesgo medio o alto. Para un
despliegue real, los umbrales 0.30/0.70 deberían recalcularse sobre la distribución
real de probabilidades.

---

### 6.3 Curva de umbrales de decisión

![Curva de % de churn predicho vs umbral](nombre de la tabla.png)

**Explicación:** para cada umbral entre 0 y 1, se grafica el porcentaje de la
población que el modelo clasificaría como churn.

**Interpretación:** la curva cae lentamente al inicio, lo que implica que se necesitan
umbrales muy altos para reducir la proporción de churn predicho. Por ejemplo, para
que solo el 10 % de la población sea marcada como churn, el umbral debe estar en
**0.79**; para el 20 %, en **0.75**. Esto es útil para elegir el punto de corte
operativo según el costo que la empresa asigne a los falsos positivos (retenciones
innecesarias) frente a los falsos negativos (usuarios perdidos).

---

### 6.4 Importancia de características (Gini)

![Barras horizontales de importancia de las 20 mejores features](nombre de la tabla.png)

**Explicación:** importancia media de Gini que el propio bosque asigna a cada
característica en sus nodos de decisión.

**Interpretación:** el modelo depende fuertemente de variables de **historial de
pago y actividad**:

- `membership_duration_days` (19.2 %)
- `days_since_last_expire` (13.9 %)
- `actual_amount_paid_sum` (11.3 %)
- `days_since_last_txn` (10.2 %)
- `is_cancel_rate` (6.6 %)

Que las 4 primeras representen más del 54 % de la importancia confirma que el
comportamiento de pago es el principal motor del churn, y que las variables de
escucha (`num_*`, `total_secs_*`) aportan poco de forma individual.

---

### 6.5 Curvas de sensibilidad de las características principales

![Matriz de curvas de respuesta del modelo por feature](nombre de la tabla.png)

**Explicación:** cada subgráfica varía una de las 6 características más importantes a
lo largo de su rango (dejando el resto en el valor por defecto) y traza la
probabilidad de churn resultante. La línea gris punteada es la probabilidad base del
usuario típico (0.97).

**Interpretación:** las curvas permiten ver si la relación es monótona o presenta
"escalones" (propios de un árbol). Por ejemplo, `days_since_last_expire` y
`days_since_last_txn` muestran un descenso marcado en su punto óptimo (~13 y ~41
días respectivamente) seguido de un repunte: el modelo detecta que tanto el expirar
hace poco como el llevar mucho tiempo sin actividad son situaciones de riesgo. Estas
curvas son clave para explicar el modelo a negocio y para detectar comportamientos
poco intuitivos que convenga revisar.

---

### 6.6 Incertidumbre del modelo (desviación entre árboles)

![Curva de desviación estándar entre árboles vs probabilidad predicha](nombre de la tabla.png)

**Explicación:** para una muestra de 3000 usuarios se calcula la desviación estándar
de las 100 predicciones (una por árbol) y se promedia por bin de probabilidad
predicha.

**Interpretación:** la incertidumbre media fue **0.375**, un valor alto: los árboles
del bosque no convergen en la misma clase. En general, los Random Forests tienden a
mostrar mayor dispersión cuando la probabilidad se acerca a 0.5. Aquí la curva indica
cuán "seguro" es el modelo en cada zona de probabilidad y ayuda a decidir dónde
colocar los umbrales: conviene que el punto de corte quede en una zona de baja
incertidumbre.

---

### 6.7 Ranking de impacto de características

![Barras horizontales del rango de impacto (max-min) por feature](nombre de la tabla.png)

**Explicación:** para cada característica se calcula el rango de probabilidad
`max - min` al recorrer sus valores (resto en valores por defecto). Mide cuánto
puede mover la predicción cada variable de forma individual.

**Interpretación:** el ranking de impacto es diferente del de importancia de Gini.
Aquí lidera `payment_method_mode` (impacto 0.583), seguida de `days_since_last_txn`
(0.468), `days_since_last_expire` (0.388) y `actual_amount_paid_sum` (0.344). Mientras
la importancia de Gini premia variables usadas en muchos árboles, el impacto mide el
rango de influencia práctica: qué palanca modifica más el riesgo individual. Ambas
vistas se complementan para priorizar qué datos recoger o qué acciones de retención
ofrecer.

---

## 7. Justificación de la eficiencia del modelo

El modelo demuestra eficiencia en los siguientes puntos:

1. **Carga y predicción rápidas.** Es un bosque de 100 árboles de profundidad 15
   sobre 36 features numéricas. En el entorno de prueba predice sobre 20 000 filas
   en segundos, y es apto para servir peticiones en tiempo real desde la web app sin
   hardware especializado.
2. **Robustez numérica.** Las probabilidades siempre están dentro de `[0, 1]`, las
   predicciones son **deterministas** (misma entrada → misma salida, test T07) y la
   predicción es **estable ante ruido pequeño** (desviación media de 0.0011 ante
   perturbaciones del 1 %, test T14).
3. **Alta capacidad discriminativa parcial.** Las curvas de sensibilidad muestran
   que el modelo reacciona de forma diferenciada a las variables de pago, lo que
   indica que es sensible a los comportamientos que la literatura asocia al churn.
4. **Interpretabilidad.** Al ser un árbol de decisión por ensamble, expone
   importancias y curvas de respuesta directamente, sin modelos adicionales. Esto
   facilita la auditoría y la explicación a áreas de negocio.
5. **Gestión del desbalance.** El uso de `class_weight='balanced'` evita que el
   modelo prediga siempre la clase mayoritaria, un requisito habitual en churn.

### Limitaciones que deben corregirse

- **Calibración deficiente:** las probabilidades se concentran en un rango alto
  (media 0.67, máximo 0.95) y casi nadie cae en riesgo "Bajo". Esto infla la tasa de
  churn a umbral 0.5 (95.5 % en la población sintética) y exige recalibrar o
  reentrenar con la distribución real.
- **Alta incertidumbre entre árboles (0.375):** conviene validar si más árboles o un
  ajuste de hiperparámetros reduce la dispersión.
- **Validación con datos reales pendiente:** las métricas de este documento se
  calcularon sobre población sintética. Para reportar ROC-AUC, matriz de confusión o
  lift sobre datos reales, se debe reentrenar usando la carpeta `data/`
  (instrucciones en `src/analisis_kkbox.py`) y ejecutar las pruebas con esa data.

---

## 8. Conclusiones y recomendaciones

1. El modelo **funciona y es desplegable**: se deserializa, valida y predice de forma
   estable (0 FAIL en 18 pruebas).
2. Es **interpretable** y prioriza correctamente las variables de pago como motor del
   churn.
3. Antes de producción se recomienda: **(a)** recalibrar las probabilidades,
   **(b)** reajustar los umbrales de riesgo de la web app según la curva de umbrales,
   y **(c)** reentrenar con `data/` para obtener métricas de discriminación reales
   (AUC, precisión, recall, F1).

---

## Anexo A: correspondencia de placeholders

| Marcador en el documento                  | Archivo generado (output/)                     |
|-------------------------------------------|------------------------------------------------|
| `(nombre de la tabla.png)` de la sección 6.1 | `prueba_01_distribucion_probabilidades.png` |
| `(nombre de la tabla.png)` de la sección 6.2 | `prueba_02_riesgo_por_grupo.png`           |
| `(nombre de la tabla.png)` de la sección 6.3 | `prueba_03_curva_umbral.png`               |
| `(nombre de la tabla.png)` de la sección 6.4 | `prueba_04_importancia_features.png`       |
| `(nombre de la tabla.png)` de la sección 6.5 | `prueba_05_sensibilidad_features.png`      |
| `(nombre de la tabla.png)` de la sección 6.6 | `prueba_06_incertidumbre.png`              |
| `(nombre de la tabla.png)` de la sección 6.7 | `prueba_07_ranking_sensibilidad.png`       |

## Anexo B: archivos generados por las pruebas

| Archivo                             | Contenido                                            |
|-------------------------------------|------------------------------------------------------|
| `output/pruebas_resumen.csv`        | Resultado de los 18 tests (test_id, nombre, estado)  |
| `output/prueba_metricas.json`       | Métricas completas (estadísticas, importancias, impacto) |
| `src/modelo.py`                     | Deserializador/wrapper del modelo (`ModeloChurn`)    |
| `src/Pruebas.py`                    | Script de pruebas y generación de gráficas           |

---

*Documento generado para el análisis del modelo `output/rf_churn_model.pkl`.*
