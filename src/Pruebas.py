"""
==============================================================================
 PRUEBAS DEL MODELO KKBOX CHURN PREDICTION
==============================================================================
Objetivo: verificar que el modelo deserializado (rf_churn_model.pkl)
funciona correctamente, generar métricas de robustez y producir gráficas
que se almacenan en output/ para su posterior análisis en Documento.md.

Como no existe el dataset original (carpeta data/), las pruebas se apoyan
en una población sintética reproducible construida a partir de los valores
por defecto usados por web_app/app.py y rangos plausibles por feature.

Gráficas generadas (en output/):
  prueba_01_distribucion_probabilidades.png
  prueba_02_riesgo_por_grupo.png
  prueba_03_curva_umbral.png
  prueba_04_importancia_features.png
  prueba_05_sensibilidad_features.png
  prueba_06_incertidumbre.png
  prueba_07_ranking_sensibilidad.png

Reportes (en output/):
  pruebas_resumen.csv
  prueba_metricas.json
==============================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelo import ModeloChurn  # noqa: E402

warnings.filterwarnings('ignore')

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT, exist_ok=True)

SEED = 42
np.random.seed(SEED)
N_POBLACION = 20_000
N_INCERTIDUMBRE = 3_000

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 120,
    'figure.figsize': (12, 6),
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

UMBRAL_BAJO = 0.30
UMBRAL_ALTO = 0.70

DEFAULTS = {
    'city': 1.0, 'registered_via': 7.0, 'txn_count': 1.0,
    'payment_method_mode': 41.0, 'payment_plan_days_mean': 30.0,
    'payment_plan_days_max': 30.0, 'plan_list_price_mean': 149.0,
    'plan_list_price_max': 149.0, 'actual_amount_paid_mean': 149.0,
    'actual_amount_paid_sum': 149.0, 'actual_amount_paid_max': 149.0,
    'is_auto_renew_rate': 1.0, 'is_cancel_rate': 0.0, 'is_cancel_sum': 0.0,
    'days_since_last_txn': 43.0, 'days_since_last_expire': 11.0,
    'membership_duration_days': 31.0, 'log_count': 19.0,
    'num_25_mean': 3.4545, 'num_25_sum': 55.0, 'num_50_mean': 0.8889,
    'num_50_sum': 14.0, 'num_75_mean': 0.6, 'num_75_sum': 10.0,
    'num_985_mean': 0.6087, 'num_985_sum': 10.0, 'num_100_mean': 17.4545,
    'num_100_sum': 305.0, 'num_unq_mean': 19.2333, 'num_unq_sum': 330.0,
    'total_secs_mean': 4709.0312, 'total_secs_sum': 82640.24,
    'total_secs_max': 13730.845, 'total_secs_std': 3873.3673,
    'days_since_last_log': 30.0, 'days_between_first_last_log': 29.0,
}

RANGOS = {
    'city': (1, 99), 'registered_via': (1, 50), 'txn_count': (1, 30),
    'payment_method_mode': (1, 100), 'payment_plan_days_mean': (1, 90),
    'payment_plan_days_max': (1, 365), 'plan_list_price_mean': (1, 400),
    'plan_list_price_max': (1, 400), 'actual_amount_paid_mean': (1, 400),
    'actual_amount_paid_sum': (1, 5000), 'actual_amount_paid_max': (1, 400),
    'is_auto_renew_rate': (0, 1), 'is_cancel_rate': (0, 1),
    'is_cancel_sum': (0, 30), 'days_since_last_txn': (0, 400),
    'days_since_last_expire': (0, 400), 'membership_duration_days': (0, 500),
    'log_count': (1, 500), 'num_25_mean': (0, 100), 'num_25_sum': (0, 5000),
    'num_50_mean': (0, 100), 'num_50_sum': (0, 5000),
    'num_75_mean': (0, 100), 'num_75_sum': (0, 5000),
    'num_985_mean': (0, 100), 'num_985_sum': (0, 5000),
    'num_100_mean': (0, 200), 'num_100_sum': (0, 10000),
    'num_unq_mean': (0, 300), 'num_unq_sum': (0, 20000),
    'total_secs_mean': (0, 20000), 'total_secs_sum': (0, 1000000),
    'total_secs_max': (0, 100000), 'total_secs_std': (0, 20000),
    'days_since_last_log': (0, 400), 'days_between_first_last_log': (0, 500),
}

# ─── Estado de las pruebas ────────────────────────────────────────────────────
PRUEBAS = []


def registrar(test_id, nombre, estado, detalle):
    PRUEBAS.append({'test_id': test_id, 'nombre': nombre,
                    'estado': estado, 'detalle': str(detalle)})
    print(f'   [{estado:>4}] {test_id} - {nombre}: {detalle}')


def guardar_figura(fig, nombre):
    ruta = os.path.join(OUTPUT, nombre)
    fig.savefig(ruta, bbox_inches='tight')
    plt.close(fig)
    print(f'   -> Gráfico guardado: output/{nombre}')
    return ruta


def generar_poblacion(modelo, n=N_POBLACION):
    datos = {}
    for feat in modelo.features:
        lo, hi = RANGOS[feat]
        base = DEFAULTS[feat]
        sigma = (hi - lo) / 4.0
        if lo == 0 and base <= 1:
            sigma = max(0.1, (hi - lo) / 4.0)
        val = np.random.normal(base, sigma, n)
        datos[feat] = np.clip(val, lo, hi)
    return pd.DataFrame(datos, columns=modelo.features)


def punto_de_corte_operativo(umbrales, proba, objetivo):
    for t in umbrales:
        if (proba >= t).mean() <= objetivo:
            return t
    return float(umbrales[-1])


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return jsonable(obj.tolist())
    return obj


print('=' * 70)
print('     PRUEBAS DEL MODELO KKBOX CHURN — DESERIALIZADO')
print('=' * 70)

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 1: CARGA, VALIDACIÓN Y ESTRUCTURA DEL MODELO                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print('\n[1] Carga y validación del modelo deserializado...')

try:
    modelo = ModeloChurn()
    res = modelo.resumen()
except Exception as e:
    print(f'   [FALLO] No se pudo cargar el modelo: {e}')
    sys.exit(1)

registrar('T01', 'Carga del modelo desde .pkl', 'PASS',
          f'{res["tipo"]} con {res["n_estimadores"]} árboles, '
          f'max_depth={res["max_depth"]}')
registrar('T02', 'Tipo de modelo esperado', 'PASS' if res['tipo'] == 'RandomForestClassifier' else 'FAIL',
          res['tipo'])
registrar('T03', 'Número de características (36)', 'PASS' if res['n_features'] == 36 else 'FAIL',
          f'{res["n_features"]} features')
registrar('T04', 'Clases del problema [0, 1]', 'PASS' if set(map(int, res['clases'])) == {0, 1} else 'FAIL',
          res['clases'])

feats_webapp = set(DEFAULTS.keys())
faltan = sorted(feats_webapp - set(modelo.features))
sobran = sorted(set(modelo.features) - feats_webapp)
registrar('T05', 'Coherencia de features con web_app', 'PASS' if not faltan and not sobran else 'WARN',
          f'faltan={faltan} sobran={sobran}')

usuario_tipico = pd.DataFrame([DEFAULTS], columns=modelo.features)
p0 = float(modelo.probabilidad_churn(usuario_tipico)[0])
registrar('T06', 'Predicción con usuario típico válida', 'PASS' if 0.0 <= p0 <= 1.0 else 'FAIL',
          f'p(churn)={p0:.4f}')

p1 = float(modelo.probabilidad_churn(usuario_tipico)[0])
registrar('T07', 'Determinismo de la predicción', 'PASS' if np.isclose(p0, p1) else 'FAIL',
          f'{p0:.6f} == {p1:.6f}')

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 2: GENERACIÓN DE POBLACIÓN SINTÉTICA Y PREDICCIÓN                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print('\n[2] Generando población sintética reproducible...')
poblacion = generar_poblacion(modelo, N_POBLACION)
proba = modelo.probabilidad_churn(poblacion)
pred = modelo.predecir(poblacion)

registrar('T08', 'Población sintética generada', 'PASS',
          f'{N_POBLACION:,} filas x {poblacion.shape[1]} features')
registrar('T09', 'Probabilidades dentro de [0, 1]', 'PASS' if (proba >= 0).all() and (proba <= 1).all() else 'FAIL',
          f'min={proba.min():.4f} max={proba.max():.4f}')

pct_churn = float((pred == 1).mean() * 100)
registrar('T10', 'Tasa de churn predicha (umbral 0.5)', 'WARN',
          f'{pct_churn:.2f}% de la población sintética')

grupos = pd.Series(np.where(proba >= UMBRAL_ALTO, 'Alto',
                            np.where(proba >= UMBRAL_BAJO, 'Medio', 'Bajo')))
pct_grupos = grupos.value_counts(normalize=True) * 100
registrar('T11', 'Grupos de riesgo (Bajo/Medio/Alto) calculados', 'PASS',
          {k: f'{v:.2f}%' for k, v in pct_grupos.items()})

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 3: PRUEBAS DE ROBUSTEZ                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print('\n[3] Pruebas de robustez...')

min_row = {f: RANGOS[f][0] for f in modelo.features}
max_row = {f: RANGOS[f][1] for f in modelo.features}
p_min = float(modelo.probabilidad_churn(pd.DataFrame([min_row], columns=modelo.features))[0])
p_max = float(modelo.probabilidad_churn(pd.DataFrame([max_row], columns=modelo.features))[0])
registrar('T12', 'Caso extremo (mínimos) produce probabilidad válida',
          'PASS' if 0.0 <= p_min <= 1.0 else 'FAIL', f'p={p_min:.4f}')
registrar('T13', 'Caso extremo (máximos) produce probabilidad válida',
          'PASS' if 0.0 <= p_max <= 1.0 else 'FAIL', f'p={p_max:.4f}')

rng = np.random.default_rng(SEED)
ruido = rng.normal(0, 0.01, size=poblacion.shape)
proba_ruido = modelo.probabilidad_churn(poblacion + ruido)
desv_media = float(np.mean(np.abs(proba_ruido - proba)))
registrar('T14', 'Estabilidad ante ruido pequeño (1%)',
          'PASS' if desv_media < 0.05 else 'WARN',
          f'desviación media = {desv_media:.5f}')

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 4: GRÁFICAS                                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print('\n[4] Generando gráficas...')

# ─── 4a) Distribución de probabilidades de churn ─────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(proba, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='Umbral riesgo medio (0.30)')
ax.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Umbral de decisión (0.50)')
ax.axvline(0.7, color='red', linestyle='--', linewidth=2, label='Umbral riesgo alto (0.70)')
ax.axvline(proba.mean(), color='purple', linestyle='-', linewidth=2, label=f'Media ({proba.mean():.3f})')
ax.set_xlabel('Probabilidad de churn predicha')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución de probabilidades de churn — población sintética', fontweight='bold')
ax.legend()
plt.tight_layout()
guardar_figura(fig, 'prueba_01_distribucion_probabilidades.png')

# ─── 4b) Grupos de riesgo ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
colores = {'Bajo': 'steelblue', 'Medio': 'orange', 'Alto': 'crimson'}
orden = ['Bajo', 'Medio', 'Alto']
pct_orden = [pct_grupos.get(g, 0) for g in orden]
ax.bar(orden, pct_orden, color=[colores[g] for g in orden], edgecolor='black')
for p, v in zip(ax.patches, pct_orden):
    ax.annotate(f'{v:.2f}%', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=11)
ax.set_ylabel('Porcentaje de usuarios (%)')
ax.set_title(f'Distribución de usuarios por grupo de riesgo '
             f'(umbrales {UMBRAL_BAJO} / {UMBRAL_ALTO})', fontweight='bold')
plt.tight_layout()
guardar_figura(fig, 'prueba_02_riesgo_por_grupo.png')

# ─── 4c) Análisis de umbrales ────────────────────────────────────────────────
umbrales = np.linspace(0, 1, 101)
tasa_churn = [(proba >= t).mean() for t in umbrales]
tasa_churn = np.asarray(tasa_churn)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(umbrales, tasa_churn * 100, color='darkorange', linewidth=2.5)
ax.axvline(0.3, color='orange', linestyle='--', alpha=0.8, label='Riesgo medio (0.30)')
ax.axvline(0.5, color='green', linestyle='--', alpha=0.8, label='Decisión (0.50)')
ax.axvline(0.7, color='red', linestyle='--', alpha=0.8, label='Riesgo alto (0.70)')
ax.set_xlabel('Umbral de probabilidad')
ax.set_ylabel('% de usuarios clasificados como churn')
ax.set_title('Curva de umbrales: % de churn predicho vs. umbral', fontweight='bold')
ax.legend()
plt.tight_layout()
guardar_figura(fig, 'prueba_03_curva_umbral.png')

corte_10 = punto_de_corte_operativo(umbrales, proba, 0.10)
corte_20 = punto_de_corte_operativo(umbrales, proba, 0.20)
registrar('T15', 'Punto de corte para 10% de churn predicho', 'WARN', f't = {corte_10:.2f}')
registrar('T16', 'Punto de corte para 20% de churn predicho', 'WARN', f't = {corte_20:.2f}')

# ─── 4d) Importancia de características del modelo ────────────────────────────
fi = pd.DataFrame({
    'feature': modelo.features,
    'importance': modelo.modelo.feature_importances_,
}).sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi)))
ax.barh(fi['feature'][::-1], fi['importance'][::-1], color=colors[::-1], edgecolor='black')
ax.set_xlabel('Importancia (Gini)')
ax.set_title('Top 20 características — RandomForestClassifier (importancia del modelo)', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
guardar_figura(fig, 'prueba_04_importancia_features.png')

# ─── 4e) Sensibilidad de características (curvas de respuesta) ───────────────
top6 = fi.head(6)['feature'].tolist()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
for i, feat in enumerate(top6):
    ax = axes[i // 3, i % 3]
    lo, hi = RANGOS[feat]
    grid = np.linspace(lo, hi, 100)
    filas = []
    for g in grid:
        fila = dict(DEFAULTS)
        fila[feat] = g
        filas.append(fila)
    df_grid = pd.DataFrame(filas, columns=modelo.features)
    p_grid = modelo.probabilidad_churn(df_grid)
    ax.plot(grid, p_grid, color='crimson', linewidth=2.5)
    ax.axhline(p0, color='grey', linestyle='--', linewidth=1, label=f'Base ({p0:.2f})')
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('Valor de la feature')
    ax.set_ylabel('P(churn)')
    ax.grid(True, alpha=0.3)
fig.suptitle('Curvas de respuesta del modelo (resto de features en valores por defecto)', fontweight='bold', fontsize=13)
plt.tight_layout()
guardar_figura(fig, 'prueba_05_sensibilidad_features.png')

impacto = []
for feat in modelo.features:
    lo, hi = RANGOS[feat]
    grid = np.linspace(lo, hi, 50)
    filas = []
    for g in grid:
        fila = dict(DEFAULTS)
        fila[feat] = g
        filas.append(fila)
    df_grid = pd.DataFrame(filas, columns=modelo.features)
    p_grid = modelo.probabilidad_churn(df_grid)
    impacto.append({'feature': feat, 'impacto': float(p_grid.max() - p_grid.min())})
impacto_df = pd.DataFrame(impacto).sort_values('impacto', ascending=False)
registrar('T17', 'Rango de impacto máximo por feature',
          'PASS', f"{impacto_df.iloc[0]['feature']} -> {impacto_df.iloc[0]['impacto']:.3f}")

fig, ax = plt.subplots(figsize=(10, 7))
imp12 = impacto_df.head(12).iloc[::-1]
ax.barh(imp12['feature'], imp12['impacto'], color='teal', edgecolor='black')
ax.set_xlabel('Rango de P(churn) (max - min) al variar la feature')
ax.set_title('Top 12 features por impacto en la predicción', fontweight='bold')
plt.tight_layout()
guardar_figura(fig, 'prueba_07_ranking_sensibilidad.png')

# ─── 4f) Incertidumbre del modelo (std entre árboles) ────────────────────────
idx_inc = np.random.choice(N_POBLACION, size=N_INCERTIDUMBRE, replace=False)
X_inc = poblacion.iloc[idx_inc]
proba_arboles = np.column_stack([
    est.predict_proba(X_inc)[:, 1] for est in modelo.modelo.estimators_
])
std_arboles = proba_arboles.std(axis=1)
proba_inc = proba[idx_inc]

bins = np.linspace(0, 1, 21)
idx_bin = np.digitize(proba_inc, bins) - 1
agg = pd.DataFrame({'bin': idx_bin, 'proba': proba_inc, 'std': std_arboles})
agg_med = agg.groupby('bin').agg(proba_media=('proba', 'mean'), std_media=('std', 'mean')).dropna()

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(agg_med['proba_media'], agg_med['std_media'], color='purple', marker='o', linewidth=2.5)
ax.fill_between(agg_med['proba_media'], agg_med['std_media'], alpha=0.15, color='purple')
ax.set_xlabel('Probabilidad de churn predicha (promedio del bin)')
ax.set_ylabel('Desviación estándar entre árboles')
ax.set_title('Incertidumbre del modelo (variabilidad entre los 100 árboles)', fontweight='bold')
plt.tight_layout()
guardar_figura(fig, 'prueba_06_incertidumbre.png')

registrar('T18', 'Incertidumbre media entre árboles', 'WARN',
          f'{std_arboles.mean():.4f} (p50={np.median(std_arboles):.4f})')

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 5: REPORTES                                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print('\n[5] Guardando reportes...')

df_report = pd.DataFrame(PRUEBAS)
ruta_csv = os.path.join(OUTPUT, 'pruebas_resumen.csv')
df_report.to_csv(ruta_csv, index=False)
print(f'   -> Guardado: output/pruebas_resumen.csv')

metricas = {
    'seed': SEED,
    'n_poblacion': N_POBLACION,
    'n_incertidumbre': N_INCERTIDUMBRE,
    'modelo': res,
    'usuario_tipico_probabilidad': p0,
    'proba_stats': {
        'media': float(proba.mean()),
        'std': float(proba.std()),
        'p25': float(np.percentile(proba, 25)),
        'p50': float(np.percentile(proba, 50)),
        'p75': float(np.percentile(proba, 75)),
        'p90': float(np.percentile(proba, 90)),
        'min': float(proba.min()),
        'max': float(proba.max()),
    },
    'tasa_churn_0_5_pct': pct_churn,
    'grupos_riesgo_pct': pct_grupos.to_dict(),
    'puntos_corte': {'10_pct': float(corte_10), '20_pct': float(corte_20)},
    'estabilidad_ruido_1pct': desv_media,
    'incertidumbre_media_arboles': float(std_arboles.mean()),
    'importancia_top20': fi.to_dict('records'),
    'impacto_top12': impacto_df.head(12).to_dict('records'),
}
ruta_json = os.path.join(OUTPUT, 'prueba_metricas.json')
with open(ruta_json, 'w', encoding='utf-8') as f:
    json.dump(jsonable(metricas), f, indent=2, ensure_ascii=False)
print(f'   -> Guardado: output/prueba_metricas.json')

print('\n[6] Resumen de pruebas')
pasadas = (df_report['estado'] == 'PASS').sum()
advertencias = (df_report['estado'] == 'WARN').sum()
fallidas = (df_report['estado'] == 'FAIL').sum()
print(f'   Pruebas: {len(df_report)} | PASS: {pasadas} | WARN: {advertencias} | FAIL: {fallidas}')

print('\n' + '=' * 70)
print('     PRUEBAS COMPLETADAS EXITOSAMENTE')
print(f'     Gráficas y reportes guardados en: {OUTPUT}')
print('=' * 70)
