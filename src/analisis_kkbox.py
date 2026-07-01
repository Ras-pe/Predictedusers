"""
================================================================================
 ANÁLISIS COMPLETO: KKBOX Churn Prediction
================================================================================
Objetivo: Predecir la variable binaria is_churn (cancelación de servicio).
Dataset: https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge

Archivos utilizados (priorizando versiones _v2 / _v3):
  - train_v2.csv.csv       : Variable objetivo (is_churn) por msno
  - members_v3.csv.csv     : Datos demográficos de miembros
  - transactions_v2.csv.csv: Historial de transacciones
  - user_logs_v2.csv.csv   : Registros de actividad de usuario

Flujo:
  1) Carga y limpieza (merge, imputación, estandarización de fechas)
  2) Análisis Exploratorio (EDA, visualizaciones, bivariado)
  3) Identificación de variables importantes (Random Forest,
     Permutation Importance, SelectKBest ANOVA)
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import io

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, classification_report)
import joblib

# ─── Configuración global ────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 120,
    'figure.figsize': (12, 6),
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

SEED = 42
np.random.seed(SEED)

print('=' * 70)
print('     KKBOX CHURN PREDICTION — ANÁLISIS COMPLETO')
print('=' * 70)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PILAR 1: CARGA, MERGE Y LIMPIEZA                                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print('\n[1] Cargando archivos fuente...\n')

# ─── 1a) Train (variable objetivo) ───────────────────────────────────────────
train = pd.read_csv(os.path.join(BASE, 'train_v2.csv.csv'))
print(f'train_v2       : {train.shape[0]:>8,} filas, {train.shape[1]} columnas')

# ─── 1b) Members (demografía) ────────────────────────────────────────────────
members = pd.read_csv(os.path.join(BASE, 'members_v3.csv.csv'))
print(f'members_v3     : {members.shape[0]:>8,} filas, {members.shape[1]} columnas')

# ─── 1c) Transactions (historial de pagos) ───────────────────────────────────
tx = pd.read_csv(os.path.join(BASE, 'transactions_v2.csv.csv'))
print(f'transactions_v2: {tx.shape[0]:>8,} filas, {tx.shape[1]} columnas')

for col in ['transaction_date', 'membership_expire_date']:
    tx[col] = pd.to_datetime(tx[col].astype(str), format='%Y%m%d', errors='coerce')

# Agregaciones por usuario en transacciones
tx_agg = tx.groupby('msno').agg(
    txn_count=('payment_method_id', 'count'),
    payment_method_mode=('payment_method_id', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
    payment_plan_days_mean=('payment_plan_days', 'mean'),
    payment_plan_days_max=('payment_plan_days', 'max'),
    plan_list_price_mean=('plan_list_price', 'mean'),
    plan_list_price_max=('plan_list_price', 'max'),
    actual_amount_paid_mean=('actual_amount_paid', 'mean'),
    actual_amount_paid_sum=('actual_amount_paid', 'sum'),
    actual_amount_paid_max=('actual_amount_paid', 'max'),
    is_auto_renew_rate=('is_auto_renew', 'mean'),
    is_cancel_rate=('is_cancel', 'mean'),
    is_cancel_sum=('is_cancel', 'sum'),
    last_transaction_date=('transaction_date', 'max'),
    last_membership_expire_date=('membership_expire_date', 'max'),
).reset_index()

REF_DATE = pd.Timestamp('20170430')
tx_agg['days_since_last_txn'] = (REF_DATE - tx_agg['last_transaction_date']).dt.days
tx_agg['days_since_last_expire'] = (REF_DATE - tx_agg['last_membership_expire_date']).dt.days
tx_agg['membership_duration_days'] = (
    tx_agg['last_membership_expire_date'] - tx_agg['last_transaction_date']
).dt.days
tx_agg.drop(columns=['last_transaction_date', 'last_membership_expire_date'], inplace=True)

print(f'transactions agg: {tx_agg.shape[0]:>8,} filas, {tx_agg.shape[1]} columnas')

del tx
gc.collect()

# ─── 1d) User Logs (actividad de escucha) ────────────────────────────────────
user_logs = pd.read_csv(os.path.join(BASE, 'user_logs_v2.csv.csv'))
print(f'user_logs_v2   : {user_logs.shape[0]:>8,} filas, {user_logs.shape[1]} columnas')

user_logs['date'] = pd.to_datetime(user_logs['date'].astype(str), format='%Y%m%d', errors='coerce')

log_agg = user_logs.groupby('msno').agg(
    log_count=('date', 'count'),
    num_25_mean=('num_25', 'mean'),
    num_25_sum=('num_25', 'sum'),
    num_50_mean=('num_50', 'mean'),
    num_50_sum=('num_50', 'sum'),
    num_75_mean=('num_75', 'mean'),
    num_75_sum=('num_75', 'sum'),
    num_985_mean=('num_985', 'mean'),
    num_985_sum=('num_985', 'sum'),
    num_100_mean=('num_100', 'mean'),
    num_100_sum=('num_100', 'sum'),
    num_unq_mean=('num_unq', 'mean'),
    num_unq_sum=('num_unq', 'sum'),
    total_secs_mean=('total_secs', 'mean'),
    total_secs_sum=('total_secs', 'sum'),
    total_secs_max=('total_secs', 'max'),
    total_secs_std=('total_secs', 'std'),
    first_log_date=('date', 'min'),
    last_log_date=('date', 'max'),
).reset_index()

log_agg['days_since_last_log'] = (REF_DATE - log_agg['last_log_date']).dt.days
log_agg['days_between_first_last_log'] = (
    log_agg['last_log_date'] - log_agg['first_log_date']
).dt.days
log_agg.drop(columns=['first_log_date', 'last_log_date'], inplace=True)

print(f'user_logs agg  : {log_agg.shape[0]:>8,} filas, {log_agg.shape[1]} columnas')

del user_logs
gc.collect()

# ─── 1e) MERGE FINAL ─────────────────────────────────────────────────────────
print('\n[2] Realizando merge de todos los datasets...')
df = train.merge(members, on='msno', how='left')
df = df.merge(tx_agg, on='msno', how='left')
df = df.merge(log_agg, on='msno', how='left')
del train, members, tx_agg, log_agg
gc.collect()

print(f'Dataset final  : {df.shape[0]:>8,} filas, {df.shape[1]} columnas')

# ─── 1f) Ingeniería de fechas en members ─────────────────────────────────────
df['registration_init_time'] = pd.to_datetime(
    df['registration_init_time'].astype(str), format='%Y%m%d', errors='coerce'
)
df['days_since_registration'] = (REF_DATE - df['registration_init_time']).dt.days
df.drop(columns=['registration_init_time'], inplace=True)

# ─── 1g) Tratamiento de la variable 'bd' (edad) ──────────────────────────────
df.loc[df['bd'] <= 0, 'bd'] = np.nan
df.loc[df['bd'] > 100, 'bd'] = np.nan
df['age'] = 2017 - df['bd']
df.drop(columns=['bd'], inplace=True)

# ─── 1h) Análisis y gráfico de valores nulos ─────────────────────────────────
print('\n[3] Análisis de valores nulos...')
null_pct = df.isnull().mean().sort_values(ascending=False) * 100
null_report = null_pct[null_pct > 0].reset_index()
null_report.columns = ['Columna', '% Nulos']
print(null_report.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, max(4, len(null_report) * 0.3)))
barras = ax.barh(null_report['Columna'], null_report['% Nulos'], color='coral', edgecolor='darkred')
ax.set_xlabel('Porcentaje de valores nulos (%)')
ax.set_title('Porcentaje de valores nulos por columna (post-merge)', fontweight='bold')
for bar, pct in zip(barras, null_report['% Nulos']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'nulos_por_columna.png'))
plt.close()
print('   -> Gráfico guardado: output/nulos_por_columna.png')

# ─── 1i) Eliminación de columnas con > 50% de nulos ──────────────────────────
high_null_cols = null_pct[null_pct > 50].index.tolist()
if high_null_cols:
    print(f'\n   Eliminando columnas con >50% nulos: {high_null_cols}')
    df.drop(columns=high_null_cols, inplace=True)
else:
    print('\n   No hay columnas con >50% de nulos.')

# ─── 1j) Imputación de valores faltantes ─────────────────────────────────────
print('\n[4] Imputación de valores faltantes...')
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for col in num_cols:
    if col == 'is_churn':
        continue
    if df[col].isnull().any():
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f'   {col}: imputada con mediana = {med:.4f}')

for col in cat_cols:
    if col == 'msno':
        continue
    if df[col].isnull().any():
        mode_vals = df[col].mode()
        if not mode_vals.empty:
            df[col] = df[col].fillna(mode_vals.iloc[0])
            print(f'   {col}: imputada con moda = {mode_vals.iloc[0]}')
        else:
            df[col] = df[col].fillna('Unknown')
            print(f'   {col}: imputada con Unknown')

print(f'\n   Shape después de limpieza: {df.shape}')

# ─── 1k) Verificación final de nulos ─────────────────────────────────────────
remaining_nulls_by_col = df.isnull().sum()
still_null_cols = remaining_nulls_by_col[remaining_nulls_by_col > 0]
if not still_null_cols.empty:
    print(f'\n   Columnas con nulos remanentes:')
    for col, cnt in still_null_cols.items():
        print(f'      {col}: {cnt} ({cnt/len(df)*100:.2f}%)')
    for col in still_null_cols.index:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 'Unknown')
print(f'   Valores nulos remanentes totales: {df.isnull().sum().sum()}')


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PILAR 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print('\n' + '=' * 70)
print('     PILAR 2: ANÁLISIS EXPLORATORIO DE DATOS')
print('=' * 70)

# ─── 2a) Estadísticas descriptivas generales ─────────────────────────────────
print('\n[5] Estadísticas descriptivas...')
print('\n--- df.info() ---')
buf = io.StringIO()
df.info(buf=buf)
print(buf.getvalue())

print('\n--- df.describe(include="all") ---')
desc = df.describe(include='all').T
print(desc.to_string())
desc.to_csv(os.path.join(OUTPUT, 'describe_all.csv'))
print('\n   -> Guardado: output/describe_all.csv')

# ─── 2b) Histogramas de variables numéricas clave ────────────────────────────
print('\n[6] Generando histogramas y boxplots...')

key_numeric = ['age', 'days_since_registration', 'txn_count',
               'actual_amount_paid_mean', 'actual_amount_paid_sum',
               'total_secs_mean', 'total_secs_sum', 'log_count',
               'days_since_last_txn', 'num_unq_mean', 'num_100_sum',
               'payment_plan_days_mean']
key_numeric = [c for c in key_numeric if c in df.columns]

fig, axes = plt.subplots(nrows=len(key_numeric), ncols=2,
                         figsize=(14, 3.5 * len(key_numeric)))
if len(key_numeric) == 1:
    axes = axes.reshape(1, 2)

for i, col in enumerate(key_numeric):
    ax_hist = axes[i, 0]
    data = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
    ax_hist.hist(data, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax_hist.set_title(f'Histograma: {col}', fontweight='bold')
    ax_hist.set_xlabel(col)
    ax_hist.set_ylabel('Frecuencia')

    ax_box = axes[i, 1]
    df.boxplot(column=[col], ax=ax_box, vert=False)
    ax_box.set_title(f'Boxplot: {col}', fontweight='bold')
    ax_box.set_xlabel(col)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'histogramas_boxplots.png'))
plt.close()
print('   -> Guardado: output/histogramas_boxplots.png')

# ─── 2c) Variables categóricas ───────────────────────────────────────────────
print('\n[7] Gráficos de barras para variables categóricas...')

categorical_plot = ['city', 'gender', 'registered_via', 'payment_method_mode',
                    'is_auto_renew_rate']
categorical_plot = [c for c in categorical_plot if c in df.columns]

for col in categorical_plot:
    fig, ax = plt.subplots(figsize=(10, 4))
    if col == 'gender':
        counts = df[col].value_counts().reindex([1.0, 2.0], fill_value=0)
        labels = ['Male', 'Female']
    elif col == 'is_auto_renew_rate':
        df['auto_renew_bin'] = pd.cut(df[col], bins=[-0.01, 0.33, 0.66, 1.01],
                                       labels=['Baja (0-33%)', 'Media (33-66%)', 'Alta (66-100%)'])
        counts = df['auto_renew_bin'].value_counts().sort_index()
        labels = counts.index.tolist()
        counts.plot(kind='bar', ax=ax, color='teal', edgecolor='black')
        ax.set_title(f'Distribución: {col}', fontweight='bold')
        ax.set_ylabel('Frecuencia')
        ax.set_xticklabels(labels, rotation=25)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f'bar_{col}.png'))
        plt.close()
        df.drop(columns=['auto_renew_bin'], inplace=True)
        continue
    else:
        counts = df[col].value_counts().head(15)
        labels = counts.index.tolist()

    counts.plot(kind='bar', ax=ax, color='teal', edgecolor='black')
    ax.set_title(f'Distribución: {col}', fontweight='bold')
    ax.set_ylabel('Frecuencia')
    ax.set_xticklabels(labels, rotation=45)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, f'bar_{col}.png'))
    plt.close()

print('   -> Gráficos guardados en output/')

# ─── 2d) Matriz de correlación (Heatmap) ─────────────────────────────────────
print('\n[8] Matriz de correlación (heatmap)...')

corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(corr, mask=mask, annot=False, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.3, cbar_kws={'shrink': 0.7})
ax.set_title('Matriz de correlación (variables numéricas)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'heatmap_correlacion.png'))
plt.close()
print('   -> Guardado: output/heatmap_correlacion.png')

# ─── 2e) Análisis bivariado con is_churn ─────────────────────────────────────
print('\n[9] Análisis bivariado con is_churn...')

for col in ['city', 'gender', 'registered_via', 'payment_method_mode']:
    if col not in df.columns:
        continue
    churn_rate = df.groupby(col)['is_churn'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    churn_rate.head(15).plot(kind='bar', ax=ax, color='crimson', edgecolor='black')
    ax.set_title(f'Tasa de Churn por {col}', fontweight='bold')
    ax.set_ylabel('Tasa de Churn')
    ax.set_xlabel(col)
    ax.set_xticklabels(churn_rate.head(15).index, rotation=45)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}',
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, f'churn_rate_by_{col}.png'))
    plt.close()

for col in ['age', 'txn_count', 'actual_amount_paid_mean', 'total_secs_mean',
            'days_since_last_txn', 'days_since_last_log', 'log_count']:
    if col not in df.columns:
        continue
    fig, ax = plt.subplots(figsize=(10, 4))
    valid = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.nunique() <= 1:
        continue
    q = valid.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    if q.nunique() < 3:
        continue
    try:
        df['bin'] = pd.qcut(valid, q=10, labels=False, duplicates='drop')
        churn_bin = df.groupby('bin')['is_churn'].mean()
        churn_bin.plot(kind='line', marker='o', ax=ax, color='darkorange', linewidth=2)
        ax.set_title(f'Tasa de Churn por decil de {col}', fontweight='bold')
        ax.set_ylabel('Tasa de Churn')
        ax.set_xlabel(f'Decil de {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f'churn_rate_by_{col}_binned.png'))
        plt.close()
    except Exception:
        pass
    finally:
        if 'bin' in df.columns:
            df.drop(columns=['bin'], inplace=True)

print('   -> Gráficos bivariados guardados en output/')


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PILAR 3: IDENTIFICACIÓN DE VARIABLES IMPORTANTES                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print('\n' + '=' * 70)
print('     PILAR 3: IDENTIFICACIÓN DE VARIABLES IMPORTANTES')
print('=' * 70)

# ─── 3a) Preprocesamiento ────────────────────────────────────────────────────
print('\n[10] Preprocesamiento para ML...')

if 'is_churn' not in df.columns:
    raise ValueError('La columna is_churn no está presente en el dataset final.')

y = df['is_churn'].values
X = df.drop(columns=['is_churn', 'msno'], errors='ignore')

feature_names_num = X.select_dtypes(include=[np.number]).columns.tolist()
feature_names_cat = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f'   Variables numéricas: {len(feature_names_num)}')
print(f'   Variables categóricas: {len(feature_names_cat)}')

if feature_names_cat:
    X = pd.get_dummies(X, columns=feature_names_cat, drop_first=True, dummy_na=False)
    X.columns = X.columns.astype(str).str.replace(r'[^\w]', '_', regex=True)

print(f'   Total features después de encoding: {X.shape[1]}')

X.replace([np.inf, -np.inf], np.nan, inplace=True)
if X.isnull().any().any():
    X.fillna(X.median(numeric_only=True), inplace=True)

# ─── 3b) Train/Test split ────────────────────────────────────────────────────
print('\n[11] Dividiendo en train/test...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f'   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}')

# ─── 3c) Random Forest con Validación Cruzada ────────────────────────────────
print('\n[12] Entrenando Random Forest con validación cruzada...')

sample_size = min(100_000, X_train.shape[0])
if X_train.shape[0] > sample_size:
    print(f'   Usando muestra aleatoria de {sample_size:,} filas.')
    idx_sample = np.random.choice(X_train.shape[0], size=sample_size, replace=False)
    X_train_samp = X_train.iloc[idx_sample]
    y_train_samp = y_train[idx_sample]
else:
    X_train_samp = X_train
    y_train_samp = y_train

rf = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_leaf=10,
    class_weight='balanced', random_state=SEED, n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_auc = cross_val_score(rf, X_train_samp, y_train_samp, cv=cv, scoring='roc_auc')
cv_f1 = cross_val_score(rf, X_train_samp, y_train_samp, cv=cv, scoring='f1')

print(f'   CV AUC-ROC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}')
print(f'   CV F1:      {cv_f1.mean():.4f} ± {cv_f1.std():.4f}')

rf.fit(X_train_samp, y_train_samp)

# ─── 3c.1) Evaluación en test con métricas adecuadas ────────────────────────
print('\n[12.1] Evaluación en test...')
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print(f'   AUC-ROC:   {roc_auc_score(y_test, y_proba):.4f}')
print(f'   F1-Score:  {f1_score(y_test, y_pred):.4f}')
print(f'   Precision: {precision_score(y_test, y_pred):.4f}')
print(f'   Recall:    {recall_score(y_test, y_pred):.4f}')
print(f'   Accuracy:  {rf.score(X_test, y_test):.4f}')
print(f'\n   Classification Report:')
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# ─── 3c.2) Feature Importances ──────────────────────────────────────────────
print('\n[12.2] Feature Importances (Random Forest)...')
fi = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(30)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi)))
ax.barh(fi['feature'][::-1], fi['importance'][::-1], color=colors[::-1], edgecolor='black')
ax.set_xlabel('Importancia (Gini)')
ax.set_title('Top 30 Feature Importances — Random Forest (balanced)', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'rf_feature_importances.png'))
plt.close()
print('   -> Guardado: output/rf_feature_importances.png')
fi.to_csv(os.path.join(OUTPUT, 'rf_feature_importances.csv'), index=False)

# ─── 3d) Permutation Importance ──────────────────────────────────────────────
print('\n[13] Calculando Permutation Importance...')

top_n = min(30, X_test.shape[1])
fi_top = fi.head(top_n)['feature'].tolist()
X_test_perm = X_test[fi_top]
X_train_perm = X_train_samp[fi_top]

rf_perm = RandomForestClassifier(
    n_estimators=50, max_depth=10, min_samples_leaf=10,
    random_state=SEED, n_jobs=-1
)
rf_perm.fit(X_train_perm, y_train_samp)

perm_result = permutation_importance(
    rf_perm, X_test_perm, y_test,
    n_repeats=10, random_state=SEED, n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature': fi_top,
    'importance_mean': perm_result.importances_mean,
    'importance_std': perm_result.importances_std
}).sort_values('importance_mean', ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = range(len(perm_df))
ax.barh(y_pos, perm_df['importance_mean'].values[::-1],
        xerr=perm_df['importance_std'].values[::-1],
        color='limegreen', edgecolor='darkgreen', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(perm_df['feature'].values[::-1])
ax.set_xlabel('Caída de rendimiento (Accuracy) al permutar')
ax.set_title('Permutation Importance (top 30 features)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'permutation_importance.png'))
plt.close()
print('   -> Guardado: output/permutation_importance.png')
perm_df.to_csv(os.path.join(OUTPUT, 'permutation_importance.csv'), index=False)

# ─── 3e) SelectKBest (ANOVA F-test) ──────────────────────────────────────────
print('\n[14] SelectKBest con ANOVA F-test...')

skb_sample = min(100_000, X_train.shape[0])
idx_skb = np.random.choice(X_train.shape[0], size=skb_sample, replace=False)
X_skb = X_train.iloc[idx_skb]
y_skb = y_train[idx_skb]

skb = SelectKBest(f_classif, k='all')
skb.fit(X_skb, y_skb)

skb_df = pd.DataFrame({
    'feature': X.columns,
    'f_score': skb.scores_,
    'p_value': skb.pvalues_
}).sort_values('f_score', ascending=False).head(30)

skb_df['p_value'] = skb_df['p_value'].clip(lower=1e-300)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(skb_df)))
ax.barh(skb_df['feature'][::-1], skb_df['f_score'][::-1], color=colors[::-1], edgecolor='black')
ax.set_xlabel('F-score (ANOVA)')
ax.set_title('Top 30 — SelectKBest ANOVA F-test', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'selectkbest_anova.png'))
plt.close()
print('   -> Guardado: output/selectkbest_anova.png')
skb_df.to_csv(os.path.join(OUTPUT, 'selectkbest_anova.csv'), index=False)
# ─── 3f) Guardar modelo entrenado ────────────────────────────────────────────
print('\n[15] Guardando modelo para predicción...')
model_path = os.path.join(OUTPUT, 'rf_churn_model.pkl')
joblib.dump(rf, model_path)
print(f'   -> Modelo guardado: {model_path}')

# ─── 3g) Resumen comparativo de métodos ──────────────────────────────────────

print('\n[16] Generando resumen comparativo...')

summary = fi.head(20)[['feature', 'importance']].rename(
    columns={'importance': 'RF_importance'}
)
summary = summary.merge(
    perm_df.head(20)[['feature', 'importance_mean']].rename(
        columns={'importance_mean': 'Perm_importance'}
    ), on='feature', how='outer'
)
summary = summary.merge(
    skb_df.head(20)[['feature', 'f_score']], on='feature', how='outer'
)
summary = summary.fillna(0)
summary.to_csv(os.path.join(OUTPUT, 'resumen_feature_selection.csv'), index=False)
print('   -> Guardado: output/resumen_feature_selection.csv')


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  ADVERTENCIAS SOBRE BUENAS PRÁCTICAS                                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print('\n' + '=' * 70)
print('     ADVERTENCIAS Y BUENAS PRÁCTICAS')
print('=' * 70)
print("""
[W1] MULTICOLINEALIDAD:
  - Varias variables agregadas (sum, mean, max, std) de una misma fuente
    (ej. num_25_mean, num_25_sum) pueden estar altamente correlacionadas.
  - La multicolinealidad afecta la interpretación de coeficientes en modelos
    lineales (regresión logística, LDA), pero NO afecta a Random Forest o
    Gradient Boosting, que son no-paramétricos y robustos ante correlaciones.
  - Recomendación: si se usan modelos lineales en etapa posterior, aplicar
    VIF (Variance Inflation Factor) y eliminar variables con VIF > 10.

[W2] VALIDACIÓN CRUZADA:
  - Las importancias reportadas aquí se basan en un solo train/test split.
  - Para una selección robusta de características, se recomienda validación
    cruzada estratificada (StratifiedKFold con k=5 o k=10) y promediar las
    importancias a través de los folds.
  - Esto reduce la varianza de la estimación y evita overfitting al split.

[W3] DESBALANCE DE CLASES:
  - Si is_churn está desbalanceado (lo habitual en churn), considerar usar
    class_weight='balanced' en RandomForest o técnicas de remuestreo (SMOTE).

[W4] DRIFT TEMPORAL:
  - Los datos contienen marcas de tiempo (transactions, logs). Es crucial
    respetar el orden temporal al hacer train/test split (TimeSeriesSplit)
    para evitar data leakage del futuro al pasado.

[W5] MEMORIA:
  - user_logs_v2 (~18M filas) se agregó por msno para reducir dimensionalidad.
  - Si persisten problemas de memoria, reducir aún más el muestreo o usar
    librerías out-of-core como Dask o Vaex.
""")

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  EJEMPLO DE PREDICCIÓN CON NUEVOS USUARIOS                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print('\n' + '=' * 70)
print('     EJEMPLO DE PREDICCIÓN')
print('=' * 70)
print("""
  Para predecir con nuevos datos:

    import joblib
    import pandas as pd

    modelo = joblib.load('output/rf_churn_model.pkl')

    # Preparar features de nuevo usuario (mismas columnas que X)
    nuevo_usuario = pd.DataFrame([{
        'days_since_last_txn': 60,
        'membership_duration_days': 30,
        'actual_amount_paid_sum': 149.0,
        ...
    }])

    proba = modelo.predict_proba(nuevo_usuario)[:, 1]
    pred = modelo.predict(nuevo_usuario)
    print(f'Probabilidad de churn: {proba[0]:.2%}')
""")

print('=' * 70)
print('     ANÁLISIS COMPLETADO EXITOSAMENTE')
print(f'     Todos los archivos fueron guardados en: {OUTPUT}')
print(f'     Modelo guardado: rf_churn_model.pkl')
print('=' * 70)
