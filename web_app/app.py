import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, '..', 'output', 'rf_churn_model.pkl')

app = Flask(__name__)

FEATURES = [
    'city', 'registered_via', 'txn_count', 'payment_method_mode',
    'payment_plan_days_mean', 'payment_plan_days_max',
    'plan_list_price_mean', 'plan_list_price_max',
    'actual_amount_paid_mean', 'actual_amount_paid_sum', 'actual_amount_paid_max',
    'is_auto_renew_rate', 'is_cancel_rate', 'is_cancel_sum',
    'days_since_last_txn', 'days_since_last_expire', 'membership_duration_days',
    'log_count',
    'num_25_mean', 'num_25_sum', 'num_50_mean', 'num_50_sum',
    'num_75_mean', 'num_75_sum', 'num_985_mean', 'num_985_sum',
    'num_100_mean', 'num_100_sum', 'num_unq_mean', 'num_unq_sum',
    'total_secs_mean', 'total_secs_sum', 'total_secs_max', 'total_secs_std',
    'days_since_last_log', 'days_between_first_last_log',
]

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

try:
    model = joblib.load(MODEL_PATH)
    print(f'Modelo cargado: {MODEL_PATH}')
except Exception as e:
    print(f'Error cargando modelo: {e}')
    model = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no disponible'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No se recibieron datos'}), 400

    row = {}
    for feat in FEATURES:
        val = data.get(feat)
        if val is None or val == '':
            val = DEFAULTS[feat]
        row[feat] = float(val)

    df = pd.DataFrame([row])[FEATURES]

    proba = model.predict_proba(df)[0, 1]
    pred = int(model.predict(df)[0])

    return jsonify({
        'prediction': pred,
        'probability': round(float(proba), 4),
        'label': 'Churn' if pred == 1 else 'No Churn',
        'risk': 'Alto' if proba >= 0.7 else ('Medio' if proba >= 0.3 else 'Bajo')
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
