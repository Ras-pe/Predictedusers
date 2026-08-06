"""
==============================================================================
 DESERIALIZADOR Y WRAPPER DEL MODELO KKBOX CHURN
==============================================================================
Carga output/rf_churn_model.pkl (RandomForestClassifier), valida su
estructura y expone una API sencilla de predicción con alineación
automática de características por nombre. Reutilizable desde cualquier
script o la aplicación web.
==============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'rf_churn_model.pkl'
)


class ModeloChurn:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.modelo = None
        self.features = []
        self.clases = []
        self.cargar()

    def cargar(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f'No se encontró el modelo serializado en: {self.model_path}'
            )
        self.modelo = joblib.load(self.model_path)
        if not hasattr(self.modelo, 'feature_names_in_'):
            raise ValueError(
                'El modelo no expone feature_names_in_. No es compatible '
                'con esta versión de scikit-learn.'
            )
        self.features = list(self.modelo.feature_names_in_)
        self.clases = list(self.modelo.classes_)
        return self

    def alinear(self, X):
        if isinstance(X, pd.DataFrame):
            faltantes = [f for f in self.features if f not in X.columns]
            if faltantes:
                raise ValueError(f'Faltan características en el DataFrame: {faltantes}')
            extra = [c for c in X.columns if c not in self.features]
            if extra:
                X = X.drop(columns=extra)
            return X[self.features]

        if isinstance(X, dict):
            faltantes = [f for f in self.features if f not in X]
            if faltantes:
                raise ValueError(f'Faltan características en el dict: {faltantes}')
            return pd.DataFrame([X], columns=self.features)

        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.modelo.n_features_in_:
            raise ValueError(
                f'Se esperaban {self.modelo.n_features_in_} características '
                f'pero se recibieron {arr.shape[1]}.'
            )
        return arr

    def predecir(self, X):
        return np.asarray(self.modelo.predict(self.alinear(X)))

    def predecir_proba(self, X):
        return np.asarray(self.modelo.predict_proba(self.alinear(X)))

    def probabilidad_churn(self, X):
        return self.predecir_proba(X)[:, 1]

    def resumen(self):
        return {
            'tipo': type(self.modelo).__name__,
            'n_features': int(self.modelo.n_features_in_),
            'n_estimadores': int(getattr(self.modelo, 'n_estimators', -1)),
            'max_depth': getattr(self.modelo, 'max_depth', None),
            'criterion': getattr(self.modelo, 'criterion', None),
            'class_weight': getattr(self.modelo, 'class_weight', None),
            'random_state': int(getattr(self.modelo, 'random_state', None) or 0),
            'clases': [int(c) for c in self.clases],
            'features': list(self.features),
        }


def deserializar(model_path=MODEL_PATH):
    return ModeloChurn(model_path)


if __name__ == '__main__':
    m = deserializar()
    res = m.resumen()
    print('Modelo deserializado correctamente:')
    for k, v in res.items():
        if k == 'features':
            print(f'  {k}: {len(v)} -> {v[:5]} ...')
        else:
            print(f'  {k}: {v}')
