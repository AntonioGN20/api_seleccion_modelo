from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

# Esta función realizará el particionado completo del conjunto de datos
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

# Esta función eliminará las etiquetas del dataframe
def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

# Cargar el conjunto de datos
df = pd.read_csv('TotalFeatures-ISCXFlowMeter.csv')

# Dividir el conjunto de datos
train_set, val_set, test_set = train_val_test_split(df)

X_train, y_train = remove_labels(train_set, 'calss')
X_val, y_val = remove_labels(val_set, 'calss')
X_test, y_test = remove_labels(test_set, 'calss')

# Entrenar el clasificador RandomForest
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_depth': randint(low=8, high=50),
    }

rnd_clf = RandomForestClassifier(n_jobs=-1)

rnd_search = RandomizedSearchCV(rnd_clf, param_distributions=param_distribs,
                                n_iter=5, cv=2, scoring='f1_weighted')

rnd_search.fit(X_train, y_train)

# Seleccionar el mejor modelo
clf_rnd = rnd_search.best_estimator_

# Definir la vista para la API
@csrf_exempt
def f1_score_api(request):
    if request.method == 'GET':
        # Calcular F1 Score en el conjunto de validación
        y_val_pred = clf_rnd.predict(X_val)
        f1_val_score = f1_score(y_val_pred, y_val, average='weighted')
        # Obtener los hiperparámetros del modelo
        max_depth = clf_rnd.get_params()['max_depth']
        n_estimators = clf_rnd.get_params()['n_estimators']
        # Obtener los resultados de la búsqueda
        cvres = rnd_search.cv_results_
        # Convertir matrices NumPy a listas para evitar el error de serialización
        cvres = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in cvres.items()}
        # Obtener los mejores hiperparámetros del modelo final
        best_params = rnd_search.best_params_
        # Devolver los resultados como un JSON
        return JsonResponse({
            'f1_score_validation': f1_val_score,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'cv_results': cvres,
            'best_params': best_params
        })
