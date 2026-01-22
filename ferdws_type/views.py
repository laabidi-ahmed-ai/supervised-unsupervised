import os
import joblib
from django.shortcuts import render
from .forms import PredictionForm
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ferdws_type/model/final_model (2).pkl')

model = joblib.load(MODEL_PATH)

FEATURES_ORDER = [
    'score',
    'ULIS',
    'SEGPA',
    'Hebergement',
    'Restauration',
    'code_nature',
    'code_region',
    'libelle_region',
    'code_departement',
    'libelle_departement',
    'longitude',
    'latitude',
    'statut_public_prive_Public',
    'etat_OUVERT'
]

DEFAULTS = {
    'score': 0,
    'ULIS': 0,
    'SEGPA': 0,
    'Hebergement': 0,
    'Restauration': 0,
    'code_nature': 0,
    'code_region': 0,
    'libelle_region': 0,
    'code_departement': 0,
    'libelle_departement': 0,
    'longitude': 0.0,
    'latitude': 0.0,
    'statut_public_prive_Public': 0,
    'etat_OUVERT': 0
}

def predict_view(request):
    prediction = None
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            # remplir toutes les colonnes manquantes
            for col in FEATURES_ORDER:
                if col not in data:
                    data[col] = DEFAULTS[col]

            # convertir les bool en int
            data['statut_public_prive_Public'] = int(data['statut_public_prive_Public'])
            data['etat_OUVERT'] = int(data['etat_OUVERT'])

            # créer DataFrame avec l'ordre exact des colonnes
            df = pd.DataFrame([data], columns=FEATURES_ORDER)

            # prédiction : on passe directement le DataFrame, pas .values
            prediction = model.predict(df)[0]

    else:
        form = PredictionForm()

    return render(request, 'ferdws_type/predict.html', {
        'form': form,
        'prediction': prediction
    })
