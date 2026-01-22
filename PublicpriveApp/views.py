# views.py
from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
import os
import pandas as pd
import joblib
# views.py
import pandas as pd
from django.shortcuts import render
import os
import pandas as pd
from django.shortcuts import render
from django.conf import settings
import numpy as np

import io, base64
from sklearn.cluster import KMeans
from django.shortcuts import render
import matplotlib.pyplot as plt

# Load the trained KNN pipeline once
MODEL_PATH = os.path.join(settings.BASE_DIR, "PublicpriveApp", "ml", "knn_public_priv.pkl")
model_pipeline = joblib.load(MODEL_PATH)
from .forms import CandidateForm

def predict_cluster_view(request):
    csv_path = os.path.join(
        settings.BASE_DIR,
        'PublicpriveApp',
        'clustered_data_full.csv'
    )

    # Load CSV
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Fix single-column CSV
    if len(df.columns) == 1:
        df = df.iloc[:, 0].str.split(',', expand=True)
        df.columns = [
            'nb_inscrits',
            'nb_presents',
            'taux_reussite',
            'taux_mentions',
            'cluster'
        ]

    df = df.apply(pd.to_numeric, errors='coerce')

    # Compute cluster centroids
    centroids = (
        df
        .groupby('cluster')[[
            'nb_inscrits',
            'nb_presents',
            'taux_reussite',
            'taux_mentions'
        ]]
        .mean()
    )

    predicted_cluster = None

    if request.method == "POST":
        form = CandidateForm(request.POST)
        if form.is_valid():
            candidate = np.array([
                form.cleaned_data['nb_inscrits'],
                form.cleaned_data['nb_presents'],
                form.cleaned_data['taux_reussite'],
                form.cleaned_data['taux_mentions']
            ])

            # Compute distances to centroids
            distances = {}
            for cluster_id, row in centroids.iterrows():
                centroid = row.values
                distances[cluster_id] = np.linalg.norm(candidate - centroid)

            # Nearest cluster
            predicted_cluster = min(distances, key=distances.get)

    else:
        form = CandidateForm()

    return render(
        request,
        'predict_cluster.html',
        {
            'form': form,
            'predicted_cluster': predicted_cluster
        }
    )
def predict_etablissement(request):
    result = None  # Default, no result yet

    if request.method == 'POST':
        try:
            # 1️⃣ Get form data
            data = {
                'ulis': [int(request.POST['ulis'])],
                'segpa': [int(request.POST['segpa'])],
                'hebergement': [int(request.POST['hebergement'])],
                'restauration': [int(request.POST['restauration'])],
                'ecole_elementaire': [int(request.POST['ecole_elementaire'])],
                'ecole_maternelle': [int(request.POST['ecole_maternelle'])],
                'lycee_agricole': [int(request.POST['lycee_agricole'])],
                'lycee_militaire': [int(request.POST['lycee_militaire'])],
                'lycee_des_metiers': [int(request.POST['lycee_des_metiers'])],
                'code_nature': [int(request.POST['code_nature'])],
                'latitude': [float(request.POST['latitude'])],
                'longitude': [float(request.POST['longitude'])]
            }

            df = pd.DataFrame(data)

            # 2️⃣ Make prediction
            prediction = model_pipeline.predict(df)[0]
            result = "Public" if prediction == 0 else "Privé"

        except Exception as e:
            messages.error(request, f"Invalid input: {e}")

    # Render the same form page, with `result` passed to template
    return render(request, 'prediction/form.html', {'result': result})

def home(request):
    return render(request, "index.html")

def cluster_summary_view(request):
    # Load CSV
    csv_path = os.path.join(settings.BASE_DIR, 'PublicpriveApp', 'clustered_data_full.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    if len(df.columns) == 1:
        df = df.iloc[:, 0].str.split(',', expand=True)
        df.columns = ['nb_inscrits','nb_presents','taux_reussite','taux_mentions','cluster']

    df = df.apply(pd.to_numeric, errors='coerce')
    df_cluster = df[['cluster','nb_inscrits','nb_presents','taux_reussite','taux_mentions']]
    cluster_summary = df_cluster.groupby('cluster').mean().reset_index()
    clusters = cluster_summary.to_dict(orient='records')

    # Handle candidate form
    form = CandidateForm(request.POST or None)
    predicted_cluster = None

    if request.method == "POST" and form.is_valid():
        candidate = np.array([
            form.cleaned_data['nb_inscrits'],
            form.cleaned_data['nb_presents'],
            form.cleaned_data['taux_reussite'],
            form.cleaned_data['taux_mentions']
        ])
        centroids = df_cluster.groupby('cluster')[['nb_inscrits','nb_presents','taux_reussite','taux_mentions']].mean()
        distances = {cid: np.linalg.norm(candidate - row.values) for cid, row in centroids.iterrows()}
        predicted_cluster = min(distances, key=distances.get)

    return render(request, 'clusters.html', {
        'clusters': clusters,
        'form': form,
        'predicted_cluster': predicted_cluster
    })
def dashboard_view(request):
    return render(request, 'dashboard.html')
