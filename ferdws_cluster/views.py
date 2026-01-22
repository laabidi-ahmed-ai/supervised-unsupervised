from django.shortcuts import render
from .forms import ClusteringForm
import joblib
import numpy as np
import os
import unicodedata
from django.conf import settings

# Chemins vers les modèles
model_path = os.path.join(settings.BASE_DIR, "ferdws_cluster", "models_ml", "kmeans_model.pkl")
features_path = os.path.join(settings.BASE_DIR,"ferdws_cluster", "models_ml", "kmeans_features.pkl")

# Charger le modèle et les features
kmeans = joblib.load(model_path)
features = joblib.load(features_path)

def clustering_view(request):
    cluster_result = None
    if request.method == "POST":
        form = ClusteringForm(request.POST)
        if form.is_valid():
            # Récupérer les valeurs dans le bon ordre, mapping feature names to form fields
            values = []
            for feat in features:
                val = form.cleaned_data.get(feat)
                if val is None and isinstance(feat, str):
                    # Normalize feature name: remove accents, replace non-alnum with underscore
                    norm = unicodedata.normalize('NFKD', feat)
                    norm = norm.encode('ascii', 'ignore').decode()
                    norm = ''.join(c if c.isalnum() else '_' for c in norm)
                    while '__' in norm:
                        norm = norm.replace('__', '_')
                    norm = norm.strip('_')
                    val = form.cleaned_data.get(norm)
                    if val is None:
                        alt = feat.replace(' ', '_')
                        val = form.cleaned_data.get(alt)
                if val is None:
                    val = 0.0
                values.append(val)
            data = np.array(values).reshape(1, -1)
            cluster_result = kmeans.predict(data)[0]
    else:
        form = ClusteringForm()
    
    return render(request, "ferdws_cluster/form.html", {"form": form, "result": cluster_result})

