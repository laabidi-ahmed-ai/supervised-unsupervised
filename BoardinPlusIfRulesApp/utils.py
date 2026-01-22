import os
import json
import joblib
from django.conf import settings

# Chemin vers le dossier ml_models
MODELS_DIR = os.path.join(settings.BASE_DIR, 'ml_models')

def load_boarding_model():
    """Charge le pipeline de prédiction boarding"""
    model_path = os.path.join(MODELS_DIR, 'boarding_pipeline.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def load_boarding_features():
    """Charge la liste des features nécessaires pour boarding"""
    features_path = os.path.join(MODELS_DIR, 'boarding_features.json')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    with open(features_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_clustering_model():
    """Charge le pipeline de clustering"""
    model_path = os.path.join(MODELS_DIR, 'clustering_pipeline.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Clustering model file not found: {model_path}")
    return joblib.load(model_path)

def load_association_rules():
    """Charge les règles d'association"""
    rules_path = os.path.join(MODELS_DIR, 'association_rules.json')
    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"Association rules file not found: {rules_path}")
    with open(rules_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_clustering_info():
    """Charge les informations sur le clustering"""
    info_path = os.path.join(MODELS_DIR, 'clustering_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Clustering info file not found: {info_path}")
    with open(info_path, 'r', encoding='utf-8') as f:
        return json.load(f)






