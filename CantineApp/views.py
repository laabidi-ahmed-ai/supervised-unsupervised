import os, json
import pandas as pd
import joblib

from django.shortcuts import render
from django.conf import settings

CAT_COLS = ["type_etablissement", "statut_public_prive", "libelle_region"]

FEATURE_COLS_RAW = [
    "type_etablissement",
    "ecole_elementaire", "ecole_maternelle",
    "lycee_agricole", "lycee_militaire", "lycee_des_metiers",
    "de_ulis", "de_segpa", "de_hebergement",
    "greta", "pial",
    "APPARTENANCE_EDUCATION_PRIORITAITRE",
    "statut_public_prive",
    "libelle_region",
    "latitude", "longitude", "nb_etudiants_region"
]

_model = None
_scaler = None
_columns = None
_scaler_cols = None


def _load_artifacts():
    global _model, _scaler, _columns, _scaler_cols
    if _model is not None:
        return

    model_dir = os.path.join(settings.BASE_DIR, "CantineApp", "ml_models")
    _model = joblib.load(os.path.join(model_dir, "model.pkl"))
    _scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    # ✅ colonnes attendues par le modèle (le plus fiable)
    _columns = list(getattr(_model, "feature_names_in_", []))

    # fallback si jamais feature_names_in_ n'existe pas
    if not _columns:
        with open(os.path.join(model_dir, "columns.json"), "r", encoding="utf-8") as f:
            _columns = json.load(f)

    # sécurité: supprimer une éventuelle entrée "..." si elle existe
    _columns = [c for c in _columns if c and c != "..."]

    # ✅ colonnes exactes du scaler
    _scaler_cols = list(getattr(_scaler, "feature_names_in_", []))

    # fallback minimal si scaler n'a pas feature_names_in_
    if not _scaler_cols:
        # mets ici EXACTEMENT ce que tu as scalé dans le notebook
        _scaler_cols = ["latitude", "longitude", "nb_etudiants_region"]


def _extract_categories(prefix: str):
    # catégories visibles dans les colonnes one-hot
    return sorted([c.replace(prefix, "") for c in _columns if c.startswith(prefix)])


def _ensure_public_prive(options):
    # Afficher Public et Privé même si drop_first a supprimé une modalité
    norm = lambda s: s.lower().replace("é", "e").strip()
    has_public = any(norm(o) == "public" for o in options)
    has_prive = any(norm(o) == "prive" for o in options)

    base = []
    if not has_public:
        base.append("Public")
    if not has_prive:
        base.append("Privé")

    # dédoublonnage
    out = []
    seen = set()
    for o in base + options:
        k = norm(o)
        if k not in seen:
            seen.add(k)
            out.append(o)
    return out


def _preprocess_one(row_dict):
    # 1) df brut
    df = pd.DataFrame([row_dict], columns=FEATURE_COLS_RAW)

    # 2) one-hot (plus robuste: drop_first=False)
    X = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)

    # 3) aligner exactement sur les colonnes du modèle
    X = X.reindex(columns=_columns, fill_value=0)

    # 4) scaler sur les colonnes EXACTES du fit
    missing = [c for c in _scaler_cols if c not in X.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour scaler: {missing[:10]} ...")

    X[_scaler_cols] = _scaler.transform(X[_scaler_cols])

    return X


def cantine_predict(request):
    _load_artifacts()

    type_options = _extract_categories("type_etablissement_")
    statut_options = _ensure_public_prive(_extract_categories("statut_public_prive_"))
    region_options = _extract_categories("libelle_region_")

    context = {
        "type_options": type_options,
        "statut_options": statut_options,
        "region_options": region_options,
        "result": None,
        "proba": None,
        "form": {}
    }

    if request.method == "POST":
        def cb(name):
            return 1 if request.POST.get(name) == "on" else 0

        def to_float(name, default=0.0):
            val = request.POST.get(name, str(default))
            val = val.replace(",", ".")  # support virgule
            try:
                return float(val)
            except:
                return float(default)

        row = {
            "type_etablissement": request.POST.get("type_etablissement", "").strip(),
            "statut_public_prive": request.POST.get("statut_public_prive", "").strip(),
            "libelle_region": request.POST.get("libelle_region", "").strip(),
            "latitude": to_float("latitude"),
            "longitude": to_float("longitude"),
            "nb_etudiants_region": to_float("nb_etudiants_region"),

            "ecole_elementaire": cb("ecole_elementaire"),
            "ecole_maternelle": cb("ecole_maternelle"),
            "lycee_agricole": cb("lycee_agricole"),
            "lycee_militaire": cb("lycee_militaire"),
            "lycee_des_metiers": cb("lycee_des_metiers"),

            "de_ulis": cb("de_ulis"),
            "de_segpa": cb("de_segpa"),
            "de_hebergement": cb("de_hebergement"),
            "greta": cb("greta"),
            "pial": cb("pial"),
            "APPARTENANCE_EDUCATION_PRIORITAITRE": cb("APPARTENANCE_EDUCATION_PRIORITAITRE"),
        }

        X_ready = _preprocess_one(row)

        pred = int(_model.predict(X_ready)[0])
        context["result"] = pred

        # ✅ proba de la classe 1 correctement
        if hasattr(_model, "predict_proba"):
            proba_vec = _model.predict_proba(X_ready)[0]
            if hasattr(_model, "classes_") and 1 in list(_model.classes_):
                idx_1 = list(_model.classes_).index(1)
                context["proba"] = float(proba_vec[idx_1])
            else:
                # fallback (si classes_ inconnu)
                context["proba"] = float(proba_vec[-1])

        context["form"] = row

    return render(request, "cantine_form.html", context)
