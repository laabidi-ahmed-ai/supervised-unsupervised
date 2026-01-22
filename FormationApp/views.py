import os
import joblib
import numpy as np
from django.conf import settings
from django.shortcuts import render

# Fichier .joblib exporté depuis ton Notebook
MODEL_PATH = os.path.join(settings.BASE_DIR, "FormationApp", "ml_models", "kmeans_archetypes.joblib")

# Champs bruts demandés au user
RAW_FIELDS = [
    ("nb_inscrits", "Number of enrolled students"),
    ("nb_presents", "Number of attendees"),
    ("nb_admis_totaux", "Number of admitted students (total)"),
    ("nb_refuses_totaux", "Number of failed students (total)"),
    ("nb_admis_mention_TB", "Admitted with honors: Highest (TB)"),
    ("nb_admis_mention_B", "Admitted with honors: Good (B)"),
    ("nb_admis_mention_AB", "Admitted with honors: Fairly Good (AB)"),
    ("nb_admis_sans_mention", "Admitted without honors"),
    ("nb_admis_mention_TB_felicitations", "Highest honors (TB) with congratulations"),
]


# Taux calculés
FEATURES = [
    "taux_presence", "taux_reussite", "taux_echec",
    "taux_TB", "taux_B", "taux_AB", "taux_sans_mention", "taux_TB_felicitations"
]

PRED_META = {
    0: {
        "emoji": "🛠️",
        "icon": "bi-tools",
        "tone": "warning",
        "title": "Assidues mais en difficulté",
        "advice": "Renforcer le soutien (tutorat, rattrapage) + suivi des échecs.",
        "image": "images/clusters/support.png",
    },
    1: {
        "emoji": "🔥",
        "icon": "bi-graph-up-arrow",
        "tone": "success",
        "title": "Très performantes",
        "advice": "Maintenir la dynamique + booster les mentions (AB/B/TB).",
        "image": "images/clusters/performance.png",
    },
    2: {
        "emoji": "⚠️",
        "icon": "bi-exclamation-triangle-fill",
        "tone": "danger",
        "title": "Décrochage / Fragiles",
        "advice": "Alerte: plan d’action immédiat (présence, remédiation, encadrement).",
        "image": "images/clusters/alert.png",
    },
    3: {
        "emoji": "📈",
        "icon": "bi-bar-chart-line-fill",
        "tone": "info",
        "title": "Performantes mais peu de mentions",
        "advice": "Améliorer l’excellence (préparation TB/B) + valoriser les meilleurs.",
        "image": "images/clusters/boost.png",
    },
    4: {
        "emoji": "🏆",
        "icon": "bi-trophy-fill",
        "tone": "elite",
        "title": "Excellence / Elite",
        "advice": "Conserver les standards + partager les bonnes pratiques.",
        "image": "images/clusters/elite.png",
    },
}


def _to_int(val):
    if val is None:
        return None
    s = str(val).strip().replace(",", ".")
    if s == "":
        return None
    # autoriser "100.0" etc.
    return int(float(s))

def formation_predict(request):
    # préparer champs affichage
    fields = [{"name": k, "label": label, "value": "", "error": ""} for k, label in RAW_FIELDS]

    calculated = None   # dict des taux à afficher
    prediction = None   # dict résultat final

    # charger modèle une seule fois (tu peux aussi le mettre global)
    if not os.path.exists(MODEL_PATH):
        return render(request, "formation_predict.html", {
            "fields": fields,
            "error_global": f"❌ Modèle introuvable: {MODEL_PATH}"
        })

    bundle = joblib.load(MODEL_PATH)
    scaler = bundle["scaler"]
    kmeans = bundle["kmeans"]
    labels_map = bundle.get("labels", {})  # dict cluster->label

    if request.method == "POST":
        action = request.POST.get("action", "calc")  # "calc" ou "predict"

        # lire inputs
        raw = {}
        has_error = False

        for f in fields:
            name = f["name"]
            f["value"] = request.POST.get(name, "")
            try:
                raw[name] = _to_int(f["value"])
                if raw[name] is None:
                    f["error"] = "Champ obligatoire"
                    has_error = True
            except Exception:
                f["error"] = "Nombre invalide"
                has_error = True

        # validations métier
        if not has_error:
            if raw["nb_inscrits"] <= 0:
                _set_error(fields, "nb_inscrits", "Doit être > 0")
                has_error = True

            if raw["nb_presents"] <= 0:
                _set_error(fields, "nb_presents", "Doit être > 0")
                has_error = True

            if raw["nb_presents"] > raw["nb_inscrits"]:
                _set_error(fields, "nb_presents", "Ne peut pas dépasser nb_inscrits")
                has_error = True

            # Mentions + sans mention <= nb_admis_totaux (optionnel mais logique)
            sum_mentions = (
                raw["nb_admis_mention_TB"] + raw["nb_admis_mention_B"] +
                raw["nb_admis_mention_AB"] + raw["nb_admis_sans_mention"]
            )
            if sum_mentions > raw["nb_admis_totaux"]:
                _set_error(fields, "nb_admis_totaux", "Doit être ≥ somme des admis par mention")
                has_error = True

            if raw["nb_admis_mention_TB_felicitations"] > raw["nb_admis_mention_TB"]:
                _set_error(fields, "nb_admis_mention_TB_felicitations", "Doit être ≤ admis TB")
                has_error = True

            # refusés + admis <= présents (optionnel)
            if raw["nb_admis_totaux"] + raw["nb_refuses_totaux"] > raw["nb_presents"]:
                _set_error(fields, "nb_refuses_totaux", "Admis + Refusés ne doit pas dépasser nb_presents")
                has_error = True

        # calcul taux (si pas d’erreur)
        if not has_error:
            nb_p = raw["nb_presents"]
            calculated = {
                "taux_presence": raw["nb_presents"] / raw["nb_inscrits"],
                "taux_reussite": raw["nb_admis_totaux"] / nb_p,
                "taux_echec": raw["nb_refuses_totaux"] / nb_p,
                "taux_TB": raw["nb_admis_mention_TB"] / nb_p,
                "taux_B": raw["nb_admis_mention_B"] / nb_p,
                "taux_AB": raw["nb_admis_mention_AB"] / nb_p,
                "taux_sans_mention": raw["nb_admis_sans_mention"] / nb_p,
                "taux_TB_felicitations": raw["nb_admis_mention_TB_felicitations"] / nb_p,
            }

            # si action = predict => prédiction
            # si action = predict => prédiction
            if action == "predict":
                X = np.array([[calculated[c] for c in FEATURES]], dtype=float)
                Xs = scaler.transform(X)

                cluster = int(kmeans.predict(Xs)[0])
                
                meta = PRED_META.get(cluster, {
                    "emoji": "✅",
                    "icon": "bi-check-circle-fill",
                    "tone": "success",
                    "title": labels_map.get(cluster, f"Cluster {cluster}"),
                    "advice": "Analyse disponible.",
                    "image": "images/clusters/default.png",
                })
                prediction = {
                    "cluster": cluster,
                    "label": labels_map.get(cluster, meta["title"]),
                    **meta
                }



    return render(request, "formation_predict.html", {
        "fields": fields,
        "calculated": calculated,
        "prediction": prediction,
    })

def _set_error(fields, name, msg):
    for f in fields:
        if f["name"] == name:
            f["error"] = msg
            return