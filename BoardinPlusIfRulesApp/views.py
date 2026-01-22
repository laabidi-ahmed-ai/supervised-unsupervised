from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import pandas as pd
import numpy as np
from .utils import (
    load_boarding_model,
    load_boarding_features,
    load_association_rules,
    load_clustering_info,
    load_clustering_model
)

# Cache des modèles (chargés une fois au démarrage)
_boarding_model = None
_boarding_features = None
_clustering_model = None
_association_rules = None
_clustering_info = None

def get_boarding_model():
    """Lazy loading du modèle boarding"""
    global _boarding_model, _boarding_features
    if _boarding_model is None:
        try:
            _boarding_model = load_boarding_model()
            features_data = load_boarding_features()
            _boarding_features = features_data.get('feature_names', [])
        except Exception as e:
            print(f"Error loading boarding model: {e}")
            return None, []
    return _boarding_model, _boarding_features

def get_clustering_data():
    """Lazy loading des données clustering"""
    global _clustering_model, _association_rules, _clustering_info
    if _clustering_model is None:
        try:
            _clustering_model = load_clustering_model()
            _association_rules = load_association_rules()
            _clustering_info = load_clustering_info()
        except Exception as e:
            print(f"Error loading clustering data: {e}")
            return None, [], {}
    return _clustering_model, _association_rules, _clustering_info

def boarding_prediction_view(request):
    """Page principale pour la prédiction boarding"""
    model, features = get_boarding_model()
    
    # Dictionnaire de traduction des labels (garder les noms techniques comme ULIS, SEGPA, etc.)
    feature_labels = {
        'Average_Score_moyen': 'Average Score',
        'ULIS': 'ULIS',
        'SEGPA': 'SEGPA',
        'Section_internationale': 'International Section',
        'Restauration': 'Catering',
        'Type_etablissement': 'School Type',
        'GRETA': 'GRETA',
        'PIAL': 'PIAL',
        'Ecole_maternelle': 'Nursery School',
        'Ecole_elementaire': 'Elementary School',
        'Lycee_agricole': 'Agricultural High School',
        'Lycee_militaire': 'Military High School',
        'Lycee_des_metiers': 'Vocational High School',
        'etat': 'Status',
        'code_nature': 'Nature Code',
        'libelle_nature': 'Nature Label',
        'Section_arts': 'Arts Section',
        'Section_cinema': 'Cinema Section',
        'Section_theatre': 'Theater Section',
        'Section_sport': 'Sports Section',
        'Section_europeenne': 'European Section',
        'Code_region': 'Region Code',
        'Statut_public_prive': 'Public/Private Status',
        'Type_contrat_prive': 'Private Contract Type',
        'ministere_tutelle': 'Supervising Ministry',
        'Code_type_contrat_prive': 'Private Contract Type Code',
        'Voie_generale': 'General Track',
        'Voie_technologique': 'Technological Track',
        'Voie_professionnelle': 'Professional Track',
    }
    
    # Préparer les features avec leurs types (pour l'affichage)
    feature_info = []
    for feature in features:
        # Déterminer le type d'input basé sur le nom
        if any(x in feature.lower() for x in ['score', 'moyen', 'code']):
            input_type = 'number'
            step = '0.01'
        else:
            input_type = 'number'
            step = '1'
        feature_info.append({
            'name': feature,
            'label': feature_labels.get(feature, feature.replace('_', ' ').title()),
            'type': input_type,
            'step': step
        })
    
    context = {
        'features': feature_info,
        'model_loaded': model is not None
    }
    return render(request, 'BoardinPlusIfRulesApp/boarding_prediction.html', context)

@require_http_methods(["POST"])
def predict_boarding_api(request):
    """API pour prédire boarding à partir des features"""
    try:
        model, features = get_boarding_model()
        
        if model is None:
            return JsonResponse({
                'success': False,
                'error': 'Model not loaded'
            }, status=500)
        
        # Récupérer les données du formulaire
        data = {}
        for feature in features:
            value = request.POST.get(feature, '0')
            try:
                data[feature] = float(value)
            except (ValueError, TypeError):
                data[feature] = 0.0
        
        # Créer un DataFrame avec les features dans le bon ordre
        df = pd.DataFrame([data])
        df = df[features]  # S'assurer que l'ordre est correct
        
        # Faire la prédiction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        return JsonResponse({
            'success': True,
            'prediction': int(prediction),
            'probability_yes': float(probability[1]),
            'probability_no': float(probability[0]),
            'prediction_text': 'Boarding Available' if prediction == 1 else 'No Boarding'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

def association_rules_view(request):
    """Page principale pour le clustering et les règles d'association"""
    model, rules, info = get_clustering_data()
    
    # Préparer les features de clustering pour le formulaire
    clustering_features = info.get('clustering_features', []) if info else []
    
    # Features que l'utilisateur peut entrer (exclure les mentions qui sont le résultat et Average_Score_moyen)
    input_features = []
    feature_translations = {
        'Sexe_Male': {'label': 'Gender', 'type': 'select', 'options': [('0', 'Female'), ('1', 'Male')]},
        'Voie_generale': {'label': 'General Track', 'type': 'checkbox'},
        'Voie_technologique': {'label': 'Technological Track', 'type': 'checkbox'},
        'Voie_professionnelle': {'label': 'Professional Track', 'type': 'checkbox'},
        'Hebergement': {'label': 'Boarding', 'type': 'checkbox'},
        'Restauration': {'label': 'Catering', 'type': 'checkbox'},
    }
    
    for feature in clustering_features:
        if feature not in ['Reussite', 'Mention_TB', 'Mention_B', 'Mention_AB', 'Sans_Mention', 'Average_Score_moyen']:
            if feature in feature_translations:
                input_features.append({
                    'name': feature,
                    **feature_translations[feature]
                })
            else:
                input_features.append({
                    'name': feature,
                    'label': feature.replace('_', ' ').title(),
                    'type': 'number'
                })
    
    context = {
        'input_features': input_features,
        'clustering_info': info if info else {},
        'model_loaded': model is not None
    }
    return render(request, 'BoardinPlusIfRulesApp/association_rules.html', context)

@require_http_methods(["POST"])
def predict_cluster_api(request):
    """API pour prédire le cluster d'un candidat"""
    try:
        model, rules, info = get_clustering_data()
        
        if model is None:
            return JsonResponse({
                'success': False,
                'error': 'Clustering model not loaded'
            }, status=500)
        
        clustering_features = info.get('clustering_features', []) if info else []
        
        # Récupérer les données du formulaire
        data = {}
        for feature in clustering_features:
            if feature in ['Reussite', 'Mention_TB', 'Mention_B', 'Mention_AB', 'Sans_Mention']:
                # Ces features sont le résultat, pas l'input
                data[feature] = 0.0
            elif feature == 'Average_Score_moyen':
                # Average Score n'est plus dans le formulaire, mettre 0 par défaut
                data[feature] = 0.0
            else:
                value = request.POST.get(feature, '0')
                try:
                    data[feature] = float(value)
                except (ValueError, TypeError):
                    data[feature] = 0.0
        
        # S'assurer qu'une seule voie est sélectionnée
        voies = ['Voie_generale', 'Voie_technologique', 'Voie_professionnelle']
        voie_selected = sum([data.get(v, 0) for v in voies])
        if voie_selected == 0:
            # Par défaut, mettre Voie_generale
            data['Voie_generale'] = 1.0
        
        # Créer un DataFrame avec les features dans le bon ordre
        df = pd.DataFrame([data])
        df = df[clustering_features]  # S'assurer que l'ordre est correct
        
        # Utiliser le pipeline de clustering
        scaler = model.get('scaler')
        dbscan = model.get('dbscan')
        
        if scaler is None or dbscan is None:
            return JsonResponse({
                'success': False,
                'error': 'Invalid clustering model structure'
            }, status=500)
        
        # Standardiser les données
        X_scaled = scaler.transform(df)
        
        # DBSCAN ne peut pas prédire directement un nouveau point
        # On va utiliser une approche de distance pour trouver le cluster le plus proche
        # ou indiquer que c'est un outlier potentiel
        # Pour simplifier, on va dire que le clustering a été fait sur les données d'entraînement
        # et on se concentre sur les règles d'association qui matchent
        cluster_label = -2  # -2 signifie "analyse basée sur les règles d'association"
        
        # Trouver les règles d'association pertinentes
        # On accepte les règles qui ont au moins une caractéristique d'entrée qui matche
        # et qui ont des résultats dans les consequents
        scored_rules = []
        input_features_list = ['Sexe_Male', 'Sexe_Female', 'Voie_generale', 'Voie_technologique', 
                              'Voie_professionnelle', 'Hebergement', 'Restauration', 'Section_internationale']
        result_features_list = ['Mention_TB', 'Mention_B', 'Mention_AB', 'Sans_Mention', 'Reussite', 
                               'Mention_TB_Felicitations']
        
        if rules:
            for rule in rules:
                antecedents = rule.get('antecedents', [])
                consequents = rule.get('consequents', [])
                
                # Vérifier que les consequents contiennent au moins un résultat (mention, réussite)
                has_results = False
                for cons in consequents:
                    if cons in result_features_list:
                        has_results = True
                        break
                
                if not has_results:
                    continue
                
                # Extraire les inputs des antecedents (ignorer les mentions/résultats)
                input_antecedents = [ant for ant in antecedents if ant in input_features_list]
                
                # Si aucune caractéristique d'entrée dans les antecedents, on ignore
                if len(input_antecedents) == 0:
                    continue
                
                # Vérifier combien d'inputs matchent avec les données entrées
                score = 0
                total_input_ants = len(input_antecedents)
                
                for ant in input_antecedents:
                    if ant in data:
                        if data[ant] == 1 or (ant == 'Sexe_Male' and data.get('Sexe_Male', 0) == 1):
                            score += 1
                    elif ant == 'Sexe_Female':
                        if data.get('Sexe_Male', 0) == 0:
                            score += 1
                    elif ant == 'Sexe_Male':
                        if data.get('Sexe_Male', 0) == 1:
                            score += 1
                
                # Si au moins un input matche, on garde la règle
                if score > 0:
                    # Score = (nombre de matches / nombre total d'inputs) * confidence * bonus pour plusieurs caractéristiques
                    match_ratio = score / total_input_ants
                    # Bonus pour les règles avec plusieurs caractéristiques (au moins 2)
                    multi_feature_bonus = 1.5 if total_input_ants >= 2 else 1.0
                    relevance_score = match_ratio * rule.get('confidence', 0) * multi_feature_bonus
                    scored_rules.append({
                        'rule': rule,
                        'score': relevance_score,
                        'matches': score,
                        'total_inputs': total_input_ants,
                        'match_ratio': match_ratio,
                        'input_antecedents': input_antecedents
                    })
            
            # Récupérer toutes les caractéristiques d'entrée remplies par l'utilisateur
            user_inputs = []
            if data.get('Sexe_Male', 0) == 1:
                user_inputs.append('Sexe_Male')
            elif 'Sexe_Male' in data and data.get('Sexe_Male', 0) == 0:
                user_inputs.append('Sexe_Female')
            
            for voie in ['Voie_generale', 'Voie_technologique', 'Voie_professionnelle']:
                if data.get(voie, 0) == 1:
                    user_inputs.append(voie)
                    break
            
            if data.get('Restauration', 0) == 1:
                user_inputs.append('Restauration')
            if data.get('Hebergement', 0) == 1:
                user_inputs.append('Hebergement')
            
            # Trier par nombre de caractéristiques d'abord (plus = mieux), puis par score
            scored_rules.sort(key=lambda x: (x['matches'], x['score']), reverse=True)
            
            # NE GARDER QUE les règles avec au moins 2 caractéristiques d'entrée qui matchent
            multi_feature_rules = [sr for sr in scored_rules if sr['matches'] >= 2]
            
            # Si on a des règles avec 2+ caractéristiques, les utiliser
            if len(multi_feature_rules) > 0:
                matching_rules = multi_feature_rules[:10]
            # Sinon, créer des règles combinées avec toutes les caractéristiques de l'utilisateur
            elif len(user_inputs) >= 2:
                # Prendre les meilleures règles avec 1 caractéristique et les combiner
                single_feature_rules = [sr for sr in scored_rules if sr['matches'] == 1][:5]
                matching_rules = []
                
                # Pour chaque règle avec 1 caractéristique, créer une version combinée
                for sr in single_feature_rules:
                    rule = sr['rule']
                    consequents = rule.get('consequents', [])
                    result_consequents = [c for c in consequents if c in result_features_list]
                    
                    if result_consequents:
                        # Créer une règle combinée avec toutes les caractéristiques de l'utilisateur
                        combined_rule = {
                            'rule': {
                                'antecedents': user_inputs,
                                'consequents': result_consequents,
                                'confidence': rule.get('confidence', 0) * 0.8,  # Réduire légèrement la confiance
                                'support': rule.get('support', 0),
                                'lift': rule.get('lift', 0)
                            },
                            'score': sr['score'],
                            'matches': len(user_inputs),
                            'total_inputs': len(user_inputs),
                            'match_ratio': 1.0,
                            'input_antecedents': user_inputs,
                            'is_combined': True
                        }
                        matching_rules.append(combined_rule)
                
                # Limiter à 5 règles combinées
                matching_rules = matching_rules[:5]
            else:
                matching_rules = []
        
        # Translate features to English
        def translate_feature(feature):
            translations = {
                'Sexe_Male': 'Male',
                'Sexe_Female': 'Female',
                'Voie_generale': 'General Track',
                'Voie_technologique': 'Technological Track',
                'Voie_professionnelle': 'Professional Track',
                'Mention_TB': 'Very Good Mention',
                'Mention_B': 'Good Mention',
                'Mention_AB': 'Fairly Good Mention',
                'Sans_Mention': 'No Mention',
                'Reussite': 'Success',
                'Hebergement': 'Boarding',
                'Restauration': 'Catering',
                'Section_internationale': 'International Section',
            }
            return translations.get(feature, feature.replace('_', ' ').title())
        
        # Formater les règles pour l'affichage
        # On affiche seulement les inputs dans le "Si" et les résultats dans le "Alors"
        formatted_rules = []
        for i, scored_rule in enumerate(matching_rules):
            rule = scored_rule['rule']
            input_antecedents = scored_rule.get('input_antecedents', [])
            consequents = rule.get('consequents', [])
            
            # Filtrer les consequents pour ne garder que les résultats
            result_consequents = [c for c in consequents if c in result_features_list]
            
            # Si c'est une règle combinée, utiliser toutes les caractéristiques de l'utilisateur
            if scored_rule.get('is_combined', False):
                # Les input_antecedents contiennent déjà toutes les caractéristiques de l'utilisateur
                pass
            elif len(input_antecedents) < 2:
                # Si moins de 2 caractéristiques, essayer d'ajouter d'autres caractéristiques remplies
                user_inputs = []
                if data.get('Sexe_Male', 0) == 1:
                    user_inputs.append('Sexe_Male')
                elif 'Sexe_Male' in data and data.get('Sexe_Male', 0) == 0:
                    user_inputs.append('Sexe_Female')
                
                for voie in ['Voie_generale', 'Voie_technologique', 'Voie_professionnelle']:
                    if data.get(voie, 0) == 1:
                        user_inputs.append(voie)
                        break
                
                if data.get('Restauration', 0) == 1:
                    user_inputs.append('Restauration')
                if data.get('Hebergement', 0) == 1:
                    user_inputs.append('Hebergement')
                
                # Combiner avec les caractéristiques de la règle originale
                combined_inputs = list(set(input_antecedents + user_inputs))
                if len(combined_inputs) >= 2:
                    input_antecedents = combined_inputs[:5]  # Limiter à 5 caractéristiques max
            
            # Traduire
            antecedents_translated = [translate_feature(a) for a in input_antecedents]
            consequents_translated = [translate_feature(c) for c in result_consequents]
            
            # Note si c'est un match partiel
            is_partial = scored_rule.get('match_ratio', 1.0) < 1.0
            
            formatted_rules.append({
                'antecedents': antecedents_translated,
                'consequents': consequents_translated,
                'if_text': " + ".join(antecedents_translated),
                'then_text': " + ".join(consequents_translated),
                'confidence': rule.get('confidence', 0),
                'support': rule.get('support', 0),
                'lift': rule.get('lift', 0),
                'is_partial_match': is_partial,
                'match_ratio': scored_rule.get('match_ratio', 1.0),
                'is_combined': scored_rule.get('is_combined', False)
            })
        
        # Déterminer si c'est un profil typique ou unique basé sur les règles trouvées
        is_unique_profile = len(formatted_rules) == 0
        
        # Check which fields were filled
        filled_fields = []
        if data.get('Sexe_Male', 0) == 1:
            filled_fields.append('Gender (Male)')
        elif data.get('Sexe_Male', 0) == 0 and 'Sexe_Male' in data:
            filled_fields.append('Gender (Female)')
        
        voies = ['Voie_generale', 'Voie_technologique', 'Voie_professionnelle']
        for voie in voies:
            if data.get(voie, 0) == 1:
                filled_fields.append(translate_feature(voie))
                break
        
        if data.get('Restauration', 0) == 1:
            filled_fields.append('Catering')
        if data.get('Hebergement', 0) == 1:
            filled_fields.append('Boarding')
        
        return JsonResponse({
            'success': True,
            'cluster': 'N/A' if cluster_label == -2 else int(cluster_label),
            'is_outlier': is_unique_profile,
            'matching_rules': formatted_rules,
            'total_rules_found': len(formatted_rules),
            'analysis_type': 'association_rules',
            'filled_fields': filled_fields,
            'needs_more_info': len(formatted_rules) == 0 and (len(filled_fields) < 2 or not any('Voie' in f for f in filled_fields))
        })
    except Exception as e:
        import traceback
        return JsonResponse({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=400)