from django.shortcuts import render
import joblib
import numpy as np
import os
import pickle
import pandas as pd
from django.conf import settings

# Create your views here.

def temporal_trajectory_clustering(request):
    """
    View for temporal trajectory clustering prediction.
    Loads the trained model and predicts the academy category based on user input.
    """
    context = {}
    
    if request.method == 'POST':
        try:
            # Get form data
            taux_reussite = float(request.POST.get('taux_reussite', 0)) / 100  # Convert % to decimal
            taux_presence = float(request.POST.get('taux_presence', 0)) / 100
            taux_tb = float(request.POST.get('taux_tb', 0)) / 100
            taux_b = float(request.POST.get('taux_b', 0)) / 100
            taux_ab = float(request.POST.get('taux_ab', 0)) / 100
            taux_sans_mention = float(request.POST.get('taux_sans_mention', 0)) / 100
            taux_1er_groupe = float(request.POST.get('taux_1er_groupe', 0)) / 100
            inscrits = float(request.POST.get('inscrits', 0))
            
            # Store form data for re-display
            context['form_data'] = {
                'taux_reussite': request.POST.get('taux_reussite', ''),
                'taux_presence': request.POST.get('taux_presence', ''),
                'taux_tb': request.POST.get('taux_tb', ''),
                'taux_b': request.POST.get('taux_b', ''),
                'taux_ab': request.POST.get('taux_ab', ''),
                'taux_sans_mention': request.POST.get('taux_sans_mention', ''),
                'taux_1er_groupe': request.POST.get('taux_1er_groupe', ''),
                'inscrits': request.POST.get('inscrits', ''),
            }
            
            # Load the model
            model_path = os.path.join(settings.BASE_DIR, 'temporal_trajectory_clustering_model.joblib')
            model_package = joblib.load(model_path)
            
            kmeans_model = model_package['kmeans_model']
            pca = model_package['pca']
            scaler = model_package['scaler']
            feature_columns = model_package['feature_columns']
            cluster_names = model_package['cluster_names']
            cluster_descriptions = model_package['cluster_descriptions']
            
            # Create a simplified feature vector based on input
            # Since we need trajectory features but only have single-point data,
            # we'll create a reasonable approximation
            
            # The model expects 20 features - we'll create them from the input
            feature_values = {
                'Taux_reussite_mean': taux_reussite,
                'Taux_reussite_r_squared': 0.5,  # Assume moderate fit
                'Taux_presence_r_squared': 0.5,
                'Taux_TB_cv': 0.1,  # Low coefficient of variation
                'Taux_TB_r_squared': 0.5,
                'Taux_TB_rel_growth': 0.0,  # Assume stable
                'Taux_B_mean': taux_b,
                'Taux_B_r_squared': 0.5,
                'Taux_B_rel_growth': 0.0,
                'Taux_AB_r_squared': 0.5,
                'Taux_sans_mention_r_squared': 0.5,
                'Taux_sans_mention_rel_growth': 0.0,
                'Taux_1er_groupe_r_squared': 0.5,
                'Inscrits_mean': inscrits,
                'Inscrits_std': inscrits * 0.05,  # 5% variation
                'Inscrits_r_squared': 0.5,
                'Inscrits_rel_growth': 0.02,  # 2% growth assumed
                'Inscrits_delta_std': inscrits * 0.02,
                'Inscrits_acceleration': 0.0,
                'Inscrits_max_drawdown': -inscrits * 0.01,
            }
            
            # Build feature vector in correct order
            X = np.array([[feature_values.get(col, 0) for col in feature_columns]])
            
            # Apply scaling and PCA
            X_scaled = scaler.transform(X)
            X_pca = pca.transform(X_scaled)
            
            # Predict cluster
            cluster_id = kmeans_model.predict(X_pca)[0]
            
            # Get prediction results
            context['prediction'] = cluster_names[cluster_id]
            context['description'] = cluster_descriptions[cluster_id]
            context['cluster_id'] = cluster_id
            
        except FileNotFoundError:
            context['error'] = 'Model file not found. Please ensure the model has been trained and saved.'
        except Exception as e:
            context['error'] = f'An error occurred during prediction: {str(e)}'
    
    return render(request, 'temporal_trajectory_clustering.html', context)


def school_ranking_by_score(request):
    """
    View for school score prediction.
    Loads the trained regression model and predicts the score based on school characteristics.
    """
    context = {}
    
    if request.method == 'POST':
        try:
            # Get form data
            latitude = float(request.POST.get('latitude', 0))
            longitude = float(request.POST.get('longitude', 0))
            statut_public_prive = request.POST.get('statut_public_prive', '')
            type_contrat_prive = request.POST.get('type_contrat_prive', '')
            libelle_nature = request.POST.get('libelle_nature', '')
            etat = request.POST.get('etat', '')
            libelle_region = request.POST.get('libelle_region', '')
            libelle_academie = request.POST.get('libelle_academie', '')
            libelle_departement = request.POST.get('libelle_departement', '')
            ecole_elementaire = request.POST.get('ecole_elementaire', '0')
            ecole_maternelle = request.POST.get('ecole_maternelle', '0')
            restauration = request.POST.get('restauration', '0')
            
            # Store form data for re-display
            context['form_data'] = {
                'latitude': request.POST.get('latitude', ''),
                'longitude': request.POST.get('longitude', ''),
                'statut_public_prive': statut_public_prive,
                'type_contrat_prive': type_contrat_prive,
                'libelle_nature': libelle_nature,
                'etat': etat,
                'libelle_region': libelle_region,
                'libelle_academie': libelle_academie,
                'libelle_departement': libelle_departement,
                'ecole_elementaire': ecole_elementaire,
                'ecole_maternelle': ecole_maternelle,
                'restauration': restauration,
            }
            
            # Load the model and preprocessing artifacts
            model_path = os.path.join(settings.BASE_DIR, 'score_prediction_model.pkl')
            prep_path = os.path.join(settings.BASE_DIR, 'score_prediction_preprocessing.pkl')
            
            with open(model_path, 'rb') as f:
                model_artifacts = pickle.load(f)
            
            with open(prep_path, 'rb') as f:
                prep_artifacts = pickle.load(f)
            
            model = model_artifacts['model']
            feature_names = model_artifacts['feature_names']
            label_encoders = prep_artifacts['label_encoders']
            scaler = prep_artifacts['scaler']
            
            # Create input dataframe with raw values
            input_data = {
                'latitude': latitude,
                'longitude': longitude,
                'statut_public_prive': statut_public_prive,
                'type_contrat_prive': type_contrat_prive,
                'libelle_nature': libelle_nature,
                'etat': etat,
                'libelle_region': libelle_region,
                'libelle_academie': libelle_academie,
                'libelle_departement': libelle_departement,
                'ecole_elementaire': ecole_elementaire,
                'ecole_maternelle': ecole_maternelle,
                'restauration': restauration,
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Apply label encoding for categorical columns
            for col in label_encoders.keys():
                if col in input_df.columns:
                    le = label_encoders[col]
                    try:
                        input_df[col] = le.transform(input_df[col].astype(str))
                    except ValueError:
                        # Handle unseen categories - use most frequent class (0)
                        input_df[col] = 0
            
            # Ensure all required features are present in correct order
            for feat in feature_names:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            
            input_df = input_df[feature_names]
            
            # Apply scaling
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            predicted_score = model.predict(input_scaled)[0]
            
            # Calculate percentile (approximate based on typical score distribution)
            # Scores typically range from 0 to 500, with mean around 250
            min_score = 0
            max_score = 500
            percentile = min(100, max(0, (predicted_score - min_score) / (max_score - min_score) * 100))
            percentile_top = 100 - percentile
            
            # Generate message based on percentile
            if percentile >= 80:
                percentile_message = f"Excellent! Your school score of {predicted_score:.1f} puts it among the top performers in France."
            elif percentile >= 60:
                percentile_message = f"Good performance! Your school score of {predicted_score:.1f} is above average."
            elif percentile >= 40:
                percentile_message = f"Your school score of {predicted_score:.1f} is around the national average."
            else:
                percentile_message = f"Your school score of {predicted_score:.1f} suggests room for improvement."
            
            context['prediction'] = predicted_score
            context['percentile'] = percentile
            context['percentile_top'] = max(1, int(percentile_top))
            context['percentile_message'] = percentile_message
            
        except FileNotFoundError:
            context['error'] = 'Model file not found. Please ensure the model has been trained and saved.'
        except Exception as e:
            context['error'] = f'An error occurred during prediction: {str(e)}'
    
    return render(request, 'school_ranking_by_score.html', context)
