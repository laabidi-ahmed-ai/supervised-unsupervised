from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import pickle
import joblib
import pandas as pd
import numpy as np
import os

# Create your views here.
def objective_yassin1(request):
    return render(request, 'ObjectiveYassin1.html')

# Create your views here.
def objective_yassin2(request):
    return render(request, 'ObjectiveYassin2.html')

@csrf_exempt
@require_http_methods(["POST"])
def extract_csv_data(request):
    """
    Extract sample data from CSV files based on selected model
    For ULIS, SEGPA, GRETA: uses dataset_ready.csv with 31 features
    """
    try:
        data = json.loads(request.body)
        model = data.get('model', '').lower()
        
        if model not in ['ulis', 'segpa', 'greta']:
            return JsonResponse({
                'success': False,
                'message': 'Invalid model selected'
            })
        
        # Path to the CSV file - models trained on dataset_ready.csv
        csv_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            'dataset_ready.csv'
        )
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Get a random row with all features
        random_row = df.sample(n=1).iloc[0]
        
        # Convert to dictionary and handle NaN values
        row_dict = random_row.to_dict()
        row_dict = {
            k: (None if pd.isna(v) else (float(v) if isinstance(v, (np.floating, np.integer)) else v))
            for k, v in row_dict.items()
        }
        
        return JsonResponse({
            'success': True,
            'data': row_dict,
            'message': f'Successfully extracted data for {model.upper()} model'
        })
        
    except FileNotFoundError:
        return JsonResponse({
            'success': False,
            'message': 'Dataset file not found'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error extracting data: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["POST"])
def run_prediction(request):
    """
    Run prediction using the trained models
    For ULIS, SEGPA, GRETA: uses dataset_ready.csv and scaler
    """
    try:
        data = json.loads(request.body)
        model_type = data.get('model', '').lower()
        input_data = data.get('data', {})
        
        if model_type not in ['ulis', 'segpa', 'greta']:
            return JsonResponse({
                'success': False,
                'message': 'Invalid model type'
            })
        
        # Define model file mapping - use best model files
        model_files = {
            'ulis': 'ulis_best_model.pkl',
            'segpa': 'segpa_best_model.pkl',
            'greta': 'greta_best_model.pkl'
        }
        
        model_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            model_files[model_type]
        )
        
        # Load the model using joblib (more robust for sklearn models)
        try:
            model = joblib.load(model_path)
        except:
            # Fallback to pickle with specific error handling
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f, encoding='utf-8-sig')
            except Exception as pickle_error:
                return JsonResponse({
                    'success': False,
                    'message': f'Failed to load model: {str(pickle_error)}'
                })
        
        # If model is a dictionary with multiple models, extract the best one
        if isinstance(model, dict):
            # Helper function to find actual model in nested dicts
            def find_model(obj, depth=0):
                if depth > 10:
                    return None
                
                if hasattr(obj, 'predict'):
                    return obj
                
                if isinstance(obj, dict):
                    # Try to find in dict values
                    for key, value in obj.items():
                        result = find_model(value, depth + 1)
                        if result is not None:
                            return result
                
                return None
            
            # Try to find model in each algorithm's dictionary
            preference_order = ['RandomForest', 'XGBoost', 'LightGBM', 'KNN']
            selected_model = None
            
            for preferred in preference_order:
                if preferred in model:
                    found_model = find_model(model[preferred])
                    if found_model is not None:
                        selected_model = found_model
                        break
            
            # If no preferred found, try all entries
            if selected_model is None:
                for key, value in model.items():
                    found_model = find_model(value)
                    if found_model is not None:
                        selected_model = found_model
                        break
            
            if selected_model is not None:
                model = selected_model
            else:
                # Debug: show keys of first model dict
                if 'RandomForest' in model and isinstance(model['RandomForest'], dict):
                    rf_keys = list(model['RandomForest'].keys())
                    return JsonResponse({
                        'success': False,
                        'message': f'RandomForest dict keys: {rf_keys}'
                    })
                return JsonResponse({
                    'success': False,
                    'message': 'Could not find model object in nested dictionary structure'
                })
        
        # If model is a dictionary, extract the actual model
        if isinstance(model, dict):
            # Try to use RandomForest first (often a good choice), then XGBoost, then the first available
            model_keys = list(model.keys())
            selected_model = None
            
            # Preference order for model selection
            preference_order = ['RandomForest', 'XGBoost', 'LightGBM', 'KNN']
            for preferred in preference_order:
                if preferred in model:
                    selected_model = model[preferred]
                    break
            
            # If no preferred model found, use the first one
            if selected_model is None:
                selected_model = model[model_keys[0]]
            
            # Recursive function to extract the actual model from nested structures
            def extract_predictor(obj, depth=0):
                if depth > 20:  # Prevent infinite loops
                    return obj
                
                if hasattr(obj, 'predict'):
                    return obj
                
                if isinstance(obj, dict):
                    # Try common keys first
                    for key in ['model', 'classifier', 'estimator', 'best_model']:
                        if key in obj:
                            result = extract_predictor(obj[key], depth + 1)
                            if hasattr(result, 'predict'):
                                return result
                    
                    # Try any value that might be a model
                    for key, value in obj.items():
                        if not isinstance(value, (int, float, str, bool, list)):
                            result = extract_predictor(value, depth + 1)
                            if hasattr(result, 'predict'):
                                return result
                
                return obj
            
            model = extract_predictor(selected_model)
        
        # Validate model was extracted properly
        if not hasattr(model, 'predict'):
            return JsonResponse({
                'success': False,
                'message': f'Could not extract valid model. Type: {type(model)}'
            })
        
        # Load dataset to get the correct feature order (all 31 features)
        csv_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            'dataset_ready.csv'
        )
        
        df = pd.read_csv(csv_path)
        feature_columns = df.columns.tolist()
        
        # Remove target feature for each model
        target_features = {
            'ulis': 'ulis',
            'segpa': 'segpa',
            'greta': 'greta'
        }
        
        target_col = target_features[model_type]
        if target_col in feature_columns:
            feature_columns.remove(target_col)
        
        # Remove specific columns for SEGPA model
        if model_type == 'segpa':
            cols_to_remove = ['ecole_maternelle', 'ecole_elementaire', 'etat', 'ministere_tutelle', 'restauration',
                             'lycee_des_metiers', 'lycee_militaire', 'section_cinema', 'greta', 'hebergement',
                             'section_theatre', 'code_region', 'libelle_region', 'lycee_agricole', 'section_internationale']
            feature_columns = [col for col in feature_columns if col not in cols_to_remove]
        
        # Remove specific columns for GRETA model
        if model_type == 'greta':
            cols_to_remove = ['ecole_elementaire', 'ecole_maternelle', 'etat', 'type_contrat_prive', 'code_type_contrat_prive']
            feature_columns = [col for col in feature_columns if col not in cols_to_remove]
        
        # Create input dataframe with all features except target and excluded columns
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Keep only the required columns in correct order
        input_df = input_df[feature_columns]
        
        # Handle missing values
        input_df = input_df.fillna(0)
        
        # Make prediction without scaler (models trained directly on dataset_ready)
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Get confidence scores
        confidence = {}
        if hasattr(model, 'classes_'):
            for idx, class_label in enumerate(model.classes_):
                confidence[str(class_label)] = float(prediction_proba[idx])
        
        return JsonResponse({
            'success': True,
            'prediction': str(prediction),
            'confidence': confidence,
            'message': 'Prediction completed successfully'
        })
        
    except FileNotFoundError as e:
        return JsonResponse({
            'success': False,
            'message': f'Model file not found: {str(e)}'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error running prediction: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["POST"])
def extract_csv_data_yassin2(request):
    """
    Extract sample data from df_encoded.csv for ObjectiveYassin2
    Uses 5 specific features with geographic coordinates
    """
    try:
        # Required features for best_model
        required_features = [
            "longitude_x",
            "latitude_x",
            "taux_reussite",
            "taux_mentions",
            "Nombre d'inscrits"
        ]
        
        # Path to the encoded CSV file
        csv_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            'df_encoded.csv'
        )
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Get a random row with required features
        random_row = df[required_features].sample(n=1).iloc[0]
        
        # Convert to dictionary and handle NaN values
        row_dict = random_row.to_dict()
        row_dict = {
            k: (None if pd.isna(v) else (float(v) if isinstance(v, (np.floating, np.integer)) else v))
            for k, v in row_dict.items()
        }
        
        return JsonResponse({
            'success': True,
            'data': row_dict,
            'message': 'Successfully extracted data for Geographic Analysis'
        })
        
    except FileNotFoundError:
        return JsonResponse({
            'success': False,
            'message': 'Dataset file not found'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error extracting data: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["POST"])
def run_prediction_yassin2(request):
    """
    Run prediction using best_model.pkl with scaler
    Uses best_model.json for model parameters
    Returns geographic visualization data (latitude, longitude)
    """
    try:
        data = json.loads(request.body)
        input_data = data.get('data', {})
        
        # Required features
        required_features = [
            "longitude_x",
            "latitude_x",
            "taux_reussite",
            "taux_mentions",
            "Nombre d'inscrits"
        ]
        
        # Load best_model.json for parameters
        json_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            'best_model.json'
        )
        
        with open(json_path, 'r') as f:
            model_config = json.load(f)
        
        best_model_type = model_config.get('best_model', 'HDBSCAN')
        best_params = model_config.get('best_params', {})
        
        # Load best_model.pkl
        model_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            'best_model.pkl'
        )
        
        try:
            model = joblib.load(model_path)
        except:
            with open(model_path, 'rb') as f:
                model = pickle.load(f, encoding='latin1')
        
        # Load scaler
        scaler_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            'scaler.pkl'
        )
        
        try:
            scaler = joblib.load(scaler_path)
        except:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f, encoding='latin1')
        
        # Create input dataframe with required features
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        for col in required_features:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Keep only required columns
        input_df = input_df[required_features]
        
        # Handle missing values
        input_df = input_df.fillna(0)
        
        # Validate input data - check for NaN/Inf values
        if input_df.isnull().any().any() or np.isinf(input_df.values).any():
            return JsonResponse({
                'success': False,
                'message': f'Invalid input data: Contains NaN or Inf values. Data: {input_df.to_dict()}'
            })
        
        # Scale the input data
        try:
            input_scaled = scaler.transform(input_df)
        except Exception as scale_error:
            return JsonResponse({
                'success': False,
                'message': f'Scaling error: {str(scale_error)}. Input data: {input_df.to_dict()}'
            })
        
        # Check if scaled data contains NaN or Inf
        if pd.isnull(input_scaled).any() or np.isinf(input_scaled).any():
            return JsonResponse({
                'success': False,
                'message': f'Scaled data contains NaN/Inf. Original: {input_df.to_dict()}, Scaled: {input_scaled}'
            })
        
        # Check if model is HDBSCAN (unsupervised clustering)
        if best_model_type == 'HDBSCAN':
            # HDBSCAN model - unsupervised clustering
            # Simple approach: find closest cluster based on training data centroids
            prediction = -1.0
            num_clusters = 0
            error_msg = None
            
            try:
                print("\n" + "="*60)
                print("HDBSCAN CLUSTERING PREDICTION")
                print("="*60)
                
                if not hasattr(model, 'labels_'):
                    raise ValueError("Model doesn't have labels_ attribute")
                
                labels = model.labels_
                unique_clusters = np.unique(labels[labels != -1])  # -1 is noise
                num_clusters = len(unique_clusters)
                print(f"Found {num_clusters} clusters")
                
                if num_clusters == 0:
                    print("No valid clusters found")
                    prediction = -1.0
                else:
                    # Load training data
                    models_dir = os.path.dirname(model_path)
                    data_file = os.path.join(models_dir, 'df_encoded.csv')
                    
                    if not os.path.exists(data_file):
                        raise FileNotFoundError(f"df_encoded.csv not found at {data_file}")
                    
                    print(f"Loading training data...")
                    X_full = pd.read_csv(data_file)[required_features]
                    print(f"Full dataset shape: {X_full.shape}")
                    
                    # Use first 100,000 rows to calculate cluster centroids (original training data)
                    n_train_samples = len(labels)
                    X_train = X_full.iloc[:n_train_samples]
                    print(f"Using first {n_train_samples} rows as training data for centroids")
                    
                    # Verify labels match
                    if len(labels) != len(X_train):
                        raise ValueError(f"Labels ({len(labels)}) don't match training data ({len(X_train)})")
                    
                    # Scale training data
                    X_train_scaled = scaler.transform(X_train)
                    
                    # Calculate centroids
                    centroids = {}
                    for cid in unique_clusters:
                        mask = labels == cid
                        centroids[cid] = np.mean(X_train_scaled[mask], axis=0)
                    
                    # Find closest cluster
                    input_point = input_scaled[0]
                    distances = {cid: np.linalg.norm(input_point - cent) for cid, cent in centroids.items()}
                    
                    print(f"Distances: {distances}")
                    prediction = float(min(distances, key=distances.get))
                    print(f"Predicted cluster: {prediction}")
                        
            except Exception as e:
                error_msg = str(e)
                print(f"ERROR: {error_msg}")
                import traceback
                traceback.print_exc()
                prediction = -1.0
            
            # Build response
            confidence = {
                'model_accuracy': model_config.get('best_score', 0.0),
                'cluster_prediction': prediction,
                'number_of_clusters': num_clusters
            }
            if error_msg:
                confidence['debug_error'] = error_msg
                
        elif hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
            # Standard supervised classifier - use predict and predict_proba
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Get confidence scores
            confidence = {}
            if hasattr(model, 'classes_'):
                for idx, class_label in enumerate(model.classes_):
                    confidence[str(class_label)] = float(prediction_proba[idx])
        else:
            # Fallback for other model types
            try:
                prediction = model.predict(input_scaled)[0]
            except Exception as e:
                prediction = 0.0
            confidence = {'prediction': float(prediction), 'error': str(e)}
        
        # Extract geographic data for map
        latitude = float(input_data.get('latitude_x', 45.5))
        longitude = float(input_data.get('longitude_x', 0.98))
        
        return JsonResponse({
            'success': True,
            'prediction': float(prediction),
            'confidence': confidence,
            'latitude': latitude,
            'longitude': longitude,
            'model_type': best_model_type,
            'message': 'Prediction and geographic data generated successfully'
        })
        
    except FileNotFoundError as e:
        return JsonResponse({
            'success': False,
            'message': f'Model file not found: {str(e)}'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error running prediction: {str(e)}'
        })
    