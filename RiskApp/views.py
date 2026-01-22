import joblib
import numpy as np
import os
from django.shortcuts import render
from django.conf import settings

def predict_sport_risk(request):
    """
    Predicts if a department is at risk of having too few sports sections 
    based on population density and total school count.
    """
    context = {}
    
    if request.method == 'POST':
        try:
            # 1. Get input from the user
            # In your training, X was [PopDensite, libelle_nature]
            pop_densite = float(request.POST.get('PopDensite', 0))
            total_schools = float(request.POST.get('total_schools', 0)) # This was 'libelle_nature' count
            
            # Store for re-display in form
            context['form_data'] = {
                'PopDensite': pop_densite,
                'total_schools': total_schools,
            }
            
            # 2. Define file paths (Ensure these files are in your main Django folder)
            model_path = os.path.join(settings.BASE_DIR, 'risk_model_sport.pkl')
            scaler_path = os.path.join(settings.BASE_DIR, 'scaler_sport.joblib')
            
            # 3. Load the model and scaler
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError("Model or Scaler file not found in the project directory.")
                
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # 4. Prepare the feature vector in the EXACT order of training
            # The order MUST be: [PopDensite, Total_Schools]
            X_input = np.array([[pop_densite, total_schools]])
            
            # 5. Apply Scaling (NEVER fit here, only transform)
            X_scaled = scaler.transform(X_input)
            
            # 6. Make Prediction
            prediction = model.predict(X_scaled)[0] # Returns 0 or 1
            
            # 7. Get Probabilities (Optional, only if using KNN or SVM with probability=True)
            try:
                probs = model.predict_proba(X_scaled)[0]
                # If prediction is 1 (At Risk), confidence is the prob of class 1
                confidence = probs[1] if prediction == 1 else probs[0]
                context['confidence'] = f"{confidence * 100:.1f}%"
            except:
                context['confidence'] = "N/A"

            # 8. Set context for the template
            context['is_at_risk'] = True if prediction == 1 else False
            context['prediction_label'] = "AT RISK" if prediction == 1 else "SAFE"
            context['alert_class'] = "danger" if prediction == 1 else "success"
            
        except FileNotFoundError as e:
            context['error'] = str(e)
        except Exception as e:
            context['error'] = f"An error occurred: {str(e)}"
            
    return render(request, 'risk.html', context)