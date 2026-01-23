# Education Analytics Platform - ML Deployment

## 📁 Repository Structure

This repository contains multiple ML objectives developed by a team:

- **`main` branch** - Complete Django deployment (production-ready)
  - All Django apps integrated
  - Trained models (.pkl files)
  - Web interfaces and APIs
  
- **`boarding-prediction` branch** - Training notebook
  - `boarding1final.ipynb` - Model training, evaluation, comparison
  
- **`if-then-rules-discovery` branch** - Training notebook  
  - `unsupervised_clean.ipynb` - Clustering and association rules analysis

**For recruiters:** The `main` branch shows the deployed work. Training details are in the separate branches.

---

## My Contribution in the group project: Boarding Prediction & Association Rules Analysis

This repository contains multiple ML objectives deployed in Django. **This README focuses on my specific contribution.**

---

## My Work Overview

I developed and deployed **two ML objectives**:

1. **Boarding Availability Prediction** - Binary classification model
2. **Candidate Pattern Discovery** - Clustering + Association Rule Mining

### My Files in This Repository

**Django Application:**
- `BoardinPlusIfRulesApp/` - Complete Django app I created
  - `views.py` - API endpoints and business logic (487 lines)
  - `utils.py` - Model loading utilities
  - `urls.py` - URL routing
  - `templates/BoardinPlusIfRulesApp/` - Web interface templates

**ML Models (My Trained Models):**
- `ml_models/boarding_pipeline.pkl` - XGBoost model for boarding prediction
- `ml_models/clustering_pipeline.pkl` - DBSCAN clustering model
- `ml_models/association_rules.json` - 112 association rules extracted
- `ml_models/boarding_features.json` - Feature metadata
- `ml_models/clustering_info.json` - Clustering metadata

**Training Notebooks (in separate branches):**
- `boarding-prediction` branch → `boarding1final.ipynb` - Full training pipeline
- `if-then-rules-discovery` branch → `unsupervised_clean.ipynb` - Clustering & rules analysis

---

## 1. Boarding Availability Prediction

### What It Does
Predicts whether a French school offers boarding facilities (`Hebergement`) based on 29 school characteristics.

### Model Performance
- **Algorithm:** XGBoost (selected after comparing 5 models)
- **Accuracy:** ~91%
- **F1-Score:** ~91%
- **Training Method:** 5-fold stratified cross-validation with GridSearchCV
- **Features:** 29 school characteristics (type, tracks, region, programs, etc.)

### Django Implementation

**Files I Created:**
- `BoardinPlusIfRulesApp/views.py` - `boarding_prediction_view()`, `predict_boarding_api()`
- `BoardinPlusIfRulesApp/utils.py` - `load_boarding_model()`, `load_boarding_features()`
- `templates/BoardinPlusIfRulesApp/boarding_prediction.html` - User interface

**API Endpoints:**
- `GET /ml/boarding/` - Prediction form
- `POST /ml/api/predict-boarding/` - Returns prediction (0/1) + probabilities

**Technical Details:**
- Lazy loading of models (loaded on first request)
- Feature validation and ordering
- StandardScaler preprocessing (matches training pipeline)
- Error handling with graceful fallbacks

**Full Training Details:** See `boarding-prediction` branch → `boarding1final.ipynb`
- Data sources: SQL Server + CSV integration
- Class balancing strategy
- Model comparison (Logistic Regression, KNN, SVM, Random Forest, XGBoost)
- Feature selection experiments
- Overfitting analysis

---

## 2. Candidate Pattern Discovery (Association Rules)

### What It Does
Discovers "if-then" rules connecting candidate characteristics (gender, track, services) to exam outcomes (success, mentions) using clustering and association rule mining.

### Model Performance
- **Clustering Algorithm:** DBSCAN (best performing after comparing 3 methods)
  - **Silhouette Score:** 0.980 (excellent)
  - **Davies-Bouldin Index:** 0.431 (very low, excellent)
  - **Clusters Found:** 178 clusters with automatic outlier detection
- **Association Rules:** 112 high-quality rules extracted
  - **Minimum Support:** 5%
  - **Minimum Confidence:** 60%
  - **Minimum Lift:** 1.2
- **Dataset:** 240,597 candidate records

### Django Implementation

**Files I Created:**
- `BoardinPlusIfRulesApp/views.py` - `association_rules_view()`, `predict_cluster_api()`
- `BoardinPlusIfRulesApp/utils.py` - `load_clustering_model()`, `load_association_rules()`, `load_clustering_info()`
- `templates/BoardinPlusIfRulesApp/association_rules.html` - User interface

**API Endpoints:**
- `GET /ml/association-rules/` - Pattern discovery form
- `POST /ml/api/predict-cluster/` - Returns matching rules + cluster analysis

**Technical Details:**
- Intelligent rule matching algorithm (scores rules by relevance)
- Multi-feature rule combination logic
- DBSCAN clustering integration
- Rule formatting with confidence, support, and lift metrics

**Full Training Details:** See `if-then-rules-discovery` branch → `unsupervised_clean.ipynb`
- Data warehouse: 240K+ candidate records from SQL Server
- Three clustering algorithms compared (K-Means, Hierarchical, DBSCAN)
- Optimal K selection (Elbow, Silhouette, Davies-Bouldin)
- Association rule mining (Apriori + FP-Growth)
- Rule quality analysis

---

## Code Structure

### My Django App Structure
```
BoardinPlusIfRulesApp/
├── views.py              # Main business logic (487 lines)
│   ├── boarding_prediction_view()
│   ├── predict_boarding_api()
│   ├── association_rules_view()
│   └── predict_cluster_api()
├── utils.py              # Model loading utilities
│   ├── load_boarding_model()
│   ├── load_boarding_features()
│   ├── load_clustering_model()
│   ├── load_association_rules()
│   └── load_clustering_info()
├── urls.py               # URL routing
└── templates/
    ├── boarding_prediction.html
    └── association_rules.html
```

### My ML Models
```
ml_models/
├── boarding_pipeline.pkl        # XGBoost + preprocessing pipeline
├── clustering_pipeline.pkl       # DBSCAN + StandardScaler
├── association_rules.json        # 112 rules with metrics
├── boarding_features.json        # 29 feature names + metadata
└── clustering_info.json          # Clustering metadata (210 clusters, etc.)
```

---

## Key Technical Highlights

### Production-Ready Implementation
- ✅ **Lazy Model Loading:** Models loaded on first request (avoids slow startup)
- ✅ **Error Handling:** Try/except blocks with graceful fallbacks
- ✅ **Feature Validation:** Ensures correct feature ordering and types
- ✅ **Pipeline Consistency:** Uses same preprocessing as training (StandardScaler)

### ML Engineering Best Practices
- ✅ **Model Serialization:** joblib for efficient model storage
- ✅ **API Design:** RESTful endpoints with JSON responses
- ✅ **Rule Matching Algorithm:** Intelligent scoring for association rule relevance
- ✅ **Probability Scores:** Returns confidence levels, not just predictions

### Code Quality
- ✅ **Separation of Concerns:** `utils.py` for model loading, `views.py` for business logic
- ✅ **Professional UI:** Bootstrap-based responsive interface
- ✅ **Documentation:** Clear function structure and error messages

---

## How to Use My Contribution

### Access My Features

1. **Boarding Prediction:**
   - Navigate to: `http://localhost:8000/ml/boarding/`
   - Enter 29 school characteristics
   - Get prediction + probability scores

2. **Association Rules Discovery:**
   - Navigate to: `http://localhost:8000/ml/association-rules/`
   - Enter candidate characteristics (gender, track, services)
   - Discover matching patterns and rules

### API Usage Examples

**Boarding Prediction API:**
```bash
POST /ml/api/predict-boarding/
Content-Type: application/x-www-form-urlencoded

Average_Score_moyen=15.5&ULIS=0&SEGPA=0&...
```

**Response:**
```json
{
  "success": true,
  "prediction": 1,
  "probability_yes": 0.91,
  "probability_no": 0.09,
  "prediction_text": "Boarding Available"
}
```

**Association Rules API:**
```bash
POST /ml/api/predict-cluster/
Content-Type: application/x-www-form-urlencoded

Sexe_Male=1&Voie_generale=1&Restauration=1&...
```

**Response:**
```json
{
  "success": true,
  "cluster": "N/A",
  "matching_rules": [
    {
      "if_text": "Male + General Track",
      "then_text": "Very Good Mention",
      "confidence": 0.85,
      "support": 0.15,
      "lift": 1.52
    }
  ],
  "total_rules_found": 3
}
```

---

## Training Notebooks (Separate Branches)

For complete methodology, hyperparameter tuning, and evaluation:

### Boarding Prediction Training
📓 **Branch:** `boarding-prediction`  
📓 **Notebook:** `boarding1final.ipynb`

**Contains:**
- Data loading from SQL Server + CSV
- Class balancing strategy
- 5 model comparison (Logistic Regression, KNN, SVM, Random Forest, XGBoost)
- GridSearchCV hyperparameter tuning
- Feature selection experiments
- Overfitting analysis (train vs test)
- Confusion matrices and detailed metrics

### Association Rules Training
📓 **Branch:** `if-then-rules-discovery`  
📓 **Notebook:** `unsupervised_clean.ipynb`

**Contains:**
- Data warehouse extraction (240K+ records)
- Feature engineering (binary encoding)
- Three clustering algorithms (K-Means, Hierarchical, DBSCAN)
- Optimal K selection (Elbow, Silhouette, Davies-Bouldin)
- Association rule mining (Apriori + FP-Growth)
- Rule quality analysis (support, confidence, lift)
- Algorithm comparison with Adjusted Rand Index

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Django 5.2+
- Access to trained models in `ml_models/` directory

### Steps

### Quick Start (No Virtual Environment Required)

You can run the project directly without creating a virtual environment:

1. **Open project in VS Code** (or your preferred IDE)
   - Open the folder: `mlproject-finaall/mlproject-finaall`

2. **Install dependencies**
   pip install -r requirements.txt
   `

5. **Run Django migrations** (if needed)
```bash
python manage.py migrate
```

6. **Start development server**
```bash
python manage.py runserver
```

7. **Access the application** or you can use buttons in the navbar 
   - Boarding Prediction: `http://localhost:8000/ml/boarding/`  (boarding availability)
   - Association Rules: `http://localhost:8000/ml/association-rules/`  (discover "if-then" rules for significant associations)
---

## Technologies & Dependencies

**My Implementation Uses:**
- Django 5.2+ (web framework)
- scikit-learn (ML models)
- XGBoost (boarding prediction)
- mlxtend (association rules)
- pandas, numpy (data processing)
- joblib (model serialization)

**See `requirements.txt` for full dependency list**

---

## Notes

- **Model Files:** The `.pkl` files in `ml_models/` were generated from my training notebooks
- **Integration:** My app is integrated into the main Django project alongside other team members' work
- **Performance:** Models achieve production-ready accuracy (91% for boarding, 0.980 Silhouette for clustering)
- **Reproducibility:** All models use `random_state=42` for consistency

---

## Summary

**My Contribution:**
- ✅ Developed 2 ML models (boarding prediction + association rules)
- ✅ Trained and evaluated models (see branches for notebooks)
- ✅ Deployed models in Django with RESTful APIs
- ✅ Created professional web interfaces
- ✅ Implemented production-ready code with error handling

**Files I Created:**
- `BoardinPlusIfRulesApp/` (entire Django app)
- `ml_models/boarding_pipeline.pkl` and related files
- `ml_models/clustering_pipeline.pkl` and related files
- `ml_models/association_rules.json` and metadata files

**For complete training details, see the separate branches:**
- `boarding-prediction` branch
- `if-then-rules-discovery` branch
