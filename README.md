## School Boarding Prediction (France)

**Goal**: Predict whether a French school offers boarding (`Hebergement`) using administrative and academic features.

This repository currently contains:
- **Notebook**: `boarding1final.ipynb` – full data preparation, modeling, evaluation, and analysis.
- **Environment files**: `requirements.txt` to reproduce the environment.
- **(Planned)** A separate branch for Django deployment .

---

## 1. Project Overview

In many education systems, boarding capacity is important for planning transport, housing, and resources.  
This project builds a **binary classifier** (boarding vs no boarding) for French schools based on public registry data and enriched features.

**Main workflow in the notebook:**

1. Load and clean the raw data.
2. Balance the target classes (boarding vs non‑boarding).
3. Select relevant features and convert them to numeric form.
4. Train and tune several models:
   - Logistic Regression  
   - K‑Nearest Neighbors (KNN)  
   - Support Vector Machine (SVM)  
   - Random Forest  
   - XGBoost
5. Compare models using **Accuracy** and **F1‑Score**.
6. Check for overfitting by comparing train vs test performance.
7. Explore feature selection with **SelectKBest**.

---

## 2. Data Sources
> Note: The original CSV and SQL Server backup are not included in this repo due to size and privacy constraints.  
> The notebook contains all code and results; it can be adapted to new data with the same schema.

The modeling data is built from **two sources**:

- **Primary source – SQL Server**  
  Core school registry and characteristics (school type, public/private status, educational tracks, number of students, region, etc.) are stored in a SQL Server database.

- **Secondary source – CSV file**  
  Two additional features were only available via a CSV export. These columns were imported from CSV and **joined to the main table on the school identifier**.

This setup reflects a realistic scenario where data comes from both a relational database and flat files.  
In the notebook, the final modeling dataset is loaded from a prepared CSV that already integrates:
- all main SQL‑origin features, and  
- the 2 extra CSV‑origin features.

---

## 3. Modeling Approach

**Target variable**:  
- `Hebergement` (1 = school offers boarding, 0 = no boarding)

### 3.1 Class balancing

The original data is highly imbalanced (many more non‑boarding schools).

Steps:
- Drop rows with missing target (`Hebergement`).
- Keep **all positive (boarding)** samples.
- Randomly sample a subset of negative (non‑boarding) samples to obtain a more balanced dataset.

This makes training more stable and improves F1‑Score.

### 3.2 Feature selection

- Remove identifiers, textual labels, contact details, and other non‑predictive columns.
- Keep structural and educational features, such as:
  - school type,
  - public/private status,
  - educational tracks (general / technological / professional),
  - number of students,
  - region codes,
  - special sections and programs,
  - added score‑based features.
- Convert all features to numeric types and handle missing values.

### 3.3 Models and hyperparameter tuning

All models are wrapped in **`Pipeline`** objects with:
- `SimpleImputer(strategy="median")` for missing values
- `StandardScaler` for feature scaling (critical for KNN and SVM)

Models trained:
- Logistic Regression (with class weights)
- K‑Nearest Neighbors (KNN)
- Support Vector Machine (SVM, RBF kernel)
- Random Forest
- XGBoost

**Hyperparameter tuning**:
- `GridSearchCV` with **5‑fold stratified cross‑validation**
- Scoring metric: **F1‑Score** (to balance precision and recall)

The notebook reports:
- best hyperparameters,
- cross‑validated F1‑Score on training folds,
- performance on the held‑out test set.

---

## 4. Results

On the held‑out test set:

- **Best overall model (Accuracy & F1)**: **XGBoost**
  - Accuracy ≈ 0.91  
  - F1‑Score ≈ 0.91

Other models (KNN, Random Forest) also show strong and competitive performance.

Additional analysis includes:
- Confusion matrices for all models (with precision and recall per class).
- Overfitting check via train vs test F1‑Score (gaps are small, indicating **good generalization**).
- `SelectKBest` feature selection experiments:
  - Confirms that a subset of features carries most predictive power.
  - Slight improvements or similar performance depending on the model.

---

## 5. How to Run the Notebook

### 5.1. Create and activate a virtual environment (Windows PowerShell)

```bash
cd "C:\Users\abidi\OneDrive\Bureau\ML1\ML"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2. Start Jupyter and open the notebook

```bash
jupyter notebook
```

Then open `boarding1final.ipynb` and **Run All** cells from top to bottom.

> Note: The notebook expects access to the prepared CSV with the merged SQL + CSV features.  
> If you move the data file, update the path in the `pd.read_csv(...)` call.

---

## 6. Future Work and Deployment

Planned next steps (in a separate branch):

- Add a **Django** project that loads the trained model 

The deployment code and instructions will live in a dedicated branch for the supervised and unsupervised objectives named (deployment-django).

