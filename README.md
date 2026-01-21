# Unsupervised Learning: Candidate Clustering and Association Rules

**Goal:** Discover "if-then" type rules that reveal meaningful relationships between candidates' characteristics (sex, track, series, establishment features) and their exam results (success, mentions).

This repository contains:

- **Notebook:** `unsupervised_clean.ipynb` – complete unsupervised learning analysis including clustering (K-Means, Hierarchical, DBSCAN) and association rule mining (Apriori, FP-Growth).
- **Environment files:** `requirements.txt` to reproduce the environment.
- **(Planned)** A separate branch for Django deployment.

---

## 1. Project Overview

Understanding patterns in educational outcomes is crucial for policy makers, educators, and administrators. This project applies unsupervised learning techniques to identify meaningful groups of candidates and discover association rules that connect candidate characteristics to exam performance.

**Main workflow in the notebook:**

1. **Data Loading:** Extract candidate-level data from SQL Server data warehouse (240K+ records)
2. **Feature Engineering:** Create binary features for sex, tracks, series, establishment services, and outcomes
3. **Clustering Analysis:**
   - **K-Means:** Optimal K selection using Elbow, Silhouette, and Davies-Bouldin metrics
   - **Hierarchical Clustering (CAH):** Ward linkage with dendrogram visualization
   - **DBSCAN:** Density-based clustering with automatic outlier detection
4. **Algorithm Comparison:** Evaluate all methods using multiple metrics and Adjusted Rand Index
5. **Association Rule Mining:**
   - **Apriori algorithm:** Find frequent itemsets and extract "if-then" rules
   - **FP-Growth algorithm:** Efficient alternative for large datasets
   - Compare both methods and visualize rule quality metrics

---

## 2. Data Sources

**Note:** The original CSV and SQL Server database are not included in this repo due to size constraints. The notebook contains all code and results

### Primary Source – SQL Server Data Warehouse

The analysis uses a star schema data warehouse with the following tables:

- **`fact_resultats`:** Fact table with exam results (number of enrolled, present, admitted, refused, mentions)
- **`dim_candidat`:** Candidate dimensions (ID, sex, status)
- **`dim_formation`:** Training program dimensions (track: general/technological/professional, series, diploma specialty)
- **`fact_table_etab`:** Establishment features (boarding, catering, international section, average score)
- **`dim_etablissement`:** Establishment type (public/private, GRETA, PIAL)

**Data volume:** 240,597 candidate records with 24 original features.

### Feature Engineering

The notebook transforms raw data into:

- **Binary features:** Sex (Male/Female), tracks (General/Technological/Professional), series (S, ES, L, STI, STL, etc.), establishment services (boarding, catering, international section)
- **Outcome variables:** Success (admitted/refused), mentions (TB, B, AB, None)
- **Continuous features:** Average establishment score (with median imputation for missing values)

**Final feature set for clustering:** 12 selected features (sex, tracks, outcomes, establishment score, services)

---

## 3. Methodology

### 3.1 Clustering Algorithms

#### K-Means Clustering
- **Optimal K selection:** Test K=2 to K=15 using:
  - **Elbow method:** Inertia reduction curve
  - **Silhouette score:** Measure of cluster cohesion and separation
  - **Davies-Bouldin index:** Lower is better (compact, well-separated clusters)
- **Final model:** K=15 clusters (selected by highest Silhouette score)
- **Standardization:** All features standardized using `StandardScaler` before clustering

#### Hierarchical Clustering (CAH)
- **Linkage method:** Ward (minimizes within-cluster variance)
- **Optimization:** Uses sample of 5,000 candidates for linkage computation (computational efficiency)
- **Hybrid approach:** Maps full dataset using centroids from sample as K-Means initialization
- **Visualization:** Dendrogram showing cluster hierarchy

#### DBSCAN
- **Parameter tuning:** Grid search over `eps` (0.3-2.0) and `min_samples` (5-20)
- **Selection criteria:** Best Silhouette score among valid configurations
- **Advantages:** Automatically detects outliers (noise points), finds clusters of arbitrary shapes
- **Results:** 178 clusters with 2.8% outliers detected

### 3.2 Evaluation Metrics

All clustering methods are evaluated using:

- **Silhouette Score** (range: -1 to 1, higher is better)
  - Measures how similar an object is to its own cluster vs other clusters
- **Davies-Bouldin Index** (lower is better)
  - Ratio of within-cluster distance to between-cluster distance
- **Calinski-Harabasz Score** (higher is better)
  - Ratio of between-clusters dispersion to within-cluster dispersion
- **Adjusted Rand Index (ARI)** (range: 0 to 1)
  - Measures agreement between different clustering methods (1 = identical, 0 = random)

### 3.3 Association Rule Mining

**Objective:** Discover rules like "IF candidate is Male AND track is Professional THEN success rate > X%"

**Algorithms:**
- **Apriori:** Classic frequent itemset mining algorithm
- **FP-Growth:** More efficient for large datasets (same results, faster execution)

**Parameters:**
- **Minimum support:** 5% (itemset must appear in at least 5% of transactions)
- **Minimum confidence:** 60% (rule must be true in at least 60% of cases where antecedent is true)
- **Minimum lift:** 1.2 (rule must be at least 20% more likely than random)

**Transaction matrix:** 240,597 transactions × 22 binary items (sex, tracks, series, establishment features, outcomes)

---

## 4. Results

### 4.1 Clustering Performance

**Best overall method:** DBSCAN
- **Silhouette Score:** 0.980 (excellent)
- **Davies-Bouldin Index:** 0.431 (very low, excellent)
- **Calinski-Harabasz Score:** 201,696 (very high)
- **Clusters found:** 178 (with automatic outlier detection)

**K-Means (K=15):**
- **Silhouette Score:** 0.550
- **Davies-Bouldin Index:** 0.974
- **Calinski-Harabasz Score:** 67,777

**Hierarchical Clustering (CAH, K=15):**
- **ARI with K-Means:** 0.692 (moderate agreement)
- **ARI with DBSCAN:** 0.617 (moderate agreement)

**Key insights:**
- DBSCAN excels at finding natural clusters and detecting outliers
- K-Means and CAH show moderate agreement (ARI ~0.7), suggesting consistent structure
- All methods identify meaningful candidate groups based on track, outcomes, and establishment features

### 4.2 Association Rules

**Frequent itemsets found:** 132 itemsets (with max 3 items per set)

**Rules extracted:** 112 high-quality rules (after filtering by confidence ≥60% and lift ≥1.2)

**Top rules by lift:**
- `IF {Serie_S} THEN {Serie_ST2S}` (lift=5.805, confidence=100%)
- `IF {Serie_ST2S} THEN {Serie_S}` (lift=5.805, confidence=100%)
- `IF {Reussite, Serie_S} THEN {Serie_ST2S}` (lift=5.805, confidence=100%)

**Key patterns discovered:**
- Strong associations between series (S and ST2S are highly correlated)
- Success (Reussite) is strongly associated with professional tracks
- Mention types (TB, B, AB) show distinct patterns by track and series

**Apriori vs FP-Growth:**
- Both algorithms find identical results (132 itemsets, 112 rules)
- FP-Growth is faster on large datasets (same quality, better performance)

---

## 5. How to Run the Notebook

### 5.1. Create and activate a virtual environment (Windows PowerShell)

```powershell
cd "C:\Users\abidi\OneDrive\Bureau\ML2\ML"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Key Technical Highlights

### Algorithm Selection and Validation
- **Three clustering methods compared:** K-Means (fast, interpretable), Hierarchical CAH (hierarchical structure), DBSCAN (automatic outlier detection, best performance)
- **Optimal K selection:** Systematic evaluation using Elbow method, Silhouette score, and Davies-Bouldin index
- **Cross-validation:** Adjusted Rand Index (ARI) shows moderate agreement between methods (0.65-0.69), validating cluster structure

### Scalability and Optimization
- **Large dataset handling:** 240K+ records processed using sampling strategies (10K for Silhouette, 5K for CAH) and chunked SQL loading
- **Hybrid approach:** CAH computed on sample, then mapped to full dataset using K-Means initialization
- **Algorithm efficiency:** FP-Growth used for association rules (faster than Apriori on large datasets)

### Data Quality and Preprocessing
- **Feature engineering:** Binary encoding for categorical variables, median imputation for missing values
- **Standardization:** Applied before all distance-based algorithms (critical for clustering)
- **Feature selection:** Excluded redundant variables (series vs tracks) based on correlation analysis

### Production Readiness
- **Reproducibility:** Random seeds (`random_state=42`) and version-controlled dependencies
- **Model persistence:** Ready for deployment (models can be saved as `.pkl` files)
- **Scalable architecture:** Designed for batch processing and real-time inference


**The deployment code and instructions will live in a dedicated branch main .



