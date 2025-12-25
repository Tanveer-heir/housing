# 🏠 House Price Prediction using XGBoost (Advanced Feature Engineering)

This project builds a **high-performance house price prediction model** using the **Zillow Properties 2016 dataset**, focusing on **robust feature engineering, outlier handling, and tree-based modeling (XGBoost)**.  

The goal is not just prediction accuracy, but **interpretable insights into how property characteristics and amenities influence housing prices**.

---

## 📌 Problem Statement

Predict the **tax-assessed property value (`taxvaluedollarcnt`)** using structured real-estate attributes such as:

- Room counts  
- Square footage  
- Property age  
- Regional identifiers  
- Amenities (basement, pool, garage, fireplace)  

---

## 🧠 Key Highlights

- Extensive **domain-driven feature engineering**  
- **Outlier capping** instead of naive removal  
- **Log-transformation** for skewed financial features  
- Hybrid **categorical encoding** (Label Encoding + One-Hot Encoding)  
- Tuned **XGBoost Regressor**  
- **Amenity-level economic analysis**  
- Scalable pipeline (tested on **200k samples**)  

---

## 📂 Dataset

- **Source**: Zillow Properties 2016  
- **Target Variable**: `taxvaluedollarcnt`  
- **Rows used**: 200,000 (sampled for faster experimentation)  
- **Features**: Numerical, categorical, and engineered attributes  

---

## 🔧 Project Pipeline

### 1️⃣ Data Loading & Column Pruning

- Removed identifiers, high-missing columns, and irrelevant geographic fields  
- Reduced noise and memory footprint early  

---

### 2️⃣ Numeric Outlier Handling

Instead of dropping data, **caps were applied**:

- Logical caps (e.g., rooms, bathrooms)  
- 99th-percentile caps for large square-footage and price variables

  
---

### 3️⃣ Feature Engineering

- `_clean` capped features  
- `_log1p` transformations for skewed distributions  
- Binary flags (e.g., `has_basement`)  
- Year built bounds (1850–2025)  

---

### 4️⃣ Categorical Encoding Strategy

| Feature Type              | Encoding Strategy   |
|---------------------------|---------------------|
| Low-cardinality systems   | One-Hot Encoding    |
| High-cardinality regions  | Label Encoding      |

This avoids unnecessary feature explosion while preserving signal.

---

### 5️⃣ Feature Selection

Only **model-relevant engineered features** were retained:

- `_clean`, `_log1p`  
- Encoded categorical variables  
- Amenity flags  

Target-leaking variables were **explicitly excluded**.

---

### 6️⃣ Train–Test Split & Scaling

- **80–20** train–test split  
- `StandardScaler` applied to numeric columns  
- Scaling retained for model portability and experimentation  

---

### 7️⃣ Model Training (XGBoost)

**Model:** `XGBRegressor`  

- **Objective:** Squared Error Regression  
- Tuned hyperparameters:  
  - Tree depth, learning rate  
  - Subsampling & column sampling  
  - `min_child_weight` & `gamma`  

This balances the **bias–variance tradeoff** effectively.

---

### 8️⃣ Model Evaluation

Metrics used:

- **RMSE**  
- **R² Score**  

Results (indicative, dataset-dependent):

- **R²** → high on validation set  
- **RMSE** → minimized through tuning  

---

### 9️⃣ Feature Importance Analysis

- Extracted top contributing features from the trained XGBoost model  
- Visualized using **horizontal bar charts**  
- Helps interpret what actually **drives property prices**  

---

### 🔟 Amenity Impact Analysis

Evaluated economic impact of:

- Basement  
- Pool  
- Garage capacity  
- Fireplace  

**Approach:**

- Binary grouping (amenity present vs absent)  
- Mean price comparison  
- Boxplots for distribution analysis  

This converts ML results into **real-world, business-facing insights**.

---

## 📊 Sample Insights

- Properties with **basements and pools** show significantly higher median prices  
- **Garage capacity** correlates strongly with property value  
- **Structural square footage** dominates prediction power  

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  

---

## 📈 Future Improvements

- Cross-validation with **early stopping**  
- **SHAP-based** model explainability  
- Time-aware modeling (market/year trends)  
- Deployment via **FastAPI**  
- Integration into a **real-estate decision dashboard**  

---

## 🎯 Why This Project Matters

This project demonstrates:

- Applied **ML engineering**  
- **Feature design** thinking  
- Production-ready **preprocessing**  
- **Interpretability** beyond raw accuracy  

It reflects how ML models are built in **real industry workflows**, not just in isolated notebooks.

---

## 👤 Author

**Tanveer Singh**  
Machine Learning & Backend Engineering Enthusiast



