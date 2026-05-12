# earthquake-damage-prediction
ML internship project to predict earthquake structural damage grades (1–3) using XGBoost, Random Forest &amp; Logistic Regression on Nepal building data. Includes EDA, feature engineering, and a Colab-ready notebook.

# 🏚️ Richter's Predictor: Modeling Earthquake Damage
> **Internship Project** — Data Science & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Competition](https://img.shields.io/badge/DrivenData-Richter's%20Predictor-blue)](https://www.drivendata.org/competitions/57/nepal-earthquake/)

---

## 📖 About the Project

This project was built during our **Data Science internship** as a hands-on exercise in applied machine learning. The dataset is from the **DrivenData "Richter's Predictor"** competition, based on the 2015 Nepal Gorkha Earthquake — one of the most devastating earthquakes in the region's modern history.

The data was collected by **Kathmandu Living Labs** and the **Central Bureau of Statistics of Nepal**, making it one of the largest post-disaster datasets ever assembled.

### 🎯 Target Variable — Damage Grade

| Grade | Meaning | Count | Share |
|-------|---------|-------|-------|
| 1 | Low damage | 25,124 | 9.6% |
| 2 | Medium damage | 148,259 | 56.9% |
| 3 | Complete destruction | 87,218 | 33.5% |

> The dataset is imbalanced — **micro-averaged F1** is used as the evaluation metric (per DrivenData competition rules).

---

## 👥 Team

| Role | Name |
|---|---|
| Intern | Prithviraj |
| Intern | Uday, Priyanka, Nirmala |
| Mentor | Shravani |
| Organization | DataMites |
| Duration | 10th Nov 2024 – 25th Nov 2024 |

---

## 📁 Repository Structure

```
earthquake-damage-prediction/
│
├── 📓 Earthquake_Damage_Prediction.ipynb   ← Main Jupyter Notebook (Colab-ready)
├── 📄 README.md                            ← You are here
│
├── data/
│   ├── train_values.csv                    ← 260,601 buildings × 38 features
│   └── train_labels.csv                   ← building_id + damage_grade (target)
│
└── models/
    └── best_model_xgboost.pkl              ← Saved best model
```

---

## 📊 Dataset

- **Source:** 2015 Nepal Gorkha Earthquake post-disaster survey
- **Collected by:** Kathmandu Living Labs & Central Bureau of Statistics, Nepal
- **Records:** 260,601 buildings
- **Features:** 38 (geographic IDs, structural characteristics, superstructure materials, secondary use flags)
- **Missing values:** 0
- **Files:** `train_values.csv` (features) + `train_labels.csv` (target)

### Feature Groups

| Group | Features |
|---|---|
| Geographic | `geo_level_1_id`, `geo_level_2_id`, `geo_level_3_id` |
| Structural | `count_floors_pre_eq`, `age`, `area_percentage`, `height_percentage` |
| Construction | `foundation_type`, `roof_type`, `ground_floor_type`, `other_floor_type`, `plan_configuration`, `position` |
| Superstructure (binary) | `has_superstructure_adobe_mud`, `_mud_mortar_stone`, `_rc_engineered`, + 8 more |
| Secondary use (binary) | `has_secondary_use`, `_agriculture`, `_hotel`, `_school`, + 7 more |
| Ownership | `legal_ownership_status`, `count_families` |

---

## 🔄 Pipeline Overview

```
1. Data Loading      →  Merge train_values.csv + train_labels.csv (260,601 rows)
2. EDA               →  Class distribution, age histogram, foundation & material analysis,
                         correlation heatmap
3. Feature Engg.     →  5 new derived features (age_bucket, total_superstructures,
                         area_height_ratio, floor_density, total_secondary_uses)
4. Model Training    →  3 classifiers with 5-fold stratified CV (F1-micro)
5. Best Model        →  Confusion matrix, feature importance ranking
6. Predictions       →  Demo predictions for sample building scenarios
```

---

## 📈 Results

| Model | CV F1-micro | Test F1-micro | Test Accuracy |
|---|---|---|---|
| Logistic Regression | ~0.57 ± 0.001 | ~0.57 | ~57% |
| Random Forest | ~0.71 ± 0.002 | ~0.71 | ~71% |
| **XGBoost ★** | **~0.75 ± 0.001** | **~0.75** | **~75%** |

**XGBoost** was selected as the final model.

### Classification Report (XGBoost, Test Set)

```
                       precision  recall  f1-score  support
Grade 1 (Low)            0.62     0.39     0.48      5,025
Grade 2 (Medium)         0.74     0.86     0.80     29,652
Grade 3 (Complete)       0.72     0.62     0.67     17,444

accuracy                           0.75    52,121
macro avg                0.69     0.62     0.65     52,121
micro avg                0.75     0.75     0.75     52,121
```

---

## 🔍 Key Findings

- **Building age** is the strongest single predictor — older buildings suffer dramatically more damage.
- **Foundation type** is highly predictive: mud-mortar-stone correlates with highest damage; reinforced concrete is most protective.
- **Adobe mud and mud-mortar-stone superstructures** are heavily linked to complete destruction.
- **RC engineered superstructures** are the most protective material type.
- **Grade 1 (Low damage)** is hardest to classify — only 9.6% of buildings, leading to recall of ~39%.
- Geographic region adds significant predictive signal even after structural features.

---

## 🚀 How to Run

### On Google Colab (recommended)
1. Upload `train_values.csv` and `train_labels.csv` to your Colab session
2. Upload `Earthquake_Damage_Prediction.ipynb`
3. Run all cells

### Locally

```bash
git clone https://github.com/yourusername/earthquake-damage-prediction.git
cd earthquake-damage-prediction
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
jupyter notebook Earthquake_Damage_Prediction.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Pandas & NumPy | Data wrangling |
| Matplotlib & Seaborn | Visualizations |
| Scikit-learn | Preprocessing, baselines, cross-validation |
| XGBoost | Best-performing model |
| Joblib | Model persistence |
| Google Colab | Development environment |

---

## 💡 What We Learned

- How to handle large real-world datasets (260K rows, 38 features, zero missing values)
- Why **micro-F1** is more appropriate than accuracy for imbalanced multi-class problems
- The importance of feature engineering — our derived features improved F1 by ~2%
- That XGBoost consistently outperforms Random Forest on structured/tabular data
- How to build a clean, reproducible ML pipeline end-to-end in a Jupyter notebook

---

## 🙏 Acknowledgements

- Dataset & competition: [DrivenData — Richter's Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/)
- Data collected by: [Kathmandu Living Labs](http://www.kathmandulivinglabs.org/) & [Central Bureau of Statistics Nepal](https://cbs.gov.np/)
- Thanks to our mentor and internship organization for guidance throughout this project

---

*Built with ❤️ during our Data Science internship.*
