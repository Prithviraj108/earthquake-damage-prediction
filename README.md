# earthquake-damage-prediction
ML internship project to predict earthquake structural damage grades (1–3) using XGBoost, Random Forest &amp; Logistic Regression on Nepal building data. Includes EDA, feature engineering, and a Colab-ready notebook.

# 🏚️ Earthquake Damage Prediction
> **Internship Project** — Data Science & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 About the Project

This project was built during our **Data Science internship** as a hands-on exercise in applied machine learning. The objective was to predict the **structural damage grade** of buildings following an earthquake, based on building characteristics and geographic features.

The problem is inspired by the **2015 Nepal Gorkha Earthquake** — one of the deadliest earthquakes in Nepal's history. Accurately predicting damage levels helps disaster response teams prioritize resources, guide relief efforts, and inform future construction policies.

### 🎯 Target Variable — Damage Grade

| Grade | Meaning |
|-------|---------|
| **1** | Low Damage |
| **2** | Medium Damage |
| **3** | High / Complete Destruction |

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
├── 📓 Earthquake_Damage_Prediction.ipynb   ← Main Jupyter Notebook (run on Colab)
├── 🐍 earthquake_damage_prediction.py      ← Standalone Python script
├── 📄 README.md                            ← You are here
│
├── data/
│   └── earthquake_data.csv                 ← Dataset (auto-downloaded or synthetic)
│
├── outputs/
│   ├── eda_overview.png                    ← Grade distribution, age, foundation plots
│   ├── correlation_heatmap.png             ← Feature correlation matrix
│   ├── superstructure_analysis.png         ← Damage by building material
│   ├── model_comparison.png               ← F1-macro & accuracy across models
│   └── best_model_analysis.png            ← Confusion matrix + feature importances
│
└── models/
    └── best_model_xgboost.pkl              ← Saved best model (joblib)
```

---

## 🔄 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  ML PIPELINE                                │
│                                                             │
│  1. Data Loading      →  Kaggle API / Synthetic fallback    │
│  2. EDA               →  Distributions, correlations,       │
│                           material analysis                 │
│  3. Feature Engg.     →  4 new derived features             │
│  4. Model Training    →  3 classifiers, 5-fold CV           │
│  5. Best Model        →  Confusion matrix, importances      │
│  6. Predictions       →  Demo for 3 building scenarios      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Feature Engineering

We created four additional features beyond the raw dataset:

| Feature | Formula | Insight |
|---|---|---|
| `age_bucket` | Age grouped into 5 risk tiers | Older buildings = higher risk |
| `total_superstructures` | Sum of all material flags | Diversified materials → resilience |
| `area_height_ratio` | area% / height% | Squat buildings → more stable |
| `floor_density` | floors / area% | Tall, narrow buildings → higher risk |

---

## 📊 Results

### Model Comparison

| Model | CV F1-macro | Test F1-macro | Test Accuracy | Time |
|---|---|---|---|---|
| Logistic Regression | 0.743 ± 0.003 | 0.740 | 74.9% | ~5s |
| Random Forest | 0.807 ± 0.004 | 0.805 | 81.3% | ~25s |
| **XGBoost ★** | **0.825 ± 0.004** | **0.817** | **82.3%** | ~35s |

> **XGBoost** was selected as the final model based on highest F1-macro score across both cross-validation and held-out test set.

### Classification Report (XGBoost on Test Set)

```
                    precision  recall  f1-score  support
Grade 1 (Low)         0.89     0.89     0.89     4,219
Grade 2 (Medium)      0.71     0.74     0.73     3,164
Grade 3 (High)        0.85     0.82     0.83     2,617

accuracy                         0.82    10,000
macro avg             0.82     0.82     0.82    10,000
```

---

## 🔍 Key Findings

Through our analysis, we found several strong patterns in the data:

- **Age is the single strongest predictor** — buildings older than 70 years show dramatically higher rates of Grade 3 (complete destruction).
- **Foundation type is critical** — mud-mortar-stone foundations are the highest-risk category, while RC (reinforced concrete) foundations protect buildings significantly.
- **Adobe mud and mud-mortar-stone superstructures** are heavily correlated with the worst damage outcomes.
- **RC engineered superstructures** are the most protective material type and reduce predicted damage grade considerably.
- **Geographic location** introduces additional variance even after controlling for building characteristics, suggesting local soil and topographic effects.

---

## 🚀 Getting Started

### Run on Google Colab (Recommended)
1. Download `Earthquake_Damage_Prediction.ipynb`
2. Go to [colab.research.google.com](https://colab.research.google.com)
3. File → Upload Notebook → select the `.ipynb` file
4. Run All Cells

### Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/earthquake-damage-prediction.git
cd earthquake-damage-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost kagglehub joblib

# Run the pipeline
python earthquake_damage_prediction.py
```

### Dataset

The notebook automatically downloads the dataset from Kaggle if credentials are available. Otherwise, a **realistic synthetic dataset** mirroring the original schema is generated automatically — no setup needed.

To use the real Kaggle dataset:
1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account → Create API Token → download `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualizations |
| Scikit-learn | ML pipeline, preprocessing, baselines |
| XGBoost | Final model |
| Kagglehub | Dataset download |
| Joblib | Model serialization |
| Jupyter / Google Colab | Notebook environment |

---

## 📈 Sample Predictions

```
Scenario: Old mud-mortar building (80 years, adobe superstructure)
→ Grade 3 — High / Complete Destruction
   Probabilities: Grade 1: 2.1%  |  Grade 2: 11.4%  |  Grade 3: 86.5%

Scenario: New RC engineered building (5 years, reinforced concrete)
→ Grade 1 — Low Damage
   Probabilities: Grade 1: 91.2%  |  Grade 2: 7.6%  |  Grade 3: 1.2%

Scenario: Mid-age brick building (35 years, cement mortar brick)
→ Grade 2 — Medium Damage
   Probabilities: Grade 1: 22.3%  |  Grade 2: 59.8%  |  Grade 3: 17.9%
```

---

## 💡 What We Learned

This internship project gave us hands-on experience with the **end-to-end machine learning workflow**:

- Handling real-world messy datasets with mixed types and skewed distributions
- The importance of **feature engineering** — our 4 derived features improved F1 by ~2%
- **Cross-validation over a simple train/test split** to get reliable model estimates
- Why **F1-macro** matters more than accuracy for imbalanced multi-class problems
- XGBoost's robustness and speed advantage over Random Forest for tabular data
- Communicating findings clearly through visualizations and structured reports

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Dataset by [prithvirajshukla](https://www.kaggle.com/datasets/prithvirajshukla/earthquake-damage-prediction) on Kaggle
- Inspired by the [DrivenData Richter's Predictor competition](https://www.drivendata.org/competitions/57/nepal-earthquake/)
- Thanks to our mentor and the internship organization for guidance throughout this project

---

*Built with ❤️ during our Data Science internship.*
