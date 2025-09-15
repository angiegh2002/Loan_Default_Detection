# 🏦 Loan Default Detection System  
> **A Machine Learning system to predict loan default risk using ensemble models and advanced data balancing techniques**

![Python](https://img.shields.io/badge/Python-3.13.5-blue)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)  
![XGBoost](https://img.shields.io/badge/XGBoost-02569B?logo=xgboost&logoColor=white)  
![LightGBM](https://img.shields.io/badge/LightGBM-1C8EC9?logo=lightgbm&logoColor=white)  
![CatBoost](https://img.shields.io/badge/CatBoost-C8AA00?logo=catboost&logoColor=black)

This project focuses on building a robust machine learning model to predict the likelihood of **loan default** using real-world financial data. It addresses the challenge of **imbalanced datasets** through techniques like SMOTE, ADASYN, and class weighting, and leverages powerful ensemble models such as **XGBoost, LightGBM, CatBoost, and Stacking** optimized via **Optuna**.

The goal is to support financial institutions in making informed lending decisions by identifying high-risk applicants early and accurately.

## 🌟 Key Features

- ✅ **Imbalanced Data Handling**: Applied SMOTE, ADASYN, and class weight adjustment to balance the target distribution.
- 📊 **Comprehensive Feature Engineering**: Handled missing values, encoded categorical variables, and created derived features like LTV and risk flags.
- 🤖 **Multiple Ensemble Models**: Trained and compared:
  - XGBoost
  - LightGBM
  - CatBoost
  - Random Forest
  - Extra Trees
  - AdaBoost
  - Voting & Stacking Classifiers
- 🔍 **Hyperparameter Optimization**: Used **Optuna** for automated tuning to maximize F1-score.
- 📈 **Performance Evaluation**: Evaluated using precision, recall, F1-score, AUC-ROC, and G-Mean to ensure balanced performance on both classes.

## 📁 Dataset

- Source: [Kaggle Loan Default Dataset](https://www.kaggle.com/datasets)
- Size: 148,670 samples (34 features)
- Target: `Status` — `0` (No default) / `1` (Default)
- Key Features: `Loan amount`, `Interest rate`, `Income`, `Creditworthiness`, `LTV`, `dtir1`, `Property value`, `Neg amortization`, etc.

## 🛠️ Technologies Used

| Component | Technology |
|--------|------------|
| **ML Frameworks** | XGBoost, LightGBM, CatBoost, Scikit-learn |
| **Data Balancing** | SMOTE, ADASYN |
| **Hyperparameter Tuning** | Optuna |
| **Data Processing** | Pandas, NumPy |
| **Evaluation** | Classification Report, Confusion Matrix, ROC Curve |

## 🧪 Model Performance (Best Result)

| Model | F1-Score | Precision | Recall | AUC-ROC |
|------|----------|-----------|--------|--------|
| **Stacking (XGB + LGBM + CAT + DT)** | **0.7925** | 0.6842 | 0.5558 | 0.88 |
| LightGBM | 0.7810 | 0.6700 | 0.5400 | 0.87 |

> The **Stacking ensemble** achieved the highest F1-score, outperforming individual models.

