# Explainable Machine Learning for Heart Disease Risk Prediction

## 📌 Project Overview

This project develops a machine learning-based system for predicting heart disease risk using demographic, medical, and lifestyle-related features.

The project focuses on handling imbalanced healthcare data using SMOTE and improving model interpretability using SHAP (SHapley Additive exPlanations).

## 🎯 Objectives

- Analyze healthcare data related to heart disease.
- Perform exploratory data analysis (EDA).
- Handle class imbalance using SMOTE.
- Train and compare multiple machine learning models.
- Evaluate models using Accuracy, Precision, Recall, F1-score, ROC-AUC and PR-AUC.
- Apply SHAP for model explainability.
- Perform hyperparameter tuning using GridSearchCV.
- Develop a Streamlit web application for prediction.

## 📊 Dataset

- Total records: **5,568**
- Total columns: **19**
- Input features: **18**
- Target variable: `heart_disease`

The target variable is imbalanced:

| Class | Records | Percentage |
|---|---:|---:|
| No Heart Disease | 5,041 | 90.53% |
| Heart Disease | 527 | 9.46% |

## 🔧 Methodology

The project follows this workflow:

```text
Dataset
   ↓
Data Understanding & Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
SMOTE for Class Balancing
   ↓
Feature Scaling
   ↓
Machine Learning Models
   ↓
Model Evaluation
   ↓
SHAP Explainability
   ↓
Hyperparameter Tuning
   ↓
Threshold Analysis
   ↓
Final Model
   ↓
Streamlit Application
