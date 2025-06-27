# Breast Cancer Diagnosis Analysis

A Streamlit‐powered data science project that walks you through exploratory analysis, preprocessing, model training and evaluation—comparing traditional machine learning and deep learning approaches—for automatic breast cancer diagnosis using the UCI Breast Cancer Wisconsin (Diagnostic) Dataset.

---

## 🔎 Project Overview

Early detection of breast cancer is critical for patient outcomes. This application:

- **Explores & cleans** the Wisconsin Diagnostic dataset  
- **Handles** missing values, outliers, feature scaling & selection  
- **Trains** five classic ML models (SVM, Logistic Regression, KNN, XGBoost, Random Forest)  
- **Trains** four deep learning architectures (MLP, CNN, LSTM, TabNet)  
- **Evaluates** all models via 10-fold cross-validation and test‐set metrics (accuracy, precision, recall, F1-score, confusion matrices)  
- **Provides** an interactive web UI for dynamic analysis and easy result comparison  

---

## ⚙️ Features

1. **About / Profile**  
   - Quick bio card with education, GPA, LinkedIn and CV download  

2. **Dynamic Data Analysis**  
   - Upload your own CSV or use default  
   - Interactive filters, boxplots, summary stats, outlier detection  

3. **Data Analysis & Preprocessing**  
   - Missing value reports & removal options  
   - Outlier detection via IQR & boxplots  
   - Correlation-based feature selection  
   - StandardScaler / MinMaxScaler, label encoding for targets  

4. **Model Training & Evaluation**  
   - Traditional models: KNN, SVM, Logistic Regression, Random Forest, XGBoost  
   - Deep models: MLP, CNN, LSTM, TabNet  
   - GridSearchCV hyperparameter tuning  
   - 10-fold cross-validation plots & metrics dashboards  

---

## 📁 File Structure

