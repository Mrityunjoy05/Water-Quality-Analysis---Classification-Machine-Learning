# 💧 Water Quality Classification System

A production-ready machine learning system for multi-class water quality classification using physical, chemical, and biological parameters.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Configuration](#configuration)

---

## 🎯 Overview

This project implements a comprehensive machine learning pipeline for classifying water quality based on various parameters. It includes data preprocessing, feature engineering, multiple classification algorithms, hyperparameter tuning, and a user-friendly web interface.

### **Key Objectives:**
- Automated water quality classification
- Support for multiple ML algorithms
- Handle imbalanced datasets
- Production-ready deployment
- Interactive web interface

### **Dataset:**
- **Source:** Water Quality Monitoring Data
- **Samples:** 222 water samples
- **Features:** 54 parameters (physical, chemical, biological)
- **Target:** Multi-class water quality classification

---

## ✨ Features

### **Data Processing**
- ✅ Automated data validation and quality checks
- ✅ Missing value imputation (median for numeric, mode for categorical)
- ✅ Date feature extraction (year, month, day)
- ✅ Outlier detection and handling
- ✅ Feature scaling (Standard/MinMax/Robust)
- ✅ Categorical encoding (Label/Target/One-Hot)

### **Feature Engineering**
- ✅ Domain-specific features (BOD/COD ratio, pH deviation)
- ✅ Automated feature selection using XGBoost
- ✅ Missing value indicators
- ✅ Interaction features

### **Model Training**
- ✅ 4 Classification Algorithms:
  - Decision Tree
  - Random Forest
  - XGBoost
  - Logistic Regression
- ✅ Grid Search CV for hyperparameter tuning
- ✅ Cross-validation
- ✅ SMOTE for class imbalance handling

### **Evaluation**
- ✅ Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ Confusion matrices (counts + percentages)
- ✅ ROC curves for all classes
- ✅ Feature importance analysis
- ✅ Model comparison visualizations

### **Deployment**
- ✅ Streamlit web application
- ✅ Real-time predictions
- ✅ Batch predictions from CSV
- ✅ Model performance dashboard

---

## 📁 Project Structure

```
water-quality-classification/
│
├── config/
│   ├── config.yaml              # Main configuration
│   └── model_params.yaml        # Model hyperparameters
│
├── core/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading
│   ├── data_validation.py       # Data quality checks
│   ├── data_preprocessing.py    # Preprocessing pipeline
│   └── imbalance_handler.py     # SMOTE for class imbalance
│
├── features/
│   ├── __init__.py
│   ├── feature_engineering.py   # Feature creation
│   └── feature_selection.py     # Feature selection
│
├── models/
│   ├── __init__.py
│   ├── base_model.py            # Abstract base class
│   ├── decision_tree_model.py   # Decision Tree
│   ├── random_forest_model.py   # Random Forest
│   ├── xgboost_model.py         # XGBoost
│   └── logistic_regression_model.py  # Logistic Regression
│
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Model training
│   └── hyperparameter_tuner.py  # Grid Search CV
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # Metrics calculation
│   └── model_evaluator.py       # Evaluation & visualization
│
├── ui/
│   ├── __init__.py
│   ├── components.py            # UI components
│   └── model_interface.py       # Prediction interface
│
├── data/
│   ├── raw/                     # Original data
│   └── processed/               # Processed data
│
├── saved_models/                # Trained models
│   ├── decision_tree/
│   ├── random_forest/
│   ├── xgboost/
│   └── logistic_regression/
│
├── reports/                     # Outputs
│   ├── figures/                 # Visualizations
│   └── metrics/                 # Performance metrics
│
├── notebooks/                   # Jupyter notebooks
│
├── main.py                      # Training pipeline
├── app.py                       # Streamlit web app
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 💻 Usage

### **1. Data Preparation**
Place your water quality CSV file in `data/raw/` directory.

```bash
cp your_data.csv data/raw/Project\ file.csv
```

### **2. Configure Settings**
Edit `config/config.yaml` to customize:
- Data paths
- Feature engineering options
- Model selection
- Training parameters

```yaml
# Example: Enable SMOTE for imbalanced data
training:
  use_smote: true

# Enable hyperparameter tuning
training:
  hyperparameter_tuning:
    enabled: true
```

### **3. Train Models**
Run the complete training pipeline:

```bash
python main.py
```

This will:
1. Load and validate data
2. Preprocess and engineer features
3. Handle class imbalance (if enabled)
4. Train all models
5. Tune hyperparameters (if enabled)
6. Evaluate and save results

**Output:**
- Trained models → `saved_models/`
- Performance metrics → `reports/metrics/`
- Visualizations → `reports/figures/`

### **4. Run Web Application**
Launch the Streamlit web interface:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## 📊 Model Performance

### **Baseline Models (Default Parameters)**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 0.82 | 0.81 | 0.82 | 0.81 |
| Random Forest | 0.89 | 0.88 | 0.89 | 0.91 |
| XGBoost | 0.94 | 0.90 | 0.93 | 0.903 | 0.94 |
| Logistic Regression | 0.76 | 0.75 | 0.76 | 0.75 |

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Decision Tree       | 0.83     | 0.86      | 0.83   | 0.85     |
| Random Forest       | 0.92     | 0.86      | 0.92   | 0.89     |
| XGBoost             | 0.94     | 0.93      | 0.93   | 0.94     |
| Logistic Regression | 0.94     | 0.92      | 0.94   | 0.93     |

**Note:** Performance may vary based on data split and preprocessing options.

### **Key Findings:**
- **Best Model:** XGBoost (Accuracy: 91%)
- **Most Important Features:** pH, Dissolved O2, BOD, COD, Temperature
- **Class Imbalance:** Present - SMOTE recommended
- **Optimal Features:** 20-25 features selected by XGBoost

---

## ⚙️ Configuration

### **Main Configuration** (`config/config.yaml`)

```yaml
# Data settings
data:
  raw_data_path: "data/raw/Project file.csv"
  target_column: "Use Based Class"
  train_test_split: 0.2

# Feature engineering
features:
  scaling_method: "standard"
  categorical_encoding: "target"
  feature_selection:
    enabled: true
    n_features: 20

# Training
training:
  use_smote: false
  hyperparameter_tuning:
    enabled: false
```

### **Model Parameters** (`config/model_params.yaml`)

Customize hyperparameters for each model:

```yaml
random_forest:
  n_estimators: 200
  max_depth: 15
  min_samples_split: 5
  
xgboost:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.1
```

---

## 🌐 Web Application

### **Features:**
- **Home:** Project overview and model comparison
- **Make Predictions:** Upload CSV and get predictions
- **Model Evaluation:** View detailed metrics and visualizations
- **About:** Project information and methodology

---

## 📈 Results & Outputs

### **Generated Files:**

**Models:**
- `saved_models/{model_name}/model.pkl` - Baseline model
- `saved_models/{model_name}/model_tuned.pkl` - Tuned model
- `saved_models/preprocessor.pkl` - Preprocessing pipeline

**Metrics:**
- `reports/metrics/{model_name}_metrics.json` - Performance metrics

**Visualizations:**
- `reports/figures/{model_name}_confusion_matrix.png`
- `reports/figures/{model_name}_roc_curves.png`
- `reports/figures/{model_name}_feature_importance.png`
- `reports/figures/model_comparison.png`

---

## 🔬 Methodology

### **Pipeline Steps:**

1. **Data Loading** - Read CSV with proper encoding
2. **Validation** - Check data quality, missing values, outliers
3. **Preprocessing** - Clean, transform, and encode features
4. **Feature Engineering** - Create domain-specific features
5. **Imbalance Handling** - Apply SMOTE if enabled
6. **Feature Selection** - Select top N features using XGBoost
7. **Model Training** - Train all 4 models
8. **Hyperparameter Tuning** - Grid Search CV (optional)
9. **Evaluation** - Calculate metrics and create visualizations
10. **Deployment** - Save models and launch web app

### **Best Practices Implemented:**
- ✅ Stratified train-test split
- ✅ Feature scaling after split (prevent data leakage)
- ✅ Cross-validation for robust evaluation
- ✅ Separate preprocessing for train/test
- ✅ Model versioning (baseline + tuned)

---

##  Acknowledgments

- Water quality dataset providers
- scikit-learn and XGBoost communities
- Streamlit for the amazing web framework
- Open-source ML community

---

*Last Updated: January 2026*
