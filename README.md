# 🔐 Ransomware Detection via API Call Analysis

A machine learning pipeline for detecting ransomware based on the statistical analysis of Windows API call frequencies.

> 🎓 Project carried out at INPT (1st Year - Data Science Cycle) under the supervision of Mr. Kamal Idrissi Hamza.

---

## 📌 Overview

This project aims to identify ransomware based on **dynamic behavioral signatures**, particularly the frequency of system API calls made during execution. Using a dataset of 1,042 samples (ransomware vs goodware), we applied data cleaning, dimensionality reduction, and machine learning models to build a high-performance binary classifier.

---

## 🧠 Objective

> Predict whether a given software is **malicious (ransomware)** or **legitimate (goodware)** using only the API call counts observed in sandboxed executions.

 
---

## 📊 Dataset Description

- **Samples**: 1,042 (binary labels: Ransomware / Goodware)
- **Features**: 247 API call counts per sample
- **Target**: `Sample_Type` (converted to binary `Label`)
- Examples of APIs: `CreateFile`, `NtWriteVirtualMemory`, `CryptEncrypt`, etc.

---

## ⚙️ Pipeline Summary

### ✅ Preprocessing
- Cleaning rare and constant features
- Outlier handling & correlation filtering (threshold: 0.95)
- Standardization and export to `full_scaled.csv`

### 📈 Exploratory Data Analysis
- API frequency distributions
- Class-specific boxplots and barplots
- Correlation heatmaps (top 20 APIs)
- PCA & t-SNE visualizations

### 🔁 Feature Engineering
- Dimensionality reduction:
  - PCA (95 components)
  - AutoEncoder (5D)
- Feature selection:
  - ANOVA F-test
  - Random Forest importance
  - Mutual Information

### 🤖 Model Training & Evaluation
- **Classical models**: Random Forest, KNN, SVM
- **Advanced models**: MLP, XGBoost, DNN
- **Best result**:
  - XGBoost on full data:  
    `Accuracy: 89%` – `F1-score: 0.89` – `Precision: 93%` – `Recall: 86%`
  - Random Forest (RF-selected features):  
    `F1-score: 88.89%`

---

## 📌 Key Results

| Model        | Dataset         | F1-Score | Accuracy |
|--------------|------------------|----------|----------|
| XGBoost      | full_scaled.csv   | **0.89** | 89.00%   |
| Random Forest| RF-selected.csv   | 0.8889   | 89.47%   |
| KNN          | PCA               | 0.8661   | 85.65%   |
| MLP          | full_scaled.csv   | 0.7773   | 78.47%   |

---

## 📉 Limitations

- No temporal sequence analysis (pure frequency-based).
- Dataset size remains moderate (1,042 samples).
- Sandbox execution may introduce behavioral artifacts.

---

## 🔭 Future Work

- Use RNNs/LSTMs to model temporal API sequences.
- Augment with executable metadata (binary size, PE header, etc.).
- Real-time classification + system integration.
- Move from binary classification to multi-class malware family detection.

---

## 🛠️ Tech Stack

- Python (Jupyter, NumPy, Pandas, Matplotlib, Seaborn)
- Scikit-learn, XGBoost, Keras
- Visualizations: PCA, t-SNE, ROC, Boxplots, Barplots

---

## 👥 Author

- Reda Alilou  

---

## 📄 References

- [Scikit-learn](https://scikit-learn.org/)  
- [XGBoost](https://xgboost.readthedocs.io/)  
- [Keras](https://keras.io/)  
- [Microsoft Malware Classification Challenge](https://www.kaggle.com/competitions/malware-classification)

---
