# 🧠 Fetal Health Classification

This project predicts the health status of a fetus using various biomedical features from Cardiotocogram (CTG) data. It leverages data preprocessing techniques such as **SMOTE** for class balancing, **PCA** for dimensionality reduction, and trains a **Random Forest Classifier** to achieve robust and reliable predictions.

---

## 📌 Features

- 🔍 Data Preprocessing and Scaling  
- 🔁 Class Balancing using SMOTE (Synthetic Minority Over-sampling Technique)  
- 📉 Dimensionality Reduction using PCA (Principal Component Analysis)  
- 🌲 Classification using Random Forest  
- 📊 Evaluation using Cross-Validation, Accuracy, and F1-Score  
- 📈 Visualization of performance over different PCA component sizes

---

## 📂 Dataset

The dataset used is the **Fetal Health Classification Dataset** from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification). It contains 2,126 fetal records classified into three categories:

- 1: Normal  
- 2: Suspect  
- 3: Pathological  

Each record includes 21 features like FHR baseline, accelerations, uterine contractions, etc.

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Matplotlib  

---

## 📊 Model Workflow

1. **Load Dataset**  
   Load CTG data and separate features and labels.

2. **Standardize Features**  
   Apply `StandardScaler` to normalize data for optimal PCA performance.

3. **Handle Class Imbalance**  
   Use `SMOTE` to synthetically balance the target classes.

4. **Dimensionality Reduction**  
   Apply PCA to reduce features from 21 to a range of [1, 21] components.

5. **Train Model**  
   Train a Random Forest Classifier for each number of components and evaluate performance.

6. **Evaluation**  
   Metrics used: Accuracy, F1 Score, Cross-Validation Accuracy

7. **Visualization**  
   Plot the results to identify optimal number of PCA components.

---

## 📌 How to Run

### 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fetal-health-classification.git
   cd fetal-health-classification
