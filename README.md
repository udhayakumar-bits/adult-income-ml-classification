# Adult Income Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to build multiple machine learning classification models to predict whether an individual earns more than \$50K per year based on demographic and employment-related features.

This is a binary classification problem where:

- 0 → Income ≤ 50K  
- 1 → Income > 50K  

A Streamlit web application is also developed and deployed to allow interactive model selection and evaluation.

---

## 2. Dataset Description

- **Dataset Name:** Adult Census Income Dataset  
- **Source:** Kaggle  
- **Total Instances:** 32,561  
- **Total Input Features:** 14  
- **Target Variable:** `income`  

### Feature Examples

- age  
- workclass  
- education  
- marital-status  
- occupation  
- relationship  
- race  
- sex  
- capital-gain  
- capital-loss  
- hours-per-week  
- native-country  

The dataset contains both numerical and categorical features.  
Categorical features were converted using one-hot encoding before training.

---

## 3. Models Implemented

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

## 4. Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## 5. Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.8475 | 0.9031 | 0.7176 | 0.5836 | 0.6437 | 0.5528 |
| Decision Tree | 0.8139 | 0.7478 | 0.6023 | 0.6226 | 0.6123 | 0.4900 |
| KNN | 0.8174 | 0.8268 | 0.6291 | 0.5517 | 0.5879 | 0.4729 |
| Naive Bayes | 0.7947 | 0.8296 | 0.6397 | 0.2980 | 0.4066 | 0.3341 |
| Random Forest | 0.8518 | 0.9018 | 0.7183 | 0.6122 | 0.6610 | 0.5700 |
| XGBoost | 0.8726 | 0.9270 | 0.7773 | 0.6448 | 0.7048 | 0.6289 |

---

## 6. Observations on Model Performance

| ML Model | Observation |
|------------|--------------|
| Logistic Regression | Performs well with strong AUC score. Serves as a good baseline model with balanced overall performance. |
| Decision Tree | Easy to interpret but slightly lower performance compared to ensemble methods. |
| KNN | Moderate performance. Sensitive to scaling and feature distribution. |
| Naive Bayes | Fast and simple, but recall is relatively low, indicating difficulty in identifying high-income individuals. |
| Random Forest | Improved performance over a single decision tree. Handles feature interactions effectively. |
| XGBoost | Best performing model overall with highest Accuracy, AUC, F1 Score, and MCC. |

---

## 7. Streamlit Application Features

The deployed Streamlit application includes:

- CSV file upload option  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix display  

The app allows interactive testing of different classification models on uploaded test data.

---

## 8. Project Structure

```
adult-income-ml-classification/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│     │-- logistic_model.pkl
│     │-- dt_model.pkl
│     │-- knn_model.pkl
│     │-- nb_model.pkl
│     │-- rf_model.pkl
│     │-- xgb_model.pkl
│     │-- scaler.pkl
```
---

## 9. Deployment

The application is deployed using **Streamlit Community Cloud** and connected to this GitHub repository.
