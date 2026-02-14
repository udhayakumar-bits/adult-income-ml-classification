import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report

# Title
st.title("Adult Income Classification App")

st.write("Upload test dataset and select model to predict income category.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Model selection
model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", 
     "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    # Replace missing values
    df.replace(" ?", np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Encode target
    df['income'] = df['income'].apply(lambda x: 1 if ">50K" in x else 0)
    
    X = df.drop("income", axis=1)
    y = df["income"]
    
    X = pd.get_dummies(X, drop_first=True)
    
    # Load scaler
    scaler = joblib.load("model/scaler.pkl")
    
    # Scale for logistic and KNN
    if model_choice in ["Logistic Regression", "KNN"]:
        X = scaler.transform(X)
    
    # Load selected model
    if model_choice == "Logistic Regression":
        model = joblib.load("model/logistic_model.pkl")
    elif model_choice == "Decision Tree":
        model = joblib.load("model/dt_model.pkl")
    elif model_choice == "KNN":
        model = joblib.load("model/knn_model.pkl")
    elif model_choice == "Naive Bayes":
        model = joblib.load("model/nb_model.pkl")
    elif model_choice == "Random Forest":
        model = joblib.load("model/rf_model.pkl")
    else:
        model = joblib.load("model/xgb_model.pkl")
    
    # Predictions
    y_pred = model.predict(X)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:,1]
    else:
        y_prob = None
    
    # Metrics
    st.subheader("Evaluation Metrics")
    
    st.write("Accuracy:", accuracy_score(y, y_pred))
    
    if y_prob is not None:
        st.write("AUC:", roc_auc_score(y, y_prob))
    
    st.write("Precision:", precision_score(y, y_pred))
    st.write("Recall:", recall_score(y, y_pred))
    st.write("F1 Score:", f1_score(y, y_pred))
    st.write("MCC:", matthews_corrcoef(y, y_pred))
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.write(cm)
