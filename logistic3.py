import streamlit as st
import joblib
import pandas as pd


# Load the model
model = joblib.load('Logistic3.pkl')

# Define feature names
feature_names = ["APTT", "ALB", "Neutrophils", "BUN", "DBIL", "Fg", "PT", "ALP", "DBP", "HCO3"]

# Streamlit user interface
st.title("COVID-19 Subphenotype Classifier")

# APTT: numerical input
APTT = st.number_input("APTT:", min_value=0.0, max_value=200.0, value=35.0, step=0.01, format="%0.2f")

# ALB: numerical input
ALB = st.number_input("ALB:", min_value=0.0, max_value=100.0, value=35.0, step=0.01, format="%0.2f")

# Neutrophils: numerical input
Neutrophils = st.number_input("Neutrophils:", min_value=0.0, max_value=50.0, value=6.0, step=0.01, format="%0.2f")

# BUN: numerical input
BUN = st.number_input("BUN:", min_value=0.0, max_value=200.0, value=5.0, step=0.01, format="%0.2f")

# DBIL: numerical input
DBIL = st.number_input("DBIL:", min_value=0.0, max_value=100.0, value=5.0, step=0.01, format="%0.2f")

# Fg: numerical input
Fg = st.number_input("Fg:", min_value=0.0, max_value=50.0, value=3.0, step=0.01, format="%0.2f")

# PT: numerical input
PT = st.number_input("PT:", min_value=0.0, max_value=100.0, value=12.0, step=0.01, format="%0.2f")

# ALP: numerical input
ALP = st.number_input("ALP:", min_value=0.0, max_value=2000.0, value=80.0, step=0.01, format="%0.2f")

# DBP: numerical input
DBP = st.number_input("DBP:", min_value=0.0, max_value=300.0, value=80.0, step=0.01, format="%0.2f")

# HCO3: numerical input
HCO3 = st.number_input("HCO3:", min_value=0.0, max_value=100.0, value=25.0, step=0.01, format="%0.2f")


# Process inputs and make predictions
# feature_values = [APTT, ALB, Neutrophils, BUN, DBIL, Fg, PT, ALP, DBP, HCO3]
# features = np.array([feature_values])
data = {"APTT": [APTT], "ALB": [ALB], "Neutrophils": [Neutrophils], "BUN":[BUN], "DBIL":[DBIL], 
        "Fg": [Fg], "PT": [PT], "ALP":[ALP], "DBP":[DBP], "HCO3":[HCO3]}
features = pd.DataFrame(data)

if st.button("Predict"):
    # Predict probabilities
    predicted_proba = model.predict_proba(features)[0]

    
    # 根据预测概率的最高值来确定预测类别（但这里我们直接根据概率阈值判断）  
    high_risk_threshold = 0.34  # 34% 的阈值  
    if predicted_proba[1] > high_risk_threshold:  # 假设模型输出的第二个概率是高风险类的概率  
        predicted_class = 1  # Subphenotype2 
    else:  
        predicted_class = 0  # Subphenotype1

    # 显示预测结果  
    text = f"Predicted Class: {'*Subphenotype 2*' if predicted_class == 1 else '*Subphenotype 1*'}"
    st.subheader(text, anchor=False)
        
    # 根据预测类别给出建议
    advice = f"Based on the model, predicted that the probability of Subphenotype 2 is *{predicted_proba[1] * 100:.1f}%*."

    st.subheader(advice, anchor=False)
