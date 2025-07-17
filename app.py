import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("xgboost_model.pkl")

st.set_page_config(page_title="Salary Prediction", layout="centered")

st.title("💼 Salary Prediction App")
st.write("Enter the details below to predict whether income is >50K or <=50K.")

# Input fields for user data
age = st.number_input("Age", min_value=17, max_value=90, value=30)
education_num = st.slider("Education Level (Numeric)", 1, 16, 10)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)

# Example categorical encodings (replace with your real encoding if applicable)
workclass = st.selectbox("Workclass", ["Private", "Self-emp", "Gov", "Other"])
occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Other"])
sex = st.selectbox("Sex", ["Male", "Female"])

# Convert categorical to numeric (dummy example)
workclass_map = {"Private": 0, "Self-emp": 1, "Gov": 2, "Other": 3}
occupation_map = {"Tech-support": 0, "Craft-repair": 1, "Sales": 2, "Exec-managerial": 3, "Other": 4}
sex_map = {"Male": 0, "Female": 1}

# Final feature vector
features = pd.DataFrame([[
    age,
    education_num,
    hours_per_week,
    workclass_map[workclass],
    occupation_map[occupation],
    sex_map[sex]
]], columns=["age", "education_num", "hours_per_week", "workclass", "occupation", "sex"])

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(features)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"💰 Predicted Income: {result}")
