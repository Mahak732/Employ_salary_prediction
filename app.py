import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgboost_model.pkl")

st.set_page_config(page_title="Salary Prediction", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Enter the details below to predict whether income is >50K or <=50K.")

# Input fields for all features
age = st.number_input("Age", min_value=17, max_value=90, value=30)
workclass = st.selectbox("Workclass", ["Private", "Self-emp", "Gov", "Other"])
education_num = st.slider("Education Level (Numeric)", 1, 16, 10)
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Other"])
occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Other"])
relationship = st.selectbox("Relationship", ["Husband", "Not-in-family", "Own-child", "Unmarried", "Other"])
race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
native_country = st.selectbox("Native Country", ["United-States", "India", "Mexico", "Philippines", "Other"])

# Dummy encoding maps (replace these with your actual training encodings)
workclass_map = {"Private": 0, "Self-emp": 1, "Gov": 2, "Other": 3}
marital_map = {"Married": 0, "Single": 1, "Divorced": 2, "Widowed": 3, "Other": 4}
occupation_map = {"Tech-support": 0, "Craft-repair": 1, "Sales": 2, "Exec-managerial": 3, "Other": 4}
relationship_map = {"Husband": 0, "Not-in-family": 1, "Own-child": 2, "Unmarried": 3, "Other": 4}
race_map = {"White": 0, "Black": 1, "Asian-Pac-Islander": 2, "Amer-Indian-Eskimo": 3, "Other": 4}
gender_map = {"Male": 0, "Female": 1}
country_map = {"United-States": 0, "India": 1, "Mexico": 2, "Philippines": 3, "Other": 4}

# Create input DataFrame with correct column names
features = pd.DataFrame([[
    age,
    workclass_map[workclass],
    education_num,
    marital_map[marital_status],
    occupation_map[occupation],
    relationship_map[relationship],
    race_map[race],
    gender_map[gender],
    capital_gain,
    capital_loss,
    hours_per_week,
    country_map[native_country]
]], columns=[
    "age", "workclass", "educational-num", "marital-status", "occupation",
    "relationship", "race", "gender", "capital-gain", "capital-loss",
    "hours-per-week", "native-country"
])

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(features)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"💰 Predicted Income: {result}")

