import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgboost_model.pkl")

# Page config
st.set_page_config(page_title="Salary Prediction App", layout="wide")
st.markdown("# Salary Prediction App")
st.markdown("Predict if a person earns **>50K or <=50K** based on their profile.")

# Sidebar for input
st.sidebar.header("📝 Input Features")

# Input fields
age = st.sidebar.number_input("Age", min_value=17, max_value=90, value=30)

workclass = st.sidebar.selectbox("Workclass", ["Private", "Self-emp", "Gov", "Other"])

education_level = st.sidebar.selectbox("Education Level", [
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
    "HS-grad", "Some-college", "Assoc-acdm", "Assoc-voc", "Bachelors", "Masters", "Doctorate"
])
education_map = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
    "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-acdm": 11,
    "Assoc-voc": 12, "Bachelors": 13, "Masters": 14, "Doctorate": 16
}
educational_num = education_map[education_level]

marital_status = st.sidebar.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Other"])
occupation = st.sidebar.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
    'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
    'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
    'Armed-Forces'])
relationship = st.sidebar.selectbox("Relationship", ["Husband", "Not-in-family", "Own-child", "Unmarried", "Other"])
race = st.sidebar.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
native_country = st.sidebar.selectbox("Native Country", ["United-States", "India", "Mexico", "Philippines", "Other"])

has_investment_income = st.sidebar.radio("Investment Income?", ["No", "Yes"])
capital_gain, capital_loss = 0, 0
if has_investment_income == "Yes":
    capital_gain = st.sidebar.number_input("Capital Gain (₹)", min_value=0, max_value=100000, value=5000)
    capital_loss = st.sidebar.number_input("Capital Loss (₹)", min_value=0, max_value=100000, value=0)

hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)

# Encoding maps
workclass_map = {"Private": 0, "Self-emp": 1, "Gov": 2, "Other": 3}
marital_map = {"Married": 0, "Single": 1, "Divorced": 2, "Widowed": 3, "Other": 4}
occupation_map = {
    'Tech-support': 0,
    'Craft-repair': 1,
    'Other-service': 2,
    'Sales': 3,
    'Exec-managerial': 4,
    'Prof-specialty': 5,
    'Handlers-cleaners': 6,
    'Machine-op-inspct': 7,
    'Adm-clerical': 8,
    'Farming-fishing': 9,
    'Transport-moving': 10,
    'Priv-house-serv': 11,
    'Protective-serv': 12,
    'Armed-Forces': 13
}

relationship_map = {"Husband": 0, "Not-in-family": 1, "Own-child": 2, "Unmarried": 3, "Other": 4}
race_map = {"White": 0, "Black": 1, "Asian-Pac-Islander": 2, "Amer-Indian-Eskimo": 3, "Other": 4}
gender_map = {"Male": 0, "Female": 1}
country_map = {"United-States": 0, "India": 1, "Mexico": 2, "Philippines": 3, "Other": 4}

# Input as DataFrame
features = pd.DataFrame([[ 
    age,
    workclass_map[workclass],
    educational_num,
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

# Prediction and result
if st.button("🔍 Predict Salary"):
    prediction = model.predict(features)[0]
    result = ">50K" if prediction == 1 else "<=50K"

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.success(f"### 💰 Predicted Salary: **{result}**")

    if capital_gain > 0:
        st.info(f"📈 Detected capital gain of ₹{capital_gain} — likely influencing prediction.")



