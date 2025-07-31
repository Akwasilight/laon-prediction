import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load model and scaler
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# User input form

st.title("🏦💶Loan Prediction App")



# Input fields

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
with col2:
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.selectbox("Loan Amount Term (in months)", [360, 180, 240, 120, 60])
    credit_history = st.selectbox("Has Credit History (1 : yes, 0 : No)", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Feature engineering
input_dict = {
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": bool(credit_history),
    "HasCoapplicant": bool(coapplicant_income > 0),
    "Gender_Male": 1 if gender == "Male" else 0,
    "Married_Yes": 1 if married == "Yes" else 0,
    "Dependents_1": 1 if dependents == "1" else 0,
    "Dependents_2": 1 if dependents == "2" else 0,
    "Dependents_3+": 1 if dependents == "3+" else 0,
    "Education_Not Graduate": 1 if education == "Not Graduate" else 0,
    "Self_Employed_Yes": 1 if self_employed == "Yes" else 0,
    "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
    "Property_Area_Urban": 1 if property_area == "Urban" else 0
}

# Fill missing columns with 0s
input_data = pd.DataFrame([input_dict])
for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[feature_columns]

# Scale numeric
input_data[["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]] = scaler.transform(
    input_data[["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]]
)

# Predict
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    result = "Approved ✅" if prediction == 1 else "Rejected ❌"
    st.success(f"Loan Status: {result}")
