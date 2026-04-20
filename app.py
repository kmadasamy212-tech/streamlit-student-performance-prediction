import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Performance Prediction App")

st.write("Enter student details to predict if they will pass or fail")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 25)
study_hours = st.slider("Study Hours per Week", 0, 50)
attendance = st.slider("Attendance Rate", 0, 100)
parent_edu = st.selectbox("Parent Education", ["None", "High School", "Bachelor", "Master", "PhD"])
internet = st.selectbox("Internet Access", ["Yes", "No"])
extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
previous_score = st.number_input("Previous Score", 0, 100)
final_score = st.number_input("Final Score", 0, 100)

# Encoding
gender = 1 if gender == "Male" else 0
internet = 1 if internet == "Yes" else 0
extracurricular = 1 if extracurricular == "Yes" else 0

edu_map = {"None":0, "High School":1, "Bachelor":2, "Master":3, "PhD":4}
parent_edu = edu_map[parent_edu]

# Prediction button
if st.button("Predict"):
    input_data = np.array([[gender, age, study_hours, attendance,
                            parent_edu, internet, extracurricular,
                            previous_score, final_score]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("🎉 Student will PASS")
    else:
        st.error("⚠️ Student may FAIL")