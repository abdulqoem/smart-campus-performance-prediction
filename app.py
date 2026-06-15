import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_performance_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Student Academic Performance Prediction")
st.write("Predict student performance level using Machine Learning.")

user_input = {}

for feature in feature_names:
    if feature == "Age":
        user_input[feature] = st.number_input("Age", min_value=10, max_value=30, value=18, step=1)

    elif feature == "Class":
        user_input[feature] = st.number_input("Class", min_value=1, max_value=12, value=10, step=1)

    elif feature in encoders:
        user_input[feature] = st.selectbox(feature, list(encoders[feature].classes_))

    elif "Percentage" in feature:
        user_input[feature] = st.number_input(feature, min_value=0, max_value=100, value=80, step=1)

    elif "Score" in feature:
        user_input[feature] = st.number_input(feature, min_value=0, max_value=100, value=70, step=1)

    elif "Study_Hours" in feature:
        user_input[feature] = st.number_input(feature, min_value=0.0, max_value=24.0, value=4.0, step=0.5)

    else:
        user_input[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([user_input])
input_df = input_df[feature_names]

for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = encoders["Performance_Level"].inverse_transform([prediction])[0]

    st.success(f"Predicted Performance Level: {result}")