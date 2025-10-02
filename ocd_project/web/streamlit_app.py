import streamlit as st
import joblib
import pandas as pd

st.title("🧠 OCD Medication Prediction Demo")
pipeline = joblib.load("models/ocd_med_pipeline.pkl")

# Example inputs (extend as needed)
age = st.number_input("Age", min_value=5, max_value=90, value=30)
gender = st.selectbox("Gender", options=["Male","Female"])
family_history = st.radio("Family History of OCD?", ["Yes","No"])
depression = st.radio("Depression Diagnosis?", ["Yes","No"])
anxiety = st.radio("Anxiety Diagnosis?", ["Yes","No"])

if st.button("Predict Medication"):
    row = {
        "Age": age,
        "Gender": gender,
        "Family History of OCD": family_history,
        "Depression Diagnosis": depression,
        "Anxiety Diagnosis": anxiety
    }
    row_df = pd.DataFrame([row])
    row_df = pd.get_dummies(row_df)
    row_df = row_df.reindex(columns=pipeline['columns'], fill_value=0)
    x_s = pipeline['scaler'].transform(row_df)
    pred = pipeline['model'].predict(x_s)
    st.success(f"Predicted Medication: {pipeline['label_encoder_classes'][pred[0]]}")
