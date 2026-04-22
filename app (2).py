import os
import pickle
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Titanic_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


st.write("Enter customer details:")

Class = st.number_input("Class", 0.0, 3.0, 1.0)
Age = st.number_input("Age", 0.0, 100.0, 50.0)
SibSp = st.number_input("Sibling/Spouse Count", 0.0, 10.0, 0.0)
Parch = st.number_input("Parent/Child Count", 0.0,10.0,0.0)
Fare = st.number_input("Fare",0.0,300.0,10.0)

sex = st.selectbox(
    "Sex",
    ["male","female"]
)

# Define the custom threshold
SURVIVE_THRESHOLD = 0.5
st.title("Customer Survivor Predictor")
st.write("Threshold:", SURVIVE_THRESHOLD)
if st.button("Predict Survival"):

    input_df = pd.DataFrame([{
        "Class": Class,
        "Age": Age,
        "Sibling/Spouse Count": SibSp,
        "Parent/Child Count": Parch,
        "Fare": Fare,
        "Sex": sex
    }])

    # Get probability for the positive class (churn=1)
    prob = model.predict_proba(input_df)[0][1]

    # Apply the custom threshold for prediction
    pred = 1 if prob >= SURVIVE_THRESHOLD else 0

    label = "Will Survive" if pred == 1 else "Will Not Survive"

    st.subheader(label)
    st.write(f"Survival Probability: {prob:.2f}")
