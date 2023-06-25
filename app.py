import pickle

import numpy as np
import streamlit as st

st.title("Heart Attack Prediction")
model = pickle.load(open("model.pkl", "rb"))

age = st.number_input("Age")
sex = st.number_input("Sex (0 or 1)")
cp = st.number_input("Chest Pain Type (1-4)")
trtbps = st.number_input("Resting Blood Pressure (mmHg)")
chol = st.number_input("Cholestoral (mm/dl)")
fbs = st.number_input("Fasting Blood Sugar (if >120 = 1, else = 0)")
restecg = st.number_input("Resting Electrocardiographic (0-2)")
thalachh = st.number_input("Max Heart Rate Achieved")
exng = st.number_input("Exercise Induced Angina (1 = yes, 0 = no)")
oldpeak = st.number_input("oldpeak")
slp = st.number_input("slp")
caa = st.number_input("caa")
thall = st.number_input("thall")

btn = st.button("predict")

if btn:
    pred = model.predict(np.array([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]).reshape(1, -1))
    st.write(f"Your Heart Attack Prediction is (if 1 = yes, 0 = no): {pred}")
