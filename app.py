import pickle

import numpy as np
import streamlit as st

st.title("Heart Attack Prediction")
model = pickle.load(open("model.pkl", "rb"))

age = st.number_input("Age")
sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
cp = st.selectbox("Chest Pain Type (1: typical angina(all criteria present) 2: atypical angina (two of three criteria satisfied) -- 3: non-anginal pain (less than one criteria satisfied) -- 4: asymptomatic (none of the criteria are satisfied))"[0,1,2,3,4])
trtbps = st.number_input("Resting Blood Pressure (mmHg)")
chol = st.number_input("Cholestoral (mm/dl)")
fbs = st.selectbox("Fasting Blood Sugar (if >120 = 1, else = 0)"[0,1])
restecg = st.selectbox("Resting Electrocardiographic  Value 0: normal -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria"[0,1])
thalachh = st.number_input("Max Heart Rate Achieved")
exng = st.number_input("Exercise Induced Angina (1 = yes, 0 = no)"[0,1])
oldpeak = st.number_input("oldpeak : ST depression induced by exercise relative to rest (in mm, achieved by subtracting the lowest ST segment points during exercise and rest)")
slp = st.number_input("slope : the slope of the peak exercise ST segment, ST-T abnormalities are considered to be a crucial indicator for identifying presence of ischaemia -- Value 1: upsloping -- Value 2: flat -- Value 3: downsloping"[1,2,3])
caa = st.selectboxt("ca : number of major vessels (0-3) colored by fluoroscopy. Major cardial vessels are as goes: aorta, superior vena cava, inferior vena cava, pulmonary artery (oxygen-poor blood --> lungs), pulmonary veins (oxygen-rich blood --> heart), and coronary arteries (supplies blood to heart tissue)"[0,1,2,3])
thall = st.selectbox("thal : 0 = normal; 1 = fixed defect (heart tissue can't absorb thallium both under stress and in rest); 2 = reversible defect (heart tissue is unable to absorb thallium only under the exercise portion of the test) "[0,1,2])

btn = st.button("predict")

if btn:
    pred = model.predict(np.array([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]).reshape(1, -1))
    st.write(f"Your Heart Attack Prediction is (if 1 = yes, 0 = no): {pred}")
