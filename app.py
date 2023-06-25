import pickle

import numpy as np
import streamlit as st

st.title("Flower Prediction app")
model = pickle.load(open("model.pkl", "rb"))

sep_length = st.number_input("Sepal Length")
sep_width = st.number_input("Sepal Width")
pet_length = st.number_input("Petal Length")
pet_width = st.number_input("Petal Width")

btn = st.button("predict")

if btn:
    pred = model.predict(np.array([sep_length,sep_width,pet_length,pet_width]).reshape(1,-1))
    st.write(f"Your flower species is: {pred}" )