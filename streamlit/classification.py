import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Classification")


@st.cache_data  # cache the data to avoid reloading it every time the app is run
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris.target_names


df, target_name = load_data()

model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df["species"])

st.sidebar.header("User Input Parameters")


def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 3.4)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.3)
    input_data = [sepal_length, sepal_width, petal_length, petal_width]

    prediction = model.predict([input_data])
    predicted_species = target_name[prediction][0]
    return input_data, predicted_species

input_data, predicted_species = user_input_features()

st.subheader("User Input Parameters")
st.write(input_data)

st.subheader("Prediction")
st.write(predicted_species)