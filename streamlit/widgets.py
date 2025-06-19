import streamlit as st
import pandas as pd

st.title("Streamlit Text Input")

name = st.text_input("Enter your name")

# number params: start, end, default
age=st.slider("Select your age", 0, 100, 25)

st.write(f"Your age is {age}")

options = st.selectbox("Select your favorite color", ["Red", "Green", "Blue"])

st.write(f"Your favorite color is {options}")

if name:
    st.write(f"Hello, {name}!")

data = {
    "name": name,
    "age": age,
    "color": options
}

df = pd.DataFrame(data, index=[0])
df.to_csv("sample_data.csv", index=False)
st.write(df)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
else:
    st.write("No file uploaded")