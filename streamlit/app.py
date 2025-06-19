import streamlit as st
import pandas as pd
import numpy as np

st.title("My First Streamlit App")
st.write("This is a simple text")

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40],
})

st.write("Here is the dataframe")
st.write(df)

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)

st.write("Here is the line chart")
st.line_chart(chart_data)