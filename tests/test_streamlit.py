import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Test App")

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)

st.success("Streamlit is working!")