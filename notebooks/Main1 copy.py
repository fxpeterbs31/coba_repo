import streamlit as st

# Mengatur tema Streamlit dengan parameter `theme`
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
        background-attachment: fixed;
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<style>body{background-color: #55a0af;}</style>',unsafe_allow_html=True)

sidebar_bg = """
    <style>
    .sidebar .sidebar-content {
        background-image: url('https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg');
        background-repeat: no-repeat;
        background-size: cover;
    }
    </style>
    """
import time

import numpy as np
import pandas as pd


# import shapefile

st.empty()
my_bar = st.progress(0)
for i in range(100):
    my_bar.progress(i + 1)
    time.sleep(0.1)
n_elts = int(time.time() * 10) % 5 + 3
for i in range(n_elts):
    st.text("." * i)
st.write(n_elts)
for i in range(n_elts):
    st.text("." * i)
st.success("done")

# Menambahkan CSS ke dalam aplikasi Streamlit
st.markdown(sidebar_bg, unsafe_allow_html=True)

# Tampilan Streamlit
st.title("Aplikasi dengan Sidebar Berwarna")

# Pilihan menu di sidebar
option = st.sidebar.selectbox("Navigation", ["Home", "Dataframe", "Chart"])

# Konten sesuai dengan pilihan menu
if option == "Home":
    st.write("Ini adalah halaman utama")
elif option == "Dataframe":
    st.write("Ini adalah halaman data")
elif option == "Chart":
    st.write("Ini adalah halaman chart")
