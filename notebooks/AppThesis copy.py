import streamlit as st
import googlemaps
import pandas as pd
import numpy as np

# CSS untuk mengubah warna background sidebar
sidebar_bg = """
<style>
.sidebar .sidebar-content {
    background-color: #55a0af;  /* Warna tosca */
}
</style>
"""

# Menambahkan CSS ke dalam aplikasi Streamlit
st.markdown(sidebar_bg, unsafe_allow_html=True)

option = st.sidebar.selectbox("Navigation", ["Home", "Dataframe", "Chart"])



# Set your Google Maps API key
api_key = "AIzaSyDxj_PmiLCF_7FXEgDQOIIdesr9FyEkAgg"

# Create a Google Maps client
gmaps = googlemaps.Client(key=api_key)

if option == 'Home' or option == '':
    # Streamlit app
    st.title("Aplikasi Peta")

    # Input location name
    location_name = st.text_input("Masukkan nama lokasi:")

    if st.button("Cari"):
        # Geocode the location name using Google Maps Geocoding API
        geocode_result = gmaps.geocode(location_name)

        if len(geocode_result) > 0:
            # Get the latitude and longitude from the geocoding result
            latitude = geocode_result[0]['geometry']['location']['lat']
            longitude = geocode_result[0]['geometry']['location']['lng']

            # Create the map URL with API key
            map_url = f"https://www.google.com/maps/embed/v1/place?q={latitude},{longitude}&key={api_key}"

            # Display the map using Google Maps Embed API
            st.markdown(f'<iframe width="100%" height="500" src="{map_url}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
        else:
            st.error("Lokasi tidak ditemukan")
elif option == 'Dataframe':
    st.write("""## Dataframe""") # Menampilkan judul halaman dataframe

    # Membuat dataframe dengan pandas yang terdiri dari 2 kolom dan 4 baris data
    df = pd.DataFrame({
        'Column 1': [1, 2, 3, 4],
        'Column 2': [10, 12, 14, 16]
    })
    df  # Menampilkan dataframe

elif option == 'Chart':
    st.write("""## Draw Charts""") # Menampilkan judul halaman

    # Membuat variabel chart data yang berisi data dari dataframe
    # Data berupa angka acak yang di-generate menggunakan numpy
    # Data terdiri dari 2 kolom dan 20 baris
    chart_data = pd.DataFrame(
        np.random.randn(20, 2),
        columns=['a', 'b']
    )
    # Menampilkan data dalam bentuk chart
    st.line_chart(chart_data)
    # Menampilkan data dalam bentuk tabel
    chart_data
