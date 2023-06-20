import streamlit as st
import googlemaps

# Set your Google Maps API key
api_key = "AIzaSyDxj_PmiLCF_7FXEgDQOIIdesr9FyEkAgg"

# Create a Google Maps client
gmaps = googlemaps.Client(key=api_key)

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
