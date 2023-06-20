import streamlit as st
from streamlit_folium import folium_static
import folium

import googlemaps
import pandas as pd
import numpy as np
import pickle
import geopandas as gpd

import math
import matplotlib.pyplot as plt

from functools import partial

from shapely.ops import transform
from shapely.geometry import Point


from ipywidgets import *
import gzip
from scipy.spatial import Voronoi, voronoi_plot_2d
from geovoronoi import voronoi_regions_from_coords

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



import random
import randomcolor

st.set_page_config(layout="wide")

def get_points_to_poly_assignments(poly_to_pt_assignments):
    """
    Reverse of poly to points assignments: Returns a list of size N, which is the number of unique points in
    `poly_to_pt_assignments`. Each list element is an index into the list of Voronoi regions.
    """
    print(poly_to_pt_assignments)
    pt_poly = [(i_pt, i_vor)
               for i_vor, i_pt in enumerate(poly_to_pt_assignments)]

    return [i_vor for _, i_vor in sorted(pt_poly, key=lambda x: x[0])]

# Fungsi untuk membuat kolom 'Point' geometri
def create_points_column(row):
    y = row['lintang']
    x = row['bujur']
    return Point(x, y)

# Fungsi transformasi geometri
def geom_transform(x):
    x, y = x.split('POINT')[-1].strip()[1:-1].split()
    x, y = float(x), float(y)
    return Point(x, y)

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Menghitung jarak antara dua titik koordinat menggunakan rumus Haversine.
    """
    R = 6371  # Radius bumi dalam kilometer
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

#Path (lokasi) direktori yang digunakan dalam suatu proyek untuk menyimpan data
PATH_PROCESSED='data/processed/'
PATH_INTERIM='data/interim/'
PATH_MODEL="models/"

#menggunakan pandas untuk membaca file CSV
data_skul=pd.read_csv(PATH_PROCESSED+'20190901_all_sekolah_genap2019_data_latlong.csv')
map_reference=pd.read_csv(PATH_INTERIM+'reference_our_data_to_map_used.csv')
map_reference.columns=['kab_kota','NAME_2','Lev_Ratio','Inter_Ratio','Rank','Rank2']

fp = PATH_MODEL+'gadm36_IDN_shp/gadm36_IDN_2.shp'
map_df = gpd.read_file(fp)


# Set your Google Maps API key
api_key = "AIzaSyDxj_PmiLCF_7FXEgDQOIIdesr9FyEkAgg"

# Create a Google Maps client
gmaps = googlemaps.Client(key=api_key)

# Muat data dari file dengan kompresi gzip
with gzip.open(PATH_PROCESSED + 'data.pkl.gz', 'rb') as f:
    data_merge_map2 = pickle.load(f)

# Get SMA only
df_cls = data_merge_map2[(data_merge_map2.bp.str.lower().str.contains('sma'))
                        & (data_merge_map2.inside_polyg == 1)][['NAME_2', 'Point', 'bujur', 'lintang']]

# Daftar kabupaten/kota yang dapat dipilih
options = ['Jakarta Barat', 'Jakarta Pusat', 'Jakarta Utara', 'Jakarta Timur', 'Jakarta Selatan']
kab_n = st.selectbox('Pilih Kabupaten/Kota', options)

# Memfilter data berdasarkan kabupaten/kota
df_cls_kab = df_cls[df_cls.NAME_2 == kab_n]


XX = df_cls_kab[['bujur', 'lintang']]
XX = (XX - XX.mean()) / XX.std()

data_eval = []
n = len(XX)  # Jumlah total sekolah di kabupaten/kota
max_cluster_number = n // 3  # Maksimum jumlah klaster

for k0 in range(3, max_cluster_number + 1):
    km0 = KMeans(n_clusters=k0, random_state=456)
    cls_result = km0.fit_predict(XX)
    if min(np.bincount(cls_result)) >= 3:
        silscor = silhouette_score(XX, cls_result)
        data_eval.append([k0, silscor])

df_data_eval = pd.DataFrame(data_eval, columns=['k', 'sils'])
df_data_eval.set_index('k', inplace=True)
# st.write(df_data_eval)
k0_max = df_data_eval['sils'].idxmax()
k0_max = max(6, 3)  # Pastikan k0_max >= 3

real_KM0 = KMeans(n_clusters=k0_max, random_state=456)

XX0 = df_cls_kab[['bujur', 'lintang']]
XX0 = (XX0 - XX.mean()) / XX.std()
real_cls = real_KM0.fit_predict(XX0)
df_cls_kab['cls'] = real_cls

number_of_colors = len(df_cls_kab.cls.unique())
rand_color = randomcolor.RandomColor()
color_voronoi = rand_color.generate(count=number_of_colors)

df_colvo = pd.DataFrame({'cls': df_cls_kab.cls.unique(), 'colvo': color_voronoi})
df_cls_kab = pd.merge(df_cls_kab, df_colvo, on='cls', how='left')
df_cls_kab = df_cls_kab.sort_values('cls')
df_cls_kab = df_cls_kab.reset_index(drop=True)

# Plot Voronoi diagram for all schools in the kabupaten/kota
vor = Voronoi(XX.values)
plt.figure(figsize=(8, 8))
voronoi_plot_2d(vor, show_vertices=False, line_colors='gray', line_width=0.5, line_alpha=0.5)

# Plot Voronoi regions based on clusters
for i, point in enumerate(XX.values):
    regions = np.nonzero(vor.point_region == i)[0]
    if regions.size > 0:
        region = regions[0]
        cluster_label = df_cls_kab['cls'].iloc[i]
        color = df_cls_kab['colvo'].iloc[i]
        polygon = [vor.vertices[j] for j in vor.regions[region] if j != -1]
        if len(polygon) >= 3:  # Hanya gambar region dengan minimal 3 titik
            plt.fill(*zip(*polygon), color=color, alpha=0.3)

# Mendapatkan data peta berdasarkan kabupaten/kota
map_df_v = map_df[map_df.NAME_2 == kab_n]

poly_v = map_df_v['geometry'].values[0]

df_cls_kab = df_cls_kab.drop_duplicates(subset = ['bujur', 'lintang'])
poly_shapes, poly_to_pt_assignments = voronoi_regions_from_coords(df_cls_kab[['bujur', 'lintang']].values, poly_v)

df_cls_kab['col_sort'] = get_points_to_poly_assignments(list(poly_to_pt_assignments.values()))
df_cls_kab['vor'] = df_cls_kab.col_sort.apply(lambda x: poly_shapes[x])
cte = df_cls_kab.colvo.unique()





desc, plots, mapping = st.columns((0.3, 0.4, 0.3))

with plots:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#000000')

    for cc in df_cls_kab.cls.unique():
        geom_v = df_cls_kab[df_cls_kab.cls == cc]['vor']
        geo_vor = gpd.GeoDataFrame(df_cls_kab[['bujur', 'lintang']], geometry=geom_v)
        geo_vor.plot(ax=ax, markersize=20, color=cte[cc], edgecolor='grey')

    dsp2 = data_merge_map2[(data_merge_map2.bp.str.lower().str.contains('sma'))
                           & (data_merge_map2.inside_polyg == 1)
                           & (data_merge_map2.NAME_2 == kab_n)]
    # st.write(dsp2)
    XX_v = dsp2[['bujur', 'lintang']]
    XX_v = (XX_v - XX_v.mean()) / XX_v.std()
    dsp2['cls'] = real_KM0.fit_predict(XX_v)

    geom1 = dsp2['Point']
    geo_data_point = gpd.GeoDataFrame(dsp2, geometry=geom1)
    geo_data_point.plot(ax=ax, markersize=20, color='black', label='SMA')

    # Label kecamatan
    for idx, row in dsp2.iterrows():
        ax.annotate(row['nama_sekolah'], (row['bujur'], row['lintang']), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize= 4)

    plt.title(kab_n + "\nVoronoi Zoning School", {'fontsize': 20})
    st.pyplot(fig)
with desc:
    st.write("desc")
with mapping:
    # Create a Folium map centered at the mean latitude and longitude
    map_center = [df_cls_kab['lintang'].mean(), df_cls_kab['bujur'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    # Add the Voronoi polygons as GeoJSON to the map
    for idx, row in df_cls_kab.iterrows():
        if 'colvo' in row:
            color = row['colvo']
        else:
            color = f"#{random.randint(0, 0xFFFFFF):06x}"  # Generate random color if colvo not available
        folium.GeoJson(row['vor'], style_function=lambda x: {'fillColor': color, 'color': 'grey'}).add_to(m)

    # Add the school locations as markers to the map
    for idx, row in dsp2.iterrows():
        folium.Marker(location=[row['lintang'], row['bujur']], popup=row['nama_sekolah']).add_to(m)

    # Display the map
    folium_static(m)

#Start---------------------------------------

# Daftar nama sekolah, warna, dan koordinat dalam satu clustering
clustered_schools = dsp2.groupby('cls')

# Koordinat lokasi yang dicari
search_latitude = None
search_longitude = None

for cluster, schools in clustered_schools:
    color = cte[cluster]
    school_list = schools[['nama_sekolah', 'lintang', 'bujur', 'status']].values
    for school in school_list:
        school_name = school[0]
        latitude = school[1]
        longitude = school[2]
        status = school[3]
#         st.write(f"Nama Sekolah: {school_name}, Warna: {color}, Koordinat: ({latitude}, {longitude}), Status: {status}")

        # Cek jarak dengan lokasi yang dicari
        if search_latitude is not None and search_longitude is not None:
            distance = calculate_distance(search_latitude, search_longitude, latitude, longitude)
#             st.write(f"Jarak dari lokasi: {distance} km")

if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

def callback():
    st.session_state.button_clicked = True

# Input location name
location_name = st.text_input("Masukkan nama lokasi:", key ='Text1')

if st.button("Cari", on_click = callback)or st.session_state.button_clicked :
    # Geocode the location name using Google Maps Geocoding API
    geocode_result = gmaps.geocode(location_name)
    if len(geocode_result) > 0:
        # Get the latitude and longitude from the geocoding result
        latitude = geocode_result[0]['geometry']['location']['lat']
        longitude = geocode_result[0]['geometry']['location']['lng']
        search_latitude = latitude
        search_longitude = longitude

        # Create a Point object for the searched location
        point = Point(longitude, latitude)

        # Find the cluster containing the location point
        cluster_containing_location = None
        for idx, row in df_cls_kab.iterrows():
            if row['vor'].contains(point):
                cluster_containing_location = row['cls']
                break

        if cluster_containing_location is not None:
            map_center = [latitude, longitude]
            m = folium.Map(location=map_center, zoom_start=12)

            # Add the Voronoi polygons as GeoJSON to the map
            for idx, row in df_cls_kab.iterrows():
                if 'colvo' in row:
                    color = row['colvo']
                else:
                    color = f"#{random.randint(0, 0xFFFFFF):06x}"
                folium.GeoJson(row['vor'], name=row['cls'], style_function=lambda x: {'color': color, 'fillOpacity': 0.2}).add_to(m)

            # Add a red marker for the searched location on the map
            folium.Marker([latitude, longitude], popup=location_name, icon=folium.Icon(color='red')).add_to(m)

            # Find the nearest school within the cluster
            nearest_schools = []
            for cluster, schools in clustered_schools:
                color = cte[cluster]
                school_list = schools[['nama_sekolah', 'lintang', 'bujur', 'status']].values
                for school in school_list:
                    school_name = school[0]
                    latitude = school[1]
                    longitude = school[2]
                    distance = calculate_distance(search_latitude, search_longitude, latitude, longitude)
                    nearest_schools.append({'school_name': school_name, 'color': color, 'distance': distance, 'status': school[3], 'latitude': latitude, 'longitude': longitude})


            nearest_schools = pd.DataFrame(nearest_schools).sort_values(by = 'distance')
#             st.write(nearest_schools)
            same_color_negeri_schools = nearest_schools[(nearest_schools['color'] == nearest_schools['color'].unique()[0])& (nearest_schools['status'] == 'Negeri')]
#             st.write(same_color_negeri_schools)
            same_color_negeri_schools = same_color_negeri_schools.reset_index(drop = True)
            st.write("Sekolah terdekat dengan cluster yang sama:")
            tb_school_name = []
            tb_distance = []
            tb_status = []
            for school in range(len(same_color_negeri_schools)):

                school_name = same_color_negeri_schools.loc[school,'school_name']
                tb_school_name.append(school_name)
                color = same_color_negeri_schools.loc[school,'color']
                distance = same_color_negeri_schools.loc[school,'distance']
                tb_distance.append(distance)
                status = same_color_negeri_schools.loc[school,'status']
                tb_status.append(status)
                school_latitude = same_color_negeri_schools.loc[school,'latitude']
                school_longitude = same_color_negeri_schools.loc[school,'longitude']

                # Add a marker for the school on the map
                folium.Marker([school_latitude, school_longitude], popup=school_name, icon=folium.Icon(color='blue')).add_to(m)

#                 st.write(f"Nama Sekolah: {school_name},Jarak: {distance} km, Status: {status}")

            folium_static(m)
    
            df_school = pd.DataFrame(data = {
                'Nama Sekolah' : tb_school_name,
                'Jarak Sekolah' : tb_distance,
                'Status Sekolah' : tb_status
            })
            st.dataframe(df_school)
            
            
                
                
            

    else:
        print("Tidak ditemukan sekolah terdekat.")
        






