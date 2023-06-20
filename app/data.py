import streamlit as st
from streamlit_folium import folium_static
import folium

import googlemaps
import pandas as pd
import numpy as np

import geopandas as gpd

import math
import matplotlib.pyplot as plt

import pyproj
from functools import partial

from shapely.ops import transform
from shapely.geometry import Point


from ipywidgets import *

from scipy.spatial import Voronoi, voronoi_plot_2d
from geovoronoi import voronoi_regions_from_coords

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



import random
import randomcolor

import pickle
import gzip


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

# #Introduction

# row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
# with row0_1:
#     st.title('BuLiAn - Bundesliga Analyzer')
# with row0_2:
#     st.text("")
#     st.subheader('Streamlit App by [Tim Denzler](https://www.linkedin.com/in/tim-denzler/)')
# row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
# with row3_1:
#     st.markdown("Hello there! Have you ever spent your weekend watching the German Bundesliga and had your friends complain about how 'players definitely used to run more' ? However, you did not want to start an argument because you did not have any stats at hand? Well, this interactive application containing Bundesliga data from season 2013/2014 to season 2019/2020 allows you to discover just that! If you're on a mobile device, I would recommend switching over to landscape for viewing ease.")
#     st.markdown("You can find the source code in the [BuLiAn GitHub Repository](https://github.com/tdenzl/BuLiAn)")
#     st.markdown("If you are interested in how this app was developed check out my [Medium article](https://tim-denzler.medium.com/is-bayern-m%C3%BCnchen-the-laziest-team-in-the-german-bundesliga-770cfbd989c7)")


# Menghitung kolom 'is_updated' berdasarkan kondisi
data_skul['is_updated']=(data_skul.last_sync!='-').astype(int)

# Menghitung persentase pembaruan data berdasarkan provinsi
data_update_skul = data_skul.groupby(['provinsi', 'is_updated']).size().unstack()
data_update_skul['0_pct'] = data_update_skul[0] / data_update_skul.sum(axis=1)
data_update_skul['1_pct'] = data_update_skul[1] / data_update_skul.sum(axis=1)

# Filter data yang telah diperbarui
df_updated_skul = data_skul[data_skul.is_updated == 1]

# Mengubah tipe data kolom 'last_sync' menjadi datetime
df_updated_skul['last_sync'] = pd.to_datetime(df_updated_skul['last_sync'])

# Mendapatkan tanggal dari kolom 'last_sync'
df_updated_skul['last_sync_date'] = df_updated_skul.last_sync.apply(lambda x: x.date())

count_bp = data_skul.groupby('bp').size()
df_skul_count = pd.DataFrame({'skul_count': count_bp, 'skul_pct': count_bp / count_bp.sum()}).sort_values('skul_count', ascending=False)

# Menggabungkan data menggunakan merge
data_merge_map = pd.merge(data_skul, map_reference[['kab_kota', 'NAME_2']], on='kab_kota', how='left')
data_merge_map = pd.merge(data_merge_map, map_df[['NAME_2', 'NAME_1', 'geometry']], on='NAME_2', how='left')

# Konversi dataframe menjadi Geopandas
data_merge_map = gpd.GeoDataFrame(data_merge_map, geometry='geometry')

# Menerapkan fungsi 'create_points_column' untuk membuat kolom 'Point'
data_merge_map['Point'] = data_merge_map.apply(create_points_column, axis=1)

# Membaca data GeoDataFrame
df_geo_chosen = pd.read_csv(PATH_INTERIM + '20191107_radius_data.csv')
df_geo_chosen['geometry'] = df_geo_chosen['geometry'].apply(geom_transform)
df_geo_chosen = gpd.GeoDataFrame(df_geo_chosen, geometry=df_geo_chosen['geometry'])

# Inisialisasi list kosong untuk menyimpan data
data_merge_map2 = []

## looping the inside polygon
data_merge_map2=[]
data_merge_name = data_merge_map.NAME_2
data_merge = data_merge_name.dropna().unique()

for city_pp in data_merge:
    if(city_pp.split(' ')[0] != 'Jakarta'):
        continue
    try:
        data_skul_p=data_merge_map[(data_merge_name==city_pp)]
        XX=data_skul_p['Point'].values
        XX=gpd.GeoDataFrame({'geometry':XX})
        data_skul_p['inside_polyg']=XX.within(map_df[map_df.NAME_2==city_pp].geometry.values[0]).values
        data_skul_p['inside_polyg']=data_skul_p['inside_polyg'].astype(int)
        data_merge_map2.append(data_skul_p)
    except:
        print(city_pp)
data_merge_map2=pd.concat(data_merge_map2)

# Menyimpan objek ke dalam file .pkl
# with open(PATH_PROCESSED + 'data.pkl', 'wb') as f:
#     pickle.dump(data_merge_map2, f)


# Simpan data ke file dengan kompresi gzip
with gzip.open(PATH_PROCESSED + 'data.pkl.gz', 'wb') as f:
    pickle.dump(data_merge_map2, f)