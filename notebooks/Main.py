import streamlit as st
import csv
import os

import googlemaps
# %matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd
from geopandas import GeoSeries
import geoplot.crs as gcrs
import fiona
from shapely.geometry import LineString, Polygon, Point
import numpy as np

import shapely
import descartes
import Levenshtein
from scipy.spatial import Voronoi
# import geog
import shapely.geometry
import geoplot
import matplotlib
import pyproj
from shapely.ops import transform
from shapely.geometry import Point
from functools import partial
from shapely.geometry import Polygon

from ipywidgets import *
from IPython.display import display, HTML
import ipywidgets as widgets


from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


from geovoronoi import voronoi_regions_from_coords
import random
import randomcolor

import folium
from streamlit_folium import folium_static
from geopy.distance import geodesic

# Fungsi untuk melakukan validasi email
def validate_email(email):
    if "@" in email and ".com" in email:
        return True
    return False

# Fungsi untuk melakukan validasi password
def validate_password(password):
    if any(char.isdigit() for char in password) and any(char.isalpha() for char in password):
        return True
    return False

# Fungsi untuk menyimpan data user ke file CSV
def save_user_data(name, email, password):
    filename = "user_data.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writerow(['Name', 'Email', 'Password'])
        writer.writerow([name, email, password])

# Fungsi untuk memeriksa keberadaan email dan password dalam data user
def check_user_data(email, password):
    filename = "user_data.csv"
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Lewati header
        for row in reader:
            if row[1] == email and row[2] == password:
                return True
    return False

# Halaman Register
def register_page():
    st.title("Register")

    # Input form
    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Register", key="register_button"):
        if name == "":
            st.error("Name cannot be empty.")
        elif not validate_email(email):
            st.error("Invalid email format.")
        elif not validate_password(password):
            st.error("Password must contain both letters and numbers.")
        else:
            save_user_data(name, email, password)
            st.success("Registration successful. Please proceed to login.")
            st.session_state.page = "Login"

    st.markdown("Already have an account?")

# Halaman Login
def login_page():
    st.title("Login")

    # Input form
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if not validate_email(email) or not validate_password(password):
            st.error("Invalid email or password.")
        elif not check_user_data(email, password):
            st.error("Invalid email or password.")
        else:
            st.success("Login successful. Redirecting to home page...")
            st.session_state.page = "Home"

    st.markdown("Don't have an account?")

#------------- Seharusnya ini semua dari import Geovoronoi--------------------
def get_points_to_poly_assignments(poly_to_pt_assignments):
    """
    Reverse of poly to points assignments: Returns a list of size N, which is the number of unique points in
    `poly_to_pt_assignments`. Each list element is an index into the list of Voronoi regions.
    """
    print(poly_to_pt_assignments)
    pt_poly = [(i_pt, i_vor)
               for i_vor, i_pt in enumerate(poly_to_pt_assignments)]

    return [i_vor for _, i_vor in sorted(pt_poly, key=lambda x: x[0])]

def subplot_for_map(show_x_axis=False, show_y_axis=False, show_spines=None, aspect='equal', **kwargs):
    """
    Helper function to generate a matplotlib subplot Axes object suitable for plotting geographic data, i.e. axis
    labels are not shown and aspect ratio is set to 'equal' by default.

    :param show_x_axis: show x axis labels
    :param show_y_axis: show y axis labels
    :param show_spines: controls display of frame around plot; if set to None, this is "auto" mode, meaning
                        that the frame is removed when `show_x_axis` and `show_y_axis` are both set to False;
                        if set to True/False, the frame is always shown/removed
    :param aspect: aspect ratio
    :param kwargs: additional parameters passed to `plt.subplots()`
    :return: tuple with (matplotlib Figure, matplotlib Axes)
    """
    fig, ax = plt.subplots(**kwargs)
    ax.set_aspect(aspect)

    ax.get_xaxis().set_visible(show_x_axis)
    ax.get_yaxis().set_visible(show_y_axis)

    if show_spines is None:
        show_spines = show_x_axis or show_y_axis

    for sp in ax.spines.values():
        sp.set_visible(show_spines)

    if show_x_axis:
        fig.autofmt_xdate()

    return fig, ax

def plot_voronoi_polys_with_points_in_area(ax, area_shape, region_polys, points, region_pts=None,
                                           area_color=(1,1,1,1), area_edgecolor=(0,0,0,1),
                                           voronoi_and_points_cmap='tab20',
                                           voronoi_color=None, voronoi_edgecolor=None,
                                           points_color=None, points_markersize=5, points_marker='o',
                                           voronoi_labels=None, voronoi_label_fontsize=10, voronoi_label_color=None,
                                           point_labels=None, point_label_fontsize=7, point_label_color=None,
                                           plot_area_opts=None,
                                           plot_voronoi_opts=None,
                                           plot_points_opts=None):
    """
    All-in-one function to plot Voronoi region polygons `region_polys` and the respective points `points` inside a
    geographic area `area_shape` on a matplotlib Axes object `ax`.

    By default, the regions will be blue and the points black. Optionally pass `region_pts` to show Voronoi regions and
    their respective points with the same color (which is randomly drawn from color map `voronoi_and_points_cmap`).
    Labels for Voronoi regions can be passed as `voronoi_labels`. Labels for points can be passed as `point_labels`.
    Use style options to customize the plot. Pass additional (matplotlib) parameters to the individual plotting steps
    as `plot_area_opts`, `plot_voronoi_opts` or `plot_points_opts` respectively.

    :param ax: matplotlib Axes object to plot on
    :param area_shape: geographic shape surrounding the Voronoi regions; can be None to disable plotting of geogr. shape
    :param region_polys: dict mapping region IDs to Voronoi region geometries
    :param points: NumPy array or list of Shapely Point objects
    :param region_pts: dict mapping Voronoi region IDs to point indices of `points`
    :param area_color: fill color of `area_shape`
    :param area_edgecolor: edge color of `area_shape`
    :param voronoi_and_points_cmap: matplotlib color map name used for Voronoi regions and points colors when colors are
                                    not given by `voronoi_color`
    :param voronoi_color: dict mapping Voronoi region ID to fill color or None to use `voronoi_and_points_cmap`
    :param voronoi_edgecolor: Voronoi region polygon edge colors
    :param points_color: points color
    :param points_markersize: points marker size
    :param points_marker: points marker type
    :param voronoi_labels: Voronoi region labels displayed at centroid of Voronoi region polygon
    :param voronoi_label_fontsize: Voronoi region labels font size
    :param voronoi_label_color: Voronoi region labels color
    :param point_labels: point labels
    :param point_label_fontsize: point labels font size
    :param point_label_color: point labels color
    :param plot_area_opts: options passed to function for plotting the geographic shape
    :param plot_voronoi_opts: options passed to function for plotting the Voronoi regions
    :param plot_points_opts: options passed to function for plotting the points
    :return: None
    """
    plot_area_opts = plot_area_opts or {}
    plot_voronoi_opts = plot_voronoi_opts or {'alpha': 0.5}
    plot_points_opts = plot_points_opts or {}

    if area_shape is not None:
        plot_polygon_collection_with_color(ax, [area_shape], color=area_color, edgecolor=area_edgecolor,
                                           **plot_area_opts)

    if voronoi_and_points_cmap and region_pts and (voronoi_color is None or points_color is None):
        voronoi_color, points_color = colors_for_voronoi_polys_and_points(region_polys, region_pts,
                                                                          point_indices=list(range(len(points))),
                                                                          cmap_name=voronoi_and_points_cmap)

    if voronoi_color is None and voronoi_edgecolor is None:
        voronoi_edgecolor = (0,0,0,1)   # better visible default value

    plot_voronoi_polys(ax, region_polys, color=voronoi_color, edgecolor=voronoi_edgecolor,
                       labels=voronoi_labels, label_fontsize=voronoi_label_fontsize, label_color=voronoi_label_color,
                       **plot_voronoi_opts)

    plot_points(ax, points, points_markersize, points_marker, color=points_color,
                labels=point_labels, label_fontsize=point_label_fontsize, label_color=point_label_color,
                **plot_points_opts)
#------------- Seharusnya ini semua dari import Geovoronoi--------------------

def get_school_map(kab_kot_name,lvl='all'):
    outlier=False
    split=True
    map_data=map_df.copy()
    source_data=data_merge_map.copy()
    
    if outlier:
        pass
    else:
        source_data=source_data[source_data['outlier_point']==0]
    
    data_point_example=source_data[source_data.NAME_2==kab_kot_name]
    crs={'init':'epsg:4326'}
    
    fig,ax=plt.subplots(figsize=(12,12))
    map_data[map_data.NAME_2==kab_kot_name].plot(ax=ax,alpha=0.4,color='grey')
    
    if split:
        if lvl=='all':
            for jenjang,lbl in zip(['sd','smp','sma','smk'],[('red','SD'),('blue','SMP'),('green','SMA'),('black','SMK')]):
                data_point_jenjang=data_point_example[data_point_example.bp.str.lower().str.contains(jenjang)]
                # geom=[Point(xy) for xy in zip(data_point_jenjang['bujur'],data_point_jenjang['lintang'])]
                geom=data_point_jenjang['Point']
                geo_data_point=gpd.GeoDataFrame(data_point_jenjang,crs=crs,geometry=geom)
                geo_data_point.plot(ax=ax,markersize=20,color=lbl[0],label=lbl[1])
        elif lvl=='sd':
            data_point_jenjang=data_point_example[data_point_example.bp.str.lower().str.contains(lvl)]
            # geom=[Point(xy) for xy in zip(data_point_jenjang['bujur'],data_point_jenjang['lintang'])]
            geom=data_point_jenjang['Point']
            geo_data_point=gpd.GeoDataFrame(data_point_jenjang,crs=crs,geometry=geom)
            geo_data_point.plot(ax=ax,markersize=20,color='red',label='SD')
        elif lvl=='smp':
            data_point_jenjang=data_point_example[data_point_example.bp.str.lower().str.contains(lvl)]
            # geom=[Point(xy) for xy in zip(data_point_jenjang['bujur'],data_point_jenjang['lintang'])]
            geom=data_point_jenjang['Point']
            geo_data_point=gpd.GeoDataFrame(data_point_jenjang,crs=crs,geometry=geom)
            geo_data_point.plot(ax=ax,markersize=20,color='blue',label='SMP')
        elif lvl=='sma':
            data_point_jenjang=data_point_example[data_point_example.bp.str.lower().str.contains(lvl)]
            # geom=[Point(xy) for xy in zip(data_point_jenjang['bujur'],data_point_jenjang['lintang'])]
            geom=data_point_jenjang['Point']
            geo_data_point=gpd.GeoDataFrame(data_point_jenjang,crs=crs,geometry=geom)
            geo_data_point.plot(ax=ax,markersize=20,color='green',label='SMA')
        elif lvl=='smk':
            data_point_jenjang=data_point_example[data_point_example.bp.str.lower().str.contains(lvl)]
            # geom=[Point(xy) for xy in zip(data_point_jenjang['bujur'],data_point_jenjang['lintang'])]
            geom=data_point_jenjang['Point']
            geo_data_point=gpd.GeoDataFrame(data_point_jenjang,crs=crs,geometry=geom)
            geo_data_point.plot(ax=ax,markersize=20,color='black',label='SMK')
        
    else:
        # geom=[Point(xy) for xy in zip(data_point_example['bujur'],data_point_example['lintang'])]
        geom=data_point_example['Point']
        geo_data_point=gpd.GeoDataFrame(data_point_example,crs=crs,geometry=geom)
        geo_data_point.plot(ax=ax,markersize=20,color='red',label='school')
    
    plt.legend(prop={'size':15})
    plt.title(kab_kot_name)
    return plt.show()

def deg_distance(deg):
    x=1*np.sin(deg)*1.84
    y=1*np.cos(deg)*1.86
    
    return np.sqrt((x**2)+(y**2))

def dec_to_dms(buff):
    Deg=np.floor(buff)
    buff0=buff-Deg
    base_min=(1/60)
    Minu=0
    while buff0>base_min:
        buff0=buff0-base_min
        Minu+=1
    Sec=buff0*3600
    return Deg,Minu,Sec

def radius_to_km(buff):
    x,y,z=dec_to_dms(buff)
    distance=(x*110.93899)+(y*1.85018)+(z*0.030821)
    return distance

proj_wgs84 = pyproj.Proj(init='epsg:4326')


def geodesic_point_buffer(points, km):
    # Azimuthal equidistant projection
    x,y=points.coords[0]
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=x, lon=y)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    return transform(project, buf).exterior.coords[:]



def get_school_map_radius(kab_name,buff):
    outlier=False
    map_data=map_df.copy()
    source_data=data_merge_map.copy()
    
    if outlier:
        pass
    else:
        source_data=source_data[source_data['outlier_point']==0]
    
    data_point_example=source_data[source_data.NAME_2==kab_name]
    data_point_example=data_point_example.dropna()
#     crs={'init':'epsg:4326'}
    crs={'init': 'epsg:3174'}
    
    fig,ax=plt.subplots(figsize=(12,12))
    map_data[map_data.NAME_2==kab_name].plot(ax=ax,alpha=0.4,color='grey')

    # geom=[Point(xy) for xy in zip(data_point_example['bujur'],data_point_example['lintang'])]
    geom=data_point_example['Point']
    pts=GeoSeries(geom)
    ## buff in KM
    circles = pts.buffer(buff)
    distan=str(radius_to_km(buff))
    plt.title('School Radius '+distan[:4]+' KM')
    return circles.plot(ax=ax,alpha=0.4,color='black')

def random_points_in_polygon(number, polygon):
    import random
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i= 0
    while i < number:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if (polygon.contains(point))&(point not in points):
            points.append(point)
            i += 1
    return points

def random_points_in_area(number,area_name):
    polyg=map_df[map_df.NAME_2==area_name].geometry.values[0]
    points=random_points_in_polygon(number, polyg)
    return points

def plot_map_and_point(geom_points,kab_name):
    map_data=map_df.copy()
    crs={'init': 'epsg:3174'}
    
    fig,ax=plt.subplots(figsize=(12,12))
    map_data[map_data.NAME_2==kab_name].plot(ax=ax,alpha=0.4,color='grey')
    pts=GeoSeries(geom_points)
    ## buff in KM
    pts.plot(ax=ax,alpha=0.4,color='black')  
    plt.title(kab_name)
    return plt.legend(prop={'size':15})

#Path (lokasi) direktori yang digunakan dalam suatu proyek untuk menyimpan data
PATH_PROCESSED='../data/processed/'
PATH_INTERIM='../data/interim/'
PATH_MODEL="../models/"

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


# Halaman Home
def home_page():
    st.title("Home")
    st.write("Selamat datang di halaman Home!")
    if st.button("Logout"):
        st.session_state.page = "Register"
        
    #--------------------------------------------
    
    # Menghitung kolom 'is_updated' berdasarkan kondisi
    data_skul['is_updated']=(data_skul.last_sync!='-').astype(int)
#     df = data_skul.sample(10)
#     st.write(df)
    

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
    
    count_bp = data_skul.groupby(['bp', 'status']).size()
    df_skul_count = pd.DataFrame({'skul_count': count_bp, 'skul_pct': count_bp / count_bp.sum()})
    
    # Menghitung jumlah sekolah dan persentase sekolah berdasarkan bp dan provinsi
    dprov_cnt_all = []
    for prov in data_skul.provinsi.unique():
        dprov = data_skul[(data_skul.provinsi == prov) & (data_skul.bp != 'SD')]
        dprov_cnt = dprov.groupby('bp').size()
        dprov_cnt = pd.DataFrame({'skul_cnt': dprov_cnt, 'skul_pct': dprov_cnt / dprov_cnt.sum()}).reset_index()
        dprov_cnt['provinsi'] = prov
        dprov_cnt_all.append(dprov_cnt)
    dprov_cnt_all = pd.concat(dprov_cnt_all)

    # Menghitung persentase berdasarkan provinsi dan bp
    dprov_bp_pct = dprov_cnt_all.groupby(['provinsi', 'bp']).mean().unstack()['skul_pct'].fillna(0.0)
    
    # Menghitung jumlah sekolah per kecamatan
    df_kec_skul_cnt=data_skul.groupby(['provinsi','kecamatan']).size().reset_index()

    # Menghitung standar deviasi jumlah sekolah per provinsi
    XX=df_kec_skul_cnt.groupby('provinsi')[0].std()

   # Fungsi untuk membuat kolom 'Point' geometri
    def create_points_column(row):
        y = row['lintang']
        x = row['bujur']
        return Point(x, y)

    
    # Menghitung atribut 'outlier_point' berdasarkan kondisi
    data_skul['outlier_point'] = np.where((data_skul.lintang > 20) | (data_skul.lintang < -10) | (data_skul.bujur < 80), 1, 0)

    # Menampilkan scatter plot di Streamlit
    st.title("Indonesia's School Data Points")
    plt.figure(figsize=(20, 10))
    sns.scatterplot(x='bujur', y='lintang', data=data_skul[data_skul.outlier_point == 0])
    st.pyplot(plt)

    # Menggabungkan data menggunakan merge
    data_merge_map = pd.merge(data_skul, map_reference[['kab_kota', 'NAME_2']], on='kab_kota', how='left')
    data_merge_map = pd.merge(data_merge_map, map_df[['NAME_2', 'NAME_1', 'geometry']], on='NAME_2', how='left')

    # Konversi dataframe menjadi Geopandas
    data_merge_map = gpd.GeoDataFrame(data_merge_map, geometry='geometry')

    # Menerapkan fungsi 'create_points_column' untuk membuat kolom 'Point'
    data_merge_map['Point'] = data_merge_map.apply(create_points_column, axis=1)

#     # Menampilkan dataframe dan kolom 'Point' di Streamlit
#     st.write("Data Merge Map:")
#     st.write(data_merge_map)

#     st.write("Kolom 'Point':")
#     st.write(data_merge_map['Point'])



    # Menghitung jumlah sekolah per area yang tidak termasuk outlier_point
    school_cnt = data_merge_map[data_merge_map.outlier_point == 0].groupby(['NAME_2']).size().reset_index()
    school_cnt_map = pd.merge(map_df, school_cnt, how='left', on='NAME_2')

    # Mengekspansi baris data sekolah untuk memetakan jumlah sekolah di setiap area
    school_cnt_map2 = school_cnt_map[school_cnt_map[0].isnull() == False].explode(ignore_index=True)

# #     # Menampilkan data hasil perhitungan
# #     st.write("Jumlah Sekolah per Area:")
# #     st.dataframe(school_cnt_map2)

    # Memilih area tertentu
    chosen_area = ['Jakarta Utara', 'Jakarta Timur', 'Jakarta Selatan', 'Jakarta Pusat', 'Jakarta Barat']
    
    city_p = 'Surabaya'
    pp = random_points_in_area(3000, city_p)
    geo_pp = gpd.GeoSeries(pp)
    data_skul_p = data_merge_map[(data_merge_map.NAME_2 == city_p) & (data_merge_map.outlier_point == 0)]

#     st.write("Data data_skul_p:")
#     st.write(data_skul_p)

    # Flagging outside the polygon
    XX = data_skul_p['Point'].values
    XX = gpd.GeoDataFrame({'geometry': XX})
    data_skul_p['inside_polyg'] = XX.within(map_df[map_df.NAME_2 == city_p].geometry.values[0]).values
    data_skul_p['inside_polyg'] = data_skul_p['inside_polyg'].astype(int)

    # Membuat visualisasi peta
    crs = {'init': 'epsg:4326'}
    fig, ax = plt.subplots(figsize=(12, 12))
    map_df[map_df.NAME_2 == city_p].plot(ax=ax, alpha=0.4, color='grey')

    # Mapping inside the polygon
    geom1 = data_skul_p[data_skul_p.inside_polyg == 1]['Point']
    geo_data_point = gpd.GeoDataFrame(data_skul_p[data_skul_p.inside_polyg == 1], crs=crs, geometry=geom1)
    geo_data_point.plot(ax=ax, markersize=20, color='blue', label='Inside')

    # Mapping outside the polygon
    geom0 = data_skul_p[data_skul_p.inside_polyg == 0]['Point']
    geo_data_point = gpd.GeoDataFrame(data_skul_p[data_skul_p.inside_polyg == 0], crs=crs, geometry=geom0)
    geo_data_point.plot(ax=ax, markersize=20, color='red', label='Outside')

#     st.pyplot(fig)
    
    
    # Menghitung rata-rata inside_polyg untuk sekolah SMA
    mean_inside_polyg_sma = data_skul_p[data_skul_p.bp.str.lower().str.contains('sma')].inside_polyg.mean()

    # Menampilkan rata-rata di Streamlit
    st.write("Rata-rata inside_polyg untuk sekolah SMA:")
    st.write(mean_inside_polyg_sma)

    # Filter sekolah yang berada dalam polygon dan merupakan SMA
    data_skul_p2 = data_skul_p[(data_skul_p.bp.str.lower().str.contains('sma')) & (data_skul_p.inside_polyg == 1)]

    # Fungsi transformasi geometri
    def geom_transform(x):
        x, y = x.split('POINT')[-1].strip()[1:-1].split()
        x, y = float(x), float(y)
        return Point(x, y)

    # Membaca data GeoDataFrame
    df_geo_chosen = pd.read_csv(PATH_INTERIM + '20191107_radius_data.csv')
    df_geo_chosen['geometry'] = df_geo_chosen['geometry'].apply(geom_transform)
    df_geo_chosen = gpd.GeoDataFrame(df_geo_chosen, geometry=df_geo_chosen['geometry'])

    # Menampilkan GeoDataFrame di Streamlit
#     st.write("Data GeoDataFrame:")
#     st.write(df_geo_chosen.head())

    # Inisialisasi list kosong untuk menyimpan data
    data_merge_map2 = []

    # Melakukan looping untuk setiap nilai unik di kolom NAME_2
    for city_pp in data_merge_map.NAME_2.dropna().unique():
        data_skul_p = data_merge_map[data_merge_map.NAME_2 == city_pp]
        XX = data_skul_p['Point'].values
        XX = gpd.GeoDataFrame({'geometry': XX})
        try:
            data_skul_p['inside_polyg'] = XX.within(map_df[map_df.NAME_2 == city_pp].geometry.values[0]).values
        except:
            print(city_pp)
            data_skul_p['inside_polyg'] = XX.within(map_df[map_df.NAME_2 == city_pp].geometry.values[0]).values
        data_skul_p['inside_polyg'] = data_skul_p['inside_polyg'].astype(int)
        data_merge_map2.append(data_skul_p)

    # Concatenate the list of dataframes
    data_merge_map2 = pd.concat(data_merge_map2)

    # Save as CSV
    data_merge_map2.to_csv('merged_data.csv', index=False)
    data_merge_map2.read_csv('merged_data.csv')

    # Menampilkan nilai unik dari kolom NAME_2 di Streamlit
#     unique_name_2 = data_merge_map2.NAME_2.unique()
#     st.write("Nilai unik dari kolom NAME_2:")
#     st.write(unique_name_2)
    
    # Fungsi untuk menghasilkan warna acak
    def generate_random_colors(count):
        rand_color = randomcolor.RandomColor()
        return rand_color.generate(count=count)

ini kalo mau lihat lengkapnya

    ## Get SMA only
    df_cls = data_merge_map2[(data_merge_map2.bp.str.lower().str.contains('sma'))
                            &(data_merge_map2.inside_polyg==1)][['NAME_2','Point','bujur','lintang']]

    # Daftar kabupaten/kota yang dapat dipilih
    options = ['Jakarta Pusat', 'Jakarta Barat', 'Jakarta Utara', 'Jakarta Timur']
    kab_n = st.sidebar.selectbox('Pilih Kabupaten/Kota', options)
    df_cls_kab = df_cls[df_cls.NAME_2 == kab_n]

    XX = df_cls_kab[['bujur','lintang']].drop_duplicates()
    XX = (XX - XX.mean()) / XX.std()
    data_eval = []

    mkcl = int(df_cls_kab.shape[0] / 3) ## Maksimum possible clusters
    for k0 in [3 + i for i in range(mkcl)]:
        Km0 = KMeans(n_clusters=k0, random_state=456)
        cls_result = Km0.fit_predict(XX)
        if (min(cls_result.tolist().count(i) for i in set(cls_result)) >= 3):
            silscor = silhouette_score(XX, cls_result)
            data_eval.append([k0, silscor])

    df_data_eval = pd.DataFrame(data_eval, columns=['k', 'sils'])
    df_data_eval.index = df_data_eval['k']

    ks = df_data_eval.sils.idxmax()

    if ks >= mkcl:
        ks = df_data_eval[df_data_eval['k'] < mkcl].sils.idxmax()

    min_ids = 3

    if ks == min_ids:
        ks = df_data_eval[(df_data_eval['k'] > min_ids) & (df_data_eval['k'] < mkcl)].sils.idxmax()

    print(ks)

    real_KM0 = KMeans(n_clusters=ks, random_state=456)

    XX0 = df_cls_kab[['bujur','lintang']]
    XX0 = (XX0 - XX.mean()) / XX.std()
    real_cls = real_KM0.fit_predict(XX0)
    df_cls_kab['cls'] = real_cls

    number_of_colors = len(df_cls_kab.cls.unique())
    color_voronoi = generate_random_colors(count=number_of_colors)

    df_colvo = pd.DataFrame({'cls': [i for i in range(number_of_colors)], 'colvo': color_voronoi})
    df_cls_kab = pd.merge(df_cls_kab, df_colvo, on='cls', how='left')
    df_cls_kab = df_cls_kab.sort_values('cls')
    df_cls_kab = df_cls_kab.reset_index()
    del df_cls_kab['index']

    # Mendapatkan data peta berdasarkan kabupaten/kota
    map_df_v = map_df[map_df.NAME_2 == kab_n]

    poly_v = map_df_v['geometry'].values[0]
    poly_shapes, poly_to_pt_assignments = voronoi_regions_from_coords(df_cls_kab[['bujur','lintang']].values, poly_v)

    df_cls_kab['col_sort'] = get_points_to_poly_assignments(poly_to_pt_assignments)
    df_cls_kab['vor'] = df_cls_kab.col_sort.apply(lambda x: poly_shapes[x])
    cte = df_cls_kab.colvo.unique()

    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_facecolor('#EBF2F2')
    for cc in df_cls_kab.cls.unique():
        geom_v = df_cls_kab[df_cls_kab.cls == cc]['vor']
        geo_vor = gpd.GeoDataFrame(df_cls_kab[['bujur','lintang']], geometry=geom_v)
        geo_vor.plot(ax=ax, markersize=20, color=cte[cc], edgecolor='grey')

    dsp2 = data_merge_map2[(data_merge_map2.bp.str.lower().str.contains('sma'))
                &(data_merge_map2.inside_polyg == 1)
                &(data_merge_map2.NAME_2 == kab_n)]

    XX_v = dsp2[['bujur','lintang']]
    XX_v = (XX_v - XX_v.mean()) / XX_v.std()
    dsp2['cls'] = real_KM0.fit_predict(XX_v)

    geom1 = dsp2['Point']
    geo_data_point = gpd.GeoDataFrame(dsp2, geometry=geom1)
    geo_data_point.plot(ax=ax, markersize=20, color='black', label='SMA')
    
    # Label kecamatan
    for idx, row in dsp2.iterrows():
        ax.annotate(row['kecamatan'], (row['bujur'], row['lintang']), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.title(kab_n + "\nVoronoi Zoning School", {'fontsize': 20})
    st.pyplot(fig)
    
    # Create a Folium map centered at the mean latitude and longitude
    map_center = [df_cls_kab['lintang'].mean(), df_cls_kab['bujur'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    # Add the Voronoi polygons as GeoJSON to the map
    for idx, row in df_cls_kab.iterrows():
        if 'colvo' in row:
            color = row['colvo']
        else:
            color = f"#{random.randint(0, 0xFFFFFF):06x}"
        folium.GeoJson(row['vor'], name=row['cls'], style_function=lambda x: {'color': color, 'fillOpacity': 0.2}).add_to(m)

    # Add the markers for SMA points to the map
    for idx, row in dsp2.iterrows():
        folium.Marker([row['lintang'], row['bujur']], popup=row['kecamatan']).add_to(m)

    # Convert the Folium map to a Streamlit element
    folium_static(m)


        
    #Start---------------------------------------

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

#             # Display the map using Google Maps Embed API
#             st.markdown(f'<iframe width="100%" height="500" src="{map_url}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

            # Create a Folium map centered at the mean latitude and longitude
            map_center = [df_cls_kab['lintang'].mean(), df_cls_kab['bujur'].mean()]
            m = folium.Map(location=map_center, zoom_start=12)

            # Add the Voronoi polygons as GeoJSON to the map
            for idx, row in df_cls_kab.iterrows():
                if 'colvo' in row:
                    color = row['colvo']
                else:
                    color = f"#{random.randint(0, 0xFFFFFF):06x}"
                folium.GeoJson(row['vor'], name=row['cls'], style_function=lambda x: {'color': color, 'fillOpacity': 0.2}).add_to(m)

            schools = pd.DataFrame()  # Initialize schools DataFrame

            # Iterate over the polygons to check if the location is inside
            for idx, row in df_cls_kab.iterrows():
                polygon = row['vor']
                st.write(polygon)
                if polygon.contains(Point(longitude, latitude)):
                    # Filter the schools within the polygon
                    schools = dsp2[(dsp2['cls'] == row['cls']) & (dsp2['inside_polyg'] == 1)]
                    st.write(schools)

                    # Calculate distances from the searched location to each school
                    schools['distance'] = schools.apply(lambda x: geodesic((latitude, longitude), (x['lintang'], x['bujur'])).kilometers, axis=1)
                    st.write(schools['distance'])
                    # Get the nearest school
                    nearest_school = schools.loc[schools['distance'].idxmin()]
                    st.write(nearest_school)
                    # Add a red marker for the searched location on the map
                    folium.Marker([latitude, longitude], popup=location_name, icon=folium.Icon(color='red')).add_to(m)

                    # Add a marker for the nearest school
                    folium.Marker([nearest_school['lintang'], nearest_school['bujur']],
                                  popup=f"Nama Sekolah: {nearest_school['nama_sekolah']}\nJarak: {nearest_school['distance']} km").add_to(m)

            # Convert the Folium map to a Streamlit element
            folium_static(m)

            # Display the sorted schools in a table
            sorted_schools = schools.sort_values('distance')
            st.write(sorted_schools)
            st.write(sorted_schools[['nama_sekolah', 'distance']])

        else:
            st.error("Lokasi tidak ditemukan")              




            
     

home_page()