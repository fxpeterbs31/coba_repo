import streamlit as st
import googlemaps
# %matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# https://medium.com/tinghaochen/install-geopandas-on-macos-mojave-32c0ab0b7d18
import geopandas as gpd
from geopandas import GeoSeries
import geoplot.crs as gcrs
import fiona
from shapely.geometry import LineString, Polygon, Point
import numpy as np

import shapely
import descartes
import Levenshtein
# import geog
import shapely.geometry
import geoplot
import matplotlib
import pyproj
from shapely.ops import transform
from functools import partial
import collections


from ipywidgets import *
from IPython.display import display, HTML
import ipywidgets as widgets


from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


from geovoronoi import voronoi_regions_from_coords
import random
import randomcolor

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
    # 1 degree in equator
#     x=1*np.sin(deg)*111.32
#     y=1*np.cos(deg)*110.57
    
    # 1 minute in equator
    x=1*np.sin(deg)*1.84
    y=1*np.cos(deg)*1.86
    
    # 1 second in equator
#     x=1*np.sin(deg)*30.72
#     y=1*np.cos(deg)*30.92
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




option = st.sidebar.selectbox("Navigation", ["Home", "Dataframe", "Chart", "PlotMapping"])



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
    st.write("## Dataframe")  # Menampilkan judul halaman dataframe
    
    st.subheader('Data Sekolah')
    df = data_skul.head()  # Mengambil 5 baris pertama dari DataFrame data_skul
    st.write(df)  # Menampilkan dataframe menggunakan st.write
    
    st.subheader('Data Sekolah yang sudah disertai dengan jumlah sekolah tiap provinsi dari 10 baris secara acak')
    data_skul['is_updated']=(data_skul.last_sync!='-').astype(int)
    df = data_skul.sample(10)
    st.write(df)
    
    st.subheader('Data referensi untuk konvensi penamaan')
    map_reference.head()
    st.write(df)
    
    prov = len(data_skul.provinsi.unique())
    kec = len(data_skul.kecamatan.unique())
    
    st.subheader(f"Provinsi: {prov} dan Kecamatan: {kec}")
    
    count_bp = data_skul.groupby('bp').size()
    df_skul_count = pd.DataFrame({'skul_count': count_bp, 'skul_pct': count_bp / count_bp.sum()}).sort_values('skul_count', ascending=False)

    st.subheader("Jumlah Sekolah per BP")
    st.write(df_skul_count)

    count_bp = data_skul.groupby(['bp', 'status']).size()
    df_skul_count = pd.DataFrame({'skul_count': count_bp, 'skul_pct': count_bp / count_bp.sum()})

    st.subheader("Jumlah Sekolah per BP dan Status")
    st.write(df_skul_count)
    
    df_kec_skul_cnt=data_skul.groupby(['provinsi','kecamatan']).size().reset_index()
    
    st.write(df_kec_skul_cnt[df_kec_skul_cnt[0]>200])
    
    XX=df_kec_skul_cnt.groupby('provinsi')[0].std()
    st.write(XX.sort_values(ascending=False))
    
elif option == 'Chart':
    st.write("## Draw Charts")  # Menampilkan judul halaman

    data_skul['is_updated']=(data_skul.last_sync!='-').astype(int)
    data_update_skul = data_skul.groupby(['provinsi', 'is_updated']).size().unstack()
    data_update_skul['0_pct'] = data_update_skul[0] / data_update_skul.sum(axis=1)
    data_update_skul['1_pct'] = data_update_skul[1] / data_update_skul.sum(axis=1)

    # Create the bar chart with stacked bars
    fig, ax = plt.subplots(figsize=(25, 10))
    data_update_skul[['0_pct', '1_pct']].plot(kind='bar', stacked=True, ax=ax)

    # Add labels to the bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(x + width / 2, y + height / 2, '{:.1f}%'.format(height * 100),
                horizontalalignment='center', verticalalignment='center')

    # Set the legend outside the plot area
    ax.legend(bbox_to_anchor=(1, 1))

    # Display the chart using st.pyplot()
    st.subheader("Persentase Data yang Diperbarui dan Belum Diperbarui per Provinsi")
    st.pyplot(fig)
    
    
   # Menambahkan judul pada plot
    st.subheader("Jumlah Data yang Diperbarui per Tanggal")

    df_updated_skul = data_skul[data_skul.is_updated == 1]
    df_updated_skul['last_sync'] = pd.to_datetime(df_updated_skul['last_sync'])
    df_updated_skul['last_sync_date'] = df_updated_skul.last_sync.apply(lambda x: x.date())

    # Menggunakan Matplotlib untuk membuat plot
    fig, ax = plt.subplots(figsize=(25, 10))
    df_updated_skul.groupby('last_sync_date').size().plot(ax=ax)

    # Menampilkan plot di Streamlit
    st.pyplot(fig)


    # Membuat plot dengan gaya seaborn
    sns.set(style="whitegrid")

    dprov_cnt_all = []
    for prov in data_skul.provinsi.unique():
        dprov = data_skul[(data_skul.provinsi == prov) & (data_skul.bp != 'SD')]
        dprov_cnt = dprov.groupby('bp').size()
        dprov_cnt = pd.DataFrame({'skul_cnt': dprov_cnt, 'skul_pct': dprov_cnt / dprov_cnt.sum()}).reset_index()
        dprov_cnt['provinsi'] = prov
        dprov_cnt_all.append(dprov_cnt)
    dprov_cnt_all = pd.concat(dprov_cnt_all)

    dprov_bp_pct = dprov_cnt_all.groupby(['provinsi', 'bp']).mean().unstack()['skul_pct'].fillna(0.0)

    # Mengatur ukuran plot
    plt.figure(figsize=(12, 6))

    # Membuat plot stacked bar
    ax = dprov_bp_pct.plot(kind='bar', stacked=True)

    # Menambahkan label pada setiap bar
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(x + width / 2,
                y + height / 2,
                '{:.1f}%'.format(height * 100),
                ha='center',
                va='center')

    # Mengatur judul plot dan label sumbu
    plt.title("Persentase Jumlah Sekolah Berdasarkan Bentuk Pendidikan dan Provinsi")
    plt.xlabel("Provinsi")
    plt.ylabel("Persentase")

    # Menampilkan plot di Streamlit
    st.pyplot(plt)
    
    # Membuat boxplot menggunakan seaborn
    df_kec_skul_cnt = data_skul.groupby(['provinsi','kecamatan']).size().reset_index()

    plt.figure(figsize=(20, 9))
    sns.boxplot(x='provinsi', y=0, data=df_kec_skul_cnt)
    plt.xticks(rotation=90)

    # Mengatur judul plot dan label sumbu
    plt.title("Boxplot Jumlah Sekolah per Provinsi")
    plt.xlabel("Provinsi")
    plt.ylabel("Jumlah Sekolah")

    # Menampilkan plot di Streamlit
    st.pyplot(plt)

elif option == 'PlotMapping':
    st.title("Plot Mapping")
    fig, ax = plt.subplots(figsize=(20, 10))
    map_df.plot(ax=ax)

    # Menampilkan gambar plot di Streamlit
    st.pyplot(fig)

    data_merge_map = pd.merge(data_skul, map_reference[['kab_kota', 'NAME_2']], on='kab_kota', how='left')
    data_merge_map = pd.merge(data_merge_map, map_df[['NAME_2', 'NAME_1', 'geometry']], on='NAME_2', how='left')

    # Menampilkan dropdown di sidebar
    dropdown_kab = st.sidebar.selectbox("Kabupaten/Kota", map_df.NAME_2.sort_values().unique())

    # Menampilkan float slider di sidebar
    buff = st.sidebar.slider("Buffer", min_value=0.001, max_value=0.0900849, value=0.002, step=0.001)

    # Memanggil fungsi interaktif
    get_school_map_radius(dropdown_kab, buff)

    school_cnt = data_merge_map[data_merge_map.outlier_point == 0].groupby(['NAME_2']).size().reset_index()
    school_cnt_map = pd.merge(map_df, school_cnt, how='left', on='NAME_2')
    st.write(school_cnt_map[school_cnt_map[0].isnull() == True])

    school_cnt_map2 = school_cnt_map[school_cnt_map[0].isnull() == False].explode(ignore_index=True)

    # Membuat plot menggunakan geoplot
    ax = geoplot.choropleth(
        school_cnt_map2, hue=school_cnt_map2[0], figsize=(25, 15), legend=True
    )

    school_cnt_map[school_cnt_map[0].isnull() == True].plot(ax=ax, color='red')

    # Menampilkan dataframe
    st.subheader("Dataframe Jumlah Sekolah per Kabupaten/Kota")
    st.dataframe(school_cnt_map)

    # Menampilkan line chart
    st.subheader("Line Chart Jumlah Sekolah per Kabupaten/Kota")
    st.line_chart(school_cnt_map[0])


    
    




