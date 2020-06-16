# В скрипте реализована возможность визуализации всех маршрутов,
# проходящих через выбранный район
# с подписью каждого маршрута на начальной и конечной его точке, с указанием станций СВТ. 

import pandas as pd
import numpy as np
from datasets.processed import askp_passflows
from datetime import datetime
import geopandas as gpd

import os
from tqdm import tqdm_notebook as tqdm

# # В случе ошибки RuntimeError: b'no arguments in initialization list'
# # необходимо снять комментарии у этого текста
# conda_file_dir = conda.__file__
# conda_dir = conda_file_dir.split('lib')[0]
# proj_lib = os.path.join(os.path.join(conda_dir, 'pkgs'), 'proj4-5.2.0-h6538335_1006\Library\share')
# os.environ["PROJ_LIB"] = proj_lib

# # Если действие выше не помогло, то нужно задать системной переменной PROJ_LIB
# # явный путь к окружению по аналогии ниже
# Для настройки проекции координат, поменять на свой вариант
os.environ ['PROJ_LIB']=r'C:\Users\popova_kv\AppData\Local\Continuum\anaconda3\Library\share'

#отключить предупреждения pandas (так быстрее считает!!!):
pd.options.mode.chained_assignment = None


import tilemapbase
tilemapbase.start_logging()
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import re

import matplotlib.font_manager as fm
from matplotlib.cbook import get_sample_data
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from sklearn.cluster import DBSCAN

from sqlalchemy import create_engine
import getpass


import matplotlib.cm as cm
from shapely.geometry import Point

from adjustText import adjust_text

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from haversine import haversine

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider, IntSlider

#%matplotlib inline


# Шрифты для изображений
parent_directory = os.path.abspath('.')
prop = fm.FontProperties(fname=r'{}\files\fonts\MoscowSans-Regular.otf'.format(parent_directory))
# Расположение иконок
parent_directory = os.path.abspath('.')
bus_path = r'{}\files\icons_png\transport_type\bus.png'.format(parent_directory)
metro_path = r'{}\files\icons_png\transport_type\metro.png'.format(parent_directory)
train_path = r'{}\files\icons_png\transport_type\train.png'.format(parent_directory)
tram_path = r'{}\files\icons_png\transport_type\tram.png'.format(parent_directory)
trolleybus_path = r'{}\files\icons_png\transport_type\trolleybus.png'.format(parent_directory)

mcd_path = r'{}\files\icons_png\transport_type\mcd\МЦД.png'.format(parent_directory)
mcd_1_path = r'{}\files\icons_png\transport_type\mcd\МЦД-1.png'.format(parent_directory)
mcd_2_path = r'{}\files\icons_png\transport_type\mcd\МЦД-2.png'.format(parent_directory)
mcd_3_path = r'{}\files\icons_png\transport_type\mcd\МЦД-3.png'.format(parent_directory)
mcd_4_path = r'{}\files\icons_png\transport_type\mcd\МЦД-4.png'.format(parent_directory)
mcd_5_path = r'{}\files\icons_png\transport_type\mcd\МЦД-5.png'.format(parent_directory)

print("1. Ввод данных для подключения к базе")
print("______________________________________")

# Логин пользователя от базы данных UARMS
login = input('Имя пользователя: ')
# Пароль пользователя от базы данных UARMS
password = getpass.getpass('Пароль: ')

# Дата, за которую нужно получить срез данных
date_inp = input('Введите дату ГГГГММДД: ')
dt = datetime(int(date_inp[:4]), int(date_inp[4:6]), int(date_inp[6:]), 0, 0)
date = pd.Timestamp(date_inp).strftime("%Y-%m-%d")



list_of_agencies = ['межсубъектные','Прочие','Мосгортранс','commercial']
print("Список перевозчиков: " + str(list_of_agencies))
agency_inp = input('Введите перевозчиков, как в примере, с кавычками и через запятую: ')

list_variants = ['00', 'У1', 'У2', 'Ч1', 'Ч2', 'Д1', 'Д2', 'У3', 'У4', 'У5', 'Э1','У7', 'У8', 
                 'У9', 'У6', 'после_18', 'до_9']
#
print("По умолчанию будет выбран основной вариант трасс (00)")
print("Если выбрать все - введите что угодно")
sel_var_amount = input('Если хотите выбрать варианты вручную - введите 1: ')
if sel_var_amount == '1':
	print("Список вариантов: " + str(list_variants))
	variant_inp = input('Введите варианты, как в примере, с кавычками и через запятую: ')
else:
	variant_inp = "'00'"

print("______________________________________")

print("__Создание директории для сохранения картинок__")
path_new = os.getcwd()
path_imgs = path_new + '\\imgs'
try:
    os.mkdir(path_imgs)
except OSError:
    print ("Не удалось создать директорию: %s \n" % path_imgs)
    print("Возможно, она уже создана")
else:
    print ("Ок %s \n" % path_imgs)
# 
print("______________________________________")
print()
print("2. Выбор региона")
print("______________________________________")


print("Внимание!!! Выбор региона - через input")
print("Варианты районов/округов можно посмотреть в выпадающих списках")


engine = create_engine(
    'postgresql://{login}:{password}@airms.mgtniip.ru:5432/UARMS'\
    .format(login=login, password=password)
)

# выбор слоя с районами
munics = gpd.read_postgis(
    """
    select *
    from territory.munics
    """,
    engine,
    geom_col='geometry',
    crs='+init=epsg:4326'
)

# выбор слоя с округами
adms = gpd.read_postgis(
    """
    select *
    from territory.adms
    """,
    engine,
    geom_col='geometry',
    crs='+init=epsg:4326'
)

#None


list_vars_reg = ['район', 'округ']

list_names_mun = list(munics.name.unique())
list_names_mun.sort()
list_names_adms = list(adms.name.unique())
list_names_adms.sort()

print('район')
#
@interact
def selection_adm(mun_reg=list_names_mun):
	global select_reg, mun_sel
	select_reg = munics.loc[munics.name == mun_reg]
	#return select_reg
	# mun_sel = mun
#

print('округ')
#       
@interact
def selection_reg(mun_adms=list_names_adms):
	global select_reg, mun_sel
	select_reg = adms.loc[adms.name == mun_adms]
	#return select_reg
	# mun_sel = mun
#



inp_var_name = input('Напишите "район" или "округ": ')
if inp_var_name == 'район':
	inp_name_mun = input('Введите название (можно не полностью, например "Аэропорт"): ')
if inp_var_name == 'округ':
	inp_name_adm = input('Введите название (можно не полностью, например "Северо-Западный"): ')
# else:
	# print('Введено некорректное значение, будет ошибка')
#



if inp_var_name == 'район':
	select_reg = munics[munics.name.str.contains(inp_name_mun)]
if inp_var_name == 'округ':
	select_reg = adms[adms.name.str.contains(inp_name_adm)]

mun_name_f = list(select_reg.name)[0]
reg = re.compile('[^а-яА-Я0-9,._\- ]')
mun_name = reg.sub('', mun_name_f)
mun_name = re.sub(" +", " ", mun_name)


print("Выбран: ", mun_name,".")
# 

mun_sel = inp_var_name

print("Если выбрать все - введите что угодно")

# print("fig_x_scale =",fig_x_scale)
# print("fig_y_scale =", fig_y_scale)
# print("sel_dpi_scale =", sel_dpi_scale)

# print("fig_x_full =", fig_x_full)
# print("fig_y_full =", fig_y_full)
# print("sel_dpi_full =", sel_dpi_full)

answer_amount = input("Если хотите выбрать маршруты вручную, введите 2: ")
if answer_amount == '2':
	inp_rn = input("Введите route_name, через запятую, без пробелов: ")
	inp_rn = list(inp_rn.split(","))
	amnt_rt = "sel"
	print("Список выбранных маршрутов: ",inp_rn)
else:
	amnt_rt = ""
#

print("______________________________________")
print()
print("3. Ввод особенностей картинки")
print("______________________________________")


print("3.1 Выберите вариант раскраски трасс/подписей")
print("1 - все перевозчики одним цветом. 2 - все разным")
answer_color = input('Введите 1 или 2: ')
if answer_color == '1':
    print("Маршруты раскрасятся одним цветом")
elif answer_color == '2':
    print("Маршруты раскрасятся разными цветами")
else:
    print("Введено некорректное значение, попробуйте еще раз")
# 

print()
print("3.2 Выберите вариант размера/пикселей картинок")
print("Выгружаются две картинки:")
print("scaled - увеличенный масштаб (граница выбранного района)")
print("full - общий масштаб (до конца всех трасс, проходящих в выбранном районе)")

print("'все одинаково' - оба варианта будут с одинаковыми параметрами")
print("'каждый по-разному' - каждый со своими параметрами")
#list_vars_dpi = ['все одинаково', 'каждый по-разному']

print("Вариант 1:все одинаково. Вариант 2:каждый по-разному.")
dpi_sel_var = input('Введите вариант: ')

if dpi_sel_var == '1':
	print("X - от 10 до 20")
	print("Y - от 10 до 20")
	print("dpi - от 150 до 650")
	print("Размер по X")
	fig_x_scale = int(input())
	print("Размер по Y")
	fig_y_scale = int(input())
	print("Кол-во пикселей (dpi)")
	sel_dpi_scale = int(input())
	fig_x_full = fig_x_scale
	fig_y_full = fig_y_scale
	sel_dpi_full = sel_dpi_scale

if dpi_sel_var == '2':
	print("X - от 10 до 20")
	print("Y - от 10 до 20")
	print("dpi - от 150 до 650")
	print("Размер по X (scaled)")
	fig_x_scale = int(input())
	print("Размер по Y (scaled)")
	fig_y_scale = int(input())
	print("Кол-во пикселей (dpi) (scaled)")
	sel_dpi_scale = int(input())
	print("Размер по X (full)")
	fig_x_full = int(input())
	print("Размер по Y (full)")
	fig_y_full = int(input())
	print("Кол-во пикселей (dpi) (full)")
	sel_dpi_full = int(input())

else:
	fig_x_scale = 12
	fig_y_scale = 14
	sel_dpi_scale = 300
	fig_x_full = 12
	fig_y_full = 14
	sel_dpi_full = 300
#

print("______________________________________")




# выгрузка слоя с ОП
stops = gpd.read_postgis(
    """
    select *
    from routes.stops_actual
    """,
    engine,
    geom_col='geometry',
    crs='+init=epsg:4326'
)

# выгрузка слоя с маршрутами за выбранную дату, с выбранными перевозчиками и вариантами
traces = gpd.read_postgis(
    """
    select 
      trip_id, 
      length, 
      mvn, 
      route_id, 
      route_name, 
      route_long_name, 
      transport_type, 
      variant_name, 
      direction,
      agency_name,
      agency_id,
      agency_group,
      geometry,
      st_x(st_endpoint(geometry)) as lon, 
      st_y(st_endpoint(geometry)) as lat
    from routes.trip_lines('{date}', 1)
    left join routes.mvns('{date}', 1) using (trip_id)
    where agency_group in ({agency_inp})
    and variant_name in ({variant_inp})
    order by trip_id
    """.format(date = date, agency_inp=agency_inp,variant_inp=variant_inp),
    engine,
    geom_col='geometry',
    crs='+init=epsg:4326'
)

trip_stops = pd.read_sql(
    """
    select *
    from routes.trip_stops('{date}', 1)
    """.format(date=date),
    engine
)

routes = pd.read_sql(
    """
    select route_id, route_name, route_long_name, transport_type, is_circle
    from routes.routes('{date}', 1)
    """.format(date=date),
    engine
)

# Выбрать трассы маршрутов, проходящих через выбранный регион
trips_in_reg = gpd.sjoin(traces, select_reg, how='inner', 
                           op='intersects').drop("index_right", axis=1).reset_index(drop=True)
#

routes_in_reg = routes[routes.route_id.isin(trips_in_reg.route_id)]
trips_in_reg = trips_in_reg.merge(routes_in_reg[['route_id', 'is_circle']], how='left', on=['route_id'])

if answer_amount == '2':
    trips_in_reg = trips_in_reg[trips_in_reg.route_name.isin(inp_rn)]
#

# Выбрать остановки метро/жд/мцд, для картинок и подписей
route_types = ['МД', 'МЦ', 'МД; Эл', 'Эл', 'М', 'ММ']
stops_cluster = stops[stops.route_types.isin(route_types)]
metro_in_reg = gpd.sjoin(stops_cluster, select_reg, how='inner', 
                           op='intersects').drop("index_right", axis=1).reset_index(drop=True)


# 

# выбор регионов для подложки (просто наложение всех остальных регионов белым полупрозрачным слоем)
if mun_sel == 'округ':
    for_border_plot = adms.copy()
    for_border_plot = for_border_plot[for_border_plot.adm_id != list(select_reg.adm_id)[0]]
# 
elif mun_sel == 'район':
    for_border_plot = munics.copy()
    for_border_plot = for_border_plot[for_border_plot.munic_id != list(select_reg.munic_id)[0]]
# 
else:
    print('Не выбран регион')
# 

#None



### преобразование данных для подписей конечных ОП

# выгрузка всех ОП в последовательности трипа
trip_stops_reg = trip_stops[trip_stops.trip_id.isin(list(trips_in_reg.trip_id.unique()))]
trip_stops_reg = gpd.GeoDataFrame(trip_stops_reg.merge(stops[['stop_id', 'geometry']], how='left', on=['stop_id']))
trip_stops_reg = trip_stops_reg[['trip_id', 'stop_sequence', 'stop_id', 'geometry']]
trip_stops_reg['stop_id'] = trip_stops_reg['stop_id'].astype(np.int64)
trip_stops_reg = trip_stops_reg.merge(trips_in_reg[['trip_id', 'route_id', 'route_name', 'mvn', 'is_circle', 
                                                         'agency_group']], how='left', on=['trip_id'])
trip_stops_reg = trip_stops_reg.drop_duplicates(['route_id', 'trip_id', 'stop_sequence']).reset_index(drop=True)



# выбор только певрого и последнего ОП для подписей конечных
f_l_stop = gpd.GeoDataFrame()

for trpid in trip_stops_reg.trip_id.unique():
    if (trip_stops_reg[trip_stops_reg.trip_id == trpid].is_circle.all() == False):
        new_df = trip_stops_reg[(trip_stops_reg.trip_id == trpid) 
                                & ((trip_stops_reg.stop_sequence == 1) 
                                   | (trip_stops_reg.stop_sequence 
                                      == 
                                      len(trip_stops_reg[trip_stops_reg.trip_id == trpid])
                                     ))] # первый и последний ОП в последовательности
    else:
        new_df = trip_stops_reg[(trip_stops_reg.trip_id == trpid) 
                                & ((trip_stops_reg.stop_sequence == 1) 
                                   | (trip_stops_reg.stop_sequence 
                                      == 
                                      int((len(trip_stops_reg[trip_stops_reg.trip_id == trpid])/2))
                                     ))] # для кольцевых берется ОП посередине как "последний"
    f_l_stop = f_l_stop.append(new_df)
    
# 
f_l_stop = f_l_stop.drop_duplicates(['trip_id', 'stop_sequence', 'stop_id']).reset_index(drop=True)



# Дальнейшая проверка - на близость конечных (конечная прямого и начальная обратного направления).
# Чтобы не было дубликатов подписей, если они ближе 400 метров друг от друга. 
table_from = f_l_stop[f_l_stop.stop_sequence == 1]
table_to = f_l_stop[f_l_stop.stop_sequence != 1]


table_from['geo_from'] = table_from['geometry']
table_from['geo_from_lat_y'] = table_from.geometry.y
table_from['geo_from_lon_x'] = table_from.geometry.x

table_to['geo_to'] = table_to['geometry']
table_to['geo_to_lat_y'] = table_to.geometry.y
table_to['geo_to_lon_x'] = table_to.geometry.x

table_from_to = table_from.merge(table_to[['trip_id', 'route_id', 'mvn','geo_to', 'geo_to_lat_y', 'geo_to_lon_x']], how='left', on=['route_id'])
table_from_to = table_from_to[table_from_to.trip_id_x != table_from_to.trip_id_y].reset_index(drop=True)

table_from_to['hav_len'] = 0.0

for i in range(len(table_from_to)):
    table_from_to.hav_len[i] = (haversine(
        (table_from_to.geo_from_lat_y[i], table_from_to.geo_from_lon_x[i]),
        (table_from_to.geo_to_lat_y[i], table_from_to.geo_to_lon_x[i])
    ))*1000 # в метрах
# 


# конечные в пределах 400 метров
ends_in_400 = table_from_to[table_from_to.hav_len < 400]
# ends_in_400 = ends_in_400.drop_duplicates(['route_id'])
ends_in_400 = ends_in_400.rename(columns={'trip_id_x':'trip_id', 'mvn_x':'mvn'})
ends_in_400 = ends_in_400[['trip_id', 'stop_sequence', 'stop_id', 'geometry', 'route_id',
                           'route_name', 'mvn', 'is_circle', 'agency_group']]

# присоединение трипов, у которых нет второго трипа (либо кольцевой, либо второй трип не проходит в этом районе)
ends_in_400 = ends_in_400.append(f_l_stop[~f_l_stop.trip_id.isin(table_from_to.trip_id_x)]).reset_index(drop=True)


# конечные дальше 400 метров друг от друга, дублирование - чтобы оставить обе точки, обе подписи
ends_more_400 = table_from_to[table_from_to.hav_len >= 400]
ends_more_400 = ends_more_400.append(ends_more_400).sort_values(by=['route_id']).reset_index(drop=True)
for i in range(len(ends_more_400)):
    if ((i/2) == 0.0):
        ends_more_400.geometry[i] = ends_more_400.geo_from[i]
    else:
        ends_more_400.geometry[i] = ends_more_400.geo_to[i]
# 
ends_more_400 = ends_more_400.rename(columns={'trip_id_x':'trip_id', 'mvn_x':'mvn'})
ends_more_400 = ends_more_400[['trip_id', 'stop_sequence', 'stop_id', 'geometry', 'route_id',
                           'route_name', 'mvn', 'is_circle', 'agency_group']]


# формирование финальной таблицы с подписями, далее - кластеризация
ends_trips = ends_in_400.append(ends_more_400).reset_index(drop=True)
ends_trips.crs={'init': 'epsg:4326'}





## Кластеризация в зависимости от группы перевозчиков

if answer_color == '1':
    ends_trips['grouped_agency'] = 'all'
elif answer_color == '2':
    ends_trips['grouped_agency'] = ends_trips['agency_group']
else:
    print('не выбран цвет')
    ends_trips['grouped_agency'] = 'all'
# 

try_group = ends_trips.copy()
clusters_trips = pd.DataFrame()
for row in list(try_group.grouped_agency.unique()):
    #print(row)
    new_df = try_group[try_group.grouped_agency == row]
    new_df['gk1'] = new_df.to_crs({'init': 'epsg:32637'}).geometry.x
    new_df['gk2'] = new_df.to_crs({'init': 'epsg:32637'}).geometry.y
    X = new_df[['gk1', 'gk2']].values

    # Clusterizing stops.
    db = DBSCAN(eps=50, min_samples=3).fit(X)
    # Saving labels of clusters (i.e. sites).
    labels = db.labels_
    new_df['label'] = labels
    # Finding which stops were considered as core points 
    # (see DBSCAN description for what a core point is).
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # (1 = Not core point, 2 = Core point. 
    #  Ugly hack needed to enable HoloView variable sizes.)
    new_df['is_core_point'] = 1 + core_samples_mask

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #print("Число кластеров: ", n_clusters_)


    sequence = new_df[['route_name','label']].reset_index(drop = True)
    sequence = sequence.sort_values(by = ['label','route_name']).reset_index(drop = True)
    sequence = sequence.groupby('label', as_index=False).agg(lambda x: ','.join(x))
    new_df = new_df.merge(sequence, on = 'label', how = 'left').reset_index(drop =True).fillna(-1)
    new_df = new_df.rename(columns={'route_name_x':'route_name', 'route_name_y':'route_names_cluster'})
    for i in range(len(new_df)):
        if (new_df.label[i] == -1):
            new_df.route_names_cluster[i] = new_df.route_name[i]
        else:
            new_df.route_names_cluster[i] = new_df.route_names_cluster[i]
    # 
    clusters_trips = clusters_trips.append(new_df)
# 

clusters_trips=clusters_trips.reset_index(drop=True)
clusters_trips['rnc_separated'] = clusters_trips['route_names_cluster']


# разделение списка названий, для подписей, по три в ряд и перенос строки
for j in range(len(clusters_trips)):
    first=0
    list_index_sep = []
    for i in range(len(clusters_trips.route_names_cluster[j])):
        first+=1
        list_index_sep.append(str(clusters_trips.route_names_cluster[j]).find(',', first,len(clusters_trips.route_names_cluster[j])))

    list_index_sep = list(set(list_index_sep)) #убрать повторы
    list_index_sep.remove(-1) #убрать последний индекс -1
    list_index_sep.sort()
    cnt=1
    for k in (list_index_sep):
        cnt = list_index_sep.index(k) + 1
        if (cnt % 3 == 0) & (clusters_trips.route_names_cluster[j][k] == str(",")):
            clusters_trips.rnc_separated[j] = clusters_trips.rnc_separated[j][:k] + str("_") + clusters_trips.rnc_separated[j][k+1:]
            
    clusters_trips.rnc_separated[j] = clusters_trips.rnc_separated[j].replace("_", "\n")
    clusters_trips.rnc_separated[j] = clusters_trips.rnc_separated[j].replace(",", ", ")
#
# Удаление дубликатов в названии кластеров, выбор первой гшеометрии для кластера
clusters_trips = clusters_trips[['rnc_separated', 'geometry', 'grouped_agency', 'label', 'route_names_cluster']].reset_index(drop=True)
clusters_trips['str_geometry'] = clusters_trips['geometry'].astype(str)

clusters_trips_1 = clusters_trips[clusters_trips.label == -1].reset_index(drop=True)
clusters_trips_2 = clusters_trips[clusters_trips.label != -1].reset_index(drop=True)

clusters_trips_1 = clusters_trips_1.drop_duplicates(['rnc_separated','str_geometry']).reset_index(drop=True)
clusters_trips_2 = clusters_trips_2.drop_duplicates(['label', 'route_names_cluster']).reset_index(drop=True)

clusters_trips = clusters_trips_1.append(clusters_trips_2).reset_index(drop=True)



# Сохранение увеличенного участка (выбранный район)
fig, ax = plt.subplots(figsize=(fig_x_scale, fig_y_scale), dpi=sel_dpi_scale)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Экстент
extent = tilemapbase.extent_from_frame(select_reg.to_crs({'init': 'epsg:3857'}).buffer(1000))

# Подложка
plotter = tilemapbase.Plotter(extent, tilemapbase.tiles.build_OSM(), width=600, height=600)\
    .plot(ax, alpha=0.5, allow_large=True,zorder=0)

# белая "пелена" поверх не выбранных регионов
if (mun_sel == 'округ' ) | (mun_sel == 'район'):
    for_border_plot.to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='white',zorder=1,alpha=0.6)
# 

# трасы маршрутов, цвета - в зависимости от выбора (все одним или разными цветами)

if answer_color == '1':
    trips_in_reg.to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='royalblue',zorder=4)#color=colors, alpha=0.5)
elif answer_color == '2':
    trips_in_reg[trips_in_reg['agency_group'] == 'Мосгортранс'].to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='royalblue',zorder=4)
    trips_in_reg[trips_in_reg['agency_group'] == 'commercial'].to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='seagreen',zorder=4)
    trips_in_reg[trips_in_reg['agency_group'] == 'межсубъектные'].to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='tomato',zorder=4)
    trips_in_reg[trips_in_reg['agency_group'] == 'Прочие'].to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='khaki',zorder=4)
else:
    print("Не выбран цвет")
    trips_in_reg.to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='royalblue',zorder=4)#color=colors, alpha=0.5)


# граница района
select_reg.to_crs({'init': 'epsg:3857'}).boundary.plot(ax=ax, linewidth=2, linestyle='--', 
                                                    color='black',zorder=5)


# Метро картинки
with get_sample_data(metro_path) as file:
    metro_img = plt.imread(file, format='png')
metro_plot = metro_in_reg[(metro_in_reg['route_types'] == 'М') |
                           (metro_in_reg['route_types'] == 'МЦ')]
for xy in zip(metro_plot.to_crs({'init': 'epsg:3857'}).geometry.x, 
              metro_plot.to_crs({'init': 'epsg:3857'}).geometry.y):
    imagebox = OffsetImage(metro_img, zoom=0.07,zorder=8)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy, frameon=False)
    ax.add_artist(ab)
# 


# МЦД картинки
with get_sample_data(mcd_path) as file:
    mcd_img = plt.imread(file, format='png')
mcd_plot = metro_in_reg[(metro_in_reg['route_types'] == 'МД') |
                         (metro_in_reg['route_types'] == 'МД; Эл')]
for xy in zip(mcd_plot.to_crs({'init': 'epsg:3857'}).geometry.x, 
              mcd_plot.to_crs({'init': 'epsg:3857'}).geometry.y):
    imagebox = OffsetImage(mcd_img, zoom=0.013,zorder=8)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy, frameon=False)
    ax.add_artist(ab)
# 


# ЖД картинки
with get_sample_data(train_path) as file:
    train_img = plt.imread(file, format='png')
train_plot = metro_in_reg[(metro_in_reg['route_types'] == 'Эл')]
for xy in zip(train_plot.to_crs({'init': 'epsg:3857'}).geometry.x, 
              train_plot.to_crs({'init': 'epsg:3857'}).geometry.y):
    imagebox = OffsetImage(train_img, zoom=0.07,zorder=8)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy, frameon=False)
    ax.add_artist(ab)
# 


# Плашки для подписей текста
box_mgt = {'facecolor':'royalblue',    #  цвет области
       'edgecolor': 'grey',     #  цвет крайней линии
       'boxstyle': 'round', #  стиль области
         'alpha': 0.8, # прозрачность
         'pad': 0.2}    #  отступы

box_comm = {'facecolor':'seagreen',    #  цвет области
       'edgecolor': 'grey',     #  цвет крайней линии
       'boxstyle': 'round', #  стиль области
         'alpha': 0.8, # прозрачность
         'pad': 0.2}    #  отступы

box_region = {'facecolor':'tomato',    #  цвет области
       'edgecolor': 'grey',     #  цвет крайней линии
       'boxstyle': 'round', #  стиль области
         'alpha': 0.8, # прозрачность
         'pad': 0.2}    #  отступы

box_train = {'facecolor':'khaki',    #  цвет области
       'edgecolor': 'grey',     #  цвет крайней линии
       'boxstyle': 'round', #  стиль области
         'alpha': 0.8, # прозрачность
         'pad': 0.2}    #  отступы

if answer_color == '1':
    texts = [ax.text(row['geometry'].centroid.x,
                 row['geometry'].centroid.y,
                 row['rnc_separated'], 
                 bbox=box_mgt,
                 fontproperties=prop, color='white',
                 fontsize=8,zorder=10)
         for index, row in clusters_trips.to_crs({'init': 'epsg:3857'}).iterrows()]
#
elif answer_color == '2':
    
    texts = [ax.text(row['geometry'].centroid.x,
                 row['geometry'].centroid.y,
                 row['rnc_separated'], 
                 bbox=box_mgt if row['grouped_agency'] == 'Мосгортранс'
                 else box_comm if row['grouped_agency'] == 'commercial'
                 else box_region if row['grouped_agency'] == 'межсубъектные'
                 else box_train if row['grouped_agency'] == 'Прочие'
                 else box_train,
                 fontproperties=prop, color='white',
                 fontsize=8,zorder=10)
         for index, row in clusters_trips.to_crs({'init': 'epsg:3857'}).iterrows()]
#
else:
    texts = [ax.text(row['geometry'].centroid.x,
                 row['geometry'].centroid.y,
                 row['rnc_separated'], 
                 bbox=box_mgt,
                 fontproperties=prop, color='white',
                 fontsize=8,zorder=10)
         for index, row in clusters_trips.to_crs({'init': 'epsg:3857'}).iterrows()]
    print("Не выбран цвет")
# 


adjust_text(texts, expand_text=(1.9, 1.9), expand_align=(1.9, 1.9), arrowprops=dict(arrowstyle='->',color='black',lw=0.3))


# Раздвигание подписей___МЕТРО
texts_metro = [ax.text(row['geometry'].centroid.x-200,
            row['geometry'].centroid.y+50,
            row['stop_name'], fontproperties=prop, color='grey',
            fontsize=7,zorder=8)
            for index, row in metro_plot.to_crs({'init': 'epsg:3857'}).iterrows()]

# 
adjust_text(texts_metro, expand_text=(1.9, 1.9), expand_align=(1.9, 1.9))


# Раздвигание подписей___МЦД
texts_mcd = [ax.text(row['geometry'].centroid.x-200,
            row['geometry'].centroid.y+50,
            row['stop_name'], fontproperties=prop, color='grey',
            fontsize=7,zorder=8)
            for index, row in mcd_plot.to_crs({'init': 'epsg:3857'}).iterrows()]

# 
adjust_text(texts_mcd, expand_text=(1.9, 1.9), expand_align=(1.9, 1.9))



# Раздвигание подписей___ЖД
texts_train = [ax.text(row['geometry'].centroid.x-200,
            row['geometry'].centroid.y+50,
            row['stop_name'], fontproperties=prop, color='grey',
            fontsize=7,zorder=8)
            for index, row in train_plot.to_crs({'init': 'epsg:3857'}).iterrows()]

# 
adjust_text(texts_train, expand_text=(1.9, 1.9), expand_align=(1.9, 1.9))


# Легенда    
# trips = mlines.Line2D([], [], color='blue', marker='_',
#                               markersize=15, label='Трассы маршрутов')
# border_mun = mlines.Line2D([], [], color='black', marker='_',
#                               markersize=15, label='Граница района')
# ax.legend(handles=[trips, border_mun],
#           loc='lower right').set_zorder(20)#,zorder=12


# Масштаб
scale = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.1
scale = int(round(scale*0.01)*100)
scalebar = AnchoredSizeBar(ax.transData,
                           scale, '{scale} м.'.format(scale=scale), 'upper right', 
                           pad=1,
                           color='black',
                           frameon=False,
                           fontproperties=prop,
                           size_vertical=30)

ax.add_artist(scalebar)

# Сохранение изображения
ax.set_title('Маршруты, проходящие через\n{}'.format(mun_name), fontproperties=prop, fontsize=20)
plt.savefig('imgs/trips_{}_{}_{}_scaled.png'.format(mun_name,date_inp, answer_color))
plt.close()

#None

print()
print("Картинка с увеличенным масштабом выгружена.")




#########################################

# Сохранение ОБЩЕГО участка
fig, ax = plt.subplots(figsize=(fig_x_full, fig_y_full), dpi=sel_dpi_full)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)


# Экстент
extent = tilemapbase.extent_from_frame(trips_in_reg.to_crs({'init': 'epsg:3857'}).buffer(1000))

# Подложка
plotter = tilemapbase.Plotter(extent, tilemapbase.tiles.build_OSM(), width=600, height=600)\
    .plot(ax, alpha=0.5, allow_large=True,zorder=0)

# белая "пелена" поверх не выбранных регионов
if (mun_sel == 'округ' ) | (mun_sel == 'район'):
    for_border_plot.to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='white',zorder=1,alpha=0.6)
# 

# трасы маршрутов, цвета - в зависимости от выбора (все одним или разными цветами)

if answer_color == '1':
    trips_in_reg.to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='royalblue',zorder=4)#color=colors, alpha=0.5)
elif answer_color == '2':
    trips_in_reg[trips_in_reg['agency_group'] == 'Мосгортранс'].to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='royalblue',zorder=4)
    trips_in_reg[trips_in_reg['agency_group'] == 'commercial'].to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='seagreen',zorder=4)
    trips_in_reg[trips_in_reg['agency_group'] == 'межсубъектные'].to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='tomato',zorder=4)
    trips_in_reg[trips_in_reg['agency_group'] == 'Прочие'].to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='khaki',zorder=4)
else:
    print("Не выбран цвет")
    trips_in_reg.to_crs({'init': 'epsg:3857'}).plot(ax=ax, color='royalblue',zorder=4)#color=colors, alpha=0.5)


# граница района
select_reg.to_crs({'init': 'epsg:3857'}).boundary.plot(ax=ax, linewidth=2, linestyle='--', 
                                                    color='black',zorder=5)


# Метро картинки
with get_sample_data(metro_path) as file:
    metro_img = plt.imread(file, format='png')
metro_plot = metro_in_reg[(metro_in_reg['route_types'] == 'М') |
                           (metro_in_reg['route_types'] == 'МЦ')]
for xy in zip(metro_plot.to_crs({'init': 'epsg:3857'}).geometry.x, 
              metro_plot.to_crs({'init': 'epsg:3857'}).geometry.y):
    imagebox = OffsetImage(metro_img, zoom=0.07,zorder=8)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy, frameon=False)
    ax.add_artist(ab)
# 


# МЦД картинки
with get_sample_data(mcd_path) as file:
    mcd_img = plt.imread(file, format='png')
mcd_plot = metro_in_reg[(metro_in_reg['route_types'] == 'МД') |
                         (metro_in_reg['route_types'] == 'МД; Эл')]
for xy in zip(mcd_plot.to_crs({'init': 'epsg:3857'}).geometry.x, 
              mcd_plot.to_crs({'init': 'epsg:3857'}).geometry.y):
    imagebox = OffsetImage(mcd_img, zoom=0.013,zorder=8)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy, frameon=False)
    ax.add_artist(ab)
# 


# ЖД картинки
with get_sample_data(train_path) as file:
    train_img = plt.imread(file, format='png')
train_plot = metro_in_reg[(metro_in_reg['route_types'] == 'Эл')]
for xy in zip(train_plot.to_crs({'init': 'epsg:3857'}).geometry.x, 
              train_plot.to_crs({'init': 'epsg:3857'}).geometry.y):
    imagebox = OffsetImage(train_img, zoom=0.07,zorder=8)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy, frameon=False)
    ax.add_artist(ab)
# 


# Плашки для подписей текста
box_mgt = {'facecolor':'royalblue',    #  цвет области
       'edgecolor': 'grey',     #  цвет крайней линии
       'boxstyle': 'round', #  стиль области
         'alpha': 0.8, # прозрачность
         'pad': 0.2}    #  отступы

box_comm = {'facecolor':'seagreen',    #  цвет области
       'edgecolor': 'grey',     #  цвет крайней линии
       'boxstyle': 'round', #  стиль области
         'alpha': 0.8, # прозрачность
         'pad': 0.2}    #  отступы

box_region = {'facecolor':'tomato',    #  цвет области
       'edgecolor': 'grey',     #  цвет крайней линии
       'boxstyle': 'round', #  стиль области
         'alpha': 0.8, # прозрачность
         'pad': 0.2}    #  отступы

box_train = {'facecolor':'khaki',    #  цвет области
       'edgecolor': 'grey',     #  цвет крайней линии
       'boxstyle': 'round', #  стиль области
         'alpha': 0.8, # прозрачность
         'pad': 0.2}    #  отступы

if answer_color == '1':
    texts = [ax.text(row['geometry'].centroid.x,
                 row['geometry'].centroid.y,
                 row['rnc_separated'], 
                 bbox=box_mgt,
                 fontproperties=prop, color='white',
                 fontsize=8,zorder=10)
         for index, row in clusters_trips.to_crs({'init': 'epsg:3857'}).iterrows()]
#
elif answer_color == '2':
    
    texts = [ax.text(row['geometry'].centroid.x,
                 row['geometry'].centroid.y,
                 row['rnc_separated'], 
                 bbox=box_mgt if row['grouped_agency'] == 'Мосгортранс'
                 else box_comm if row['grouped_agency'] == 'commercial'
                 else box_region if row['grouped_agency'] == 'межсубъектные'
                 else box_train if row['grouped_agency'] == 'Прочие'
                 else box_train,
                 fontproperties=prop, color='white',
                 fontsize=8,zorder=10)
         for index, row in clusters_trips.to_crs({'init': 'epsg:3857'}).iterrows()]
#
else:
    texts = [ax.text(row['geometry'].centroid.x,
                 row['geometry'].centroid.y,
                 row['rnc_separated'], 
                 bbox=box_mgt,
                 fontproperties=prop, color='white',
                 fontsize=8,zorder=10)
         for index, row in clusters_trips.to_crs({'init': 'epsg:3857'}).iterrows()]
    print("Не выбран цвет")
# 


adjust_text(texts, expand_text=(1.9, 1.9), expand_align=(1.9, 1.9), arrowprops=dict(arrowstyle='->',color='black',lw=0.3))


# Раздвигание подписей___МЕТРО
texts_metro = [ax.text(row['geometry'].centroid.x-200,
            row['geometry'].centroid.y+50,
            row['stop_name'], fontproperties=prop, color='grey',
            fontsize=7,zorder=8)
            for index, row in metro_plot.to_crs({'init': 'epsg:3857'}).iterrows()]

# 
adjust_text(texts_metro, expand_text=(1.9, 1.9), expand_align=(1.9, 1.9))


# Раздвигание подписей___МЦД
texts_mcd = [ax.text(row['geometry'].centroid.x-200,
            row['geometry'].centroid.y+50,
            row['stop_name'], fontproperties=prop, color='grey',
            fontsize=7,zorder=8)
            for index, row in mcd_plot.to_crs({'init': 'epsg:3857'}).iterrows()]

# 
adjust_text(texts_mcd, expand_text=(1.9, 1.9), expand_align=(1.9, 1.9))



# Раздвигание подписей___ЖД
texts_train = [ax.text(row['geometry'].centroid.x-200,
            row['geometry'].centroid.y+50,
            row['stop_name'], fontproperties=prop, color='grey',
            fontsize=7,zorder=8)
            for index, row in train_plot.to_crs({'init': 'epsg:3857'}).iterrows()]

# 
adjust_text(texts_train, expand_text=(1.9, 1.9), expand_align=(1.9, 1.9))


# Легенда    
# trips = mlines.Line2D([], [], color='blue', marker='_',
#                               markersize=15, label='Трассы маршрутов')
# border_mun = mlines.Line2D([], [], color='black', marker='_',
#                               markersize=15, label='Граница района')
# ax.legend(handles=[trips, border_mun],
#           loc='lower right').set_zorder(20)#,zorder=12


# Масштаб
scale = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.1
scale = int(round(scale*0.01)*100)
scalebar = AnchoredSizeBar(ax.transData,
                           scale, '{scale} м.'.format(scale=scale), 'upper right', 
                           pad=1,
                           color='black',
                           frameon=False,
                           fontproperties=prop,
                           size_vertical=30)

ax.add_artist(scalebar)

# Сохранение изображения
ax.set_title('Маршруты, проходящие через\n{}'.format(mun_name), fontproperties=prop, fontsize=20)
plt.savefig('imgs/trips_{}_{}_{}_full.png'.format(mun_name,date_inp, answer_color))
plt.close()

#None

print()
print("Картинка с общим масштабом выгружена.")

###