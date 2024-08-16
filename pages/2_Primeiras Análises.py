import streamlit as st
import pandas as pd
import numpy as np
from utils.build import build_header
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.graph import *
from utils.build import  top_categories



build_header(
    title='Prinmeiras Analises',
    hdr='# PRIMEIRAS ANALISES E VISUALIZACOES',
    p='''
        <p> Aqui vamos realizar as primeiras observações dos dados e correlações entre algumas variaveis</p>
    '''
)
#data = pd.read_csv('data/price_cars.csv')
data = pd.read_parquet('data/price_cars_copy.parquet')


# data_group = data.groupby(['preco','marca', 'ano', 'modelo','estado','cidade','quilometragem']).size().reset_index(name='Total')
# data_group.sort_values('Total', ascending=True, inplace=True)
# data_cars = data[['ano', 'preco', 'marca', 'modelo','estado','cidade']]

price_by_years = data.groupby(by='Year')['Price'].mean()
price_by_years = pd.DataFrame(price_by_years).reset_index()
arr = np.array(price_by_years['Year'])
fig = px.line(data_frame=price_by_years,x='Year',y='Price')

fig


boxplot(
    data= data,
    x='Price',
    
)
scatter(
    data=data,
    x= 'Price',
    title='grafico de preco x quilometragem',
    y='Mileage'
)

x = data[data.Price < 100000].Price
hist(
    data=data,
    x='Price'
)
# data_filtered= top_categories(
#     data=data,
#     top= 10,
#     label='marca'
# )


# boxplot(
#     data= data_filtered,
#     title='BoxPlot da Marca por Quilometragem',
#     x='marca',
#     y='quilometragem',
#     p='''<p style='text-align:justify;'> As marcas tem uma concetração entre 20k e 60k  quilometros rodados, 
#     algumas marcas como Hyundai, Nissan, jeep e BMW tem veiculos passandos dos 100k de quilometragem </p>'''
# )



# hist(
#     title='HISTOGRAMA DA MARCA',
#     data = data,
#     x='marca'
# )


# #grafico de barras
# data_ano = data.groupby(['ano'])['preco'].size().reset_index()
# bar(
#     title='GRAFICO DE BARRAS, PRECO X ANO',
#     data = data_ano,
#     x='ano',
#     y='preco',
#     p=''' <p> Observamos que os precos dos veiculos tendem a ser mais caros com a variacao de ano entre 2014 e 2016</p>'''
# )






# select_chart(
#   data,
#   x = 'preco',
#   options = data.columns,
#   type_graph=px.histogram,
#   type_txt=f'Distribuição da',
# )





# select_chart(
#   data_group,
#   x = 'quilometragem',
#   options = data.columns,
#   type_graph=px.scatter,
#   type_txt=f'Distribuição da'
# )


# Contagem de veículos por estado
vehicles_by_state = data.groupby('State')['Price'].count().reset_index()
vehicles_by_state.columns = ['State', 'VehicleCount']

vehicles_by_state.sort_values(by='VehicleCount',ascending=True)
vehicles_by_state

bar(
    data=vehicles_by_state,
    x='State',
    y='VehicleCount'
)


filtrados = top_categories(
    data=data,
    top=10,
    label='cidade'
)

fig = px.bar(filtrados,x='cidade')
fig