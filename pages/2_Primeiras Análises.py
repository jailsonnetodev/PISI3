import streamlit as st
import pandas as pd
import numpy as np
from utils.build import build_header
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.graph import boxplot,scatter,treemap,hist,bar,select_chart
from utils.build import  top_categories



build_header(
    title='Prinmeiras Analises',
    hdr='# PRIMEIRAS ANALISES E VISUALIZACOES',
    p='''
        <p> Aqui vamos realizar as primeiras observações dos dados e correlações entre algumas variaveis</p>
    '''
)
data = pd.read_parquet('data/price_cars10k.parquet')


data_group = data.groupby(['preco','marca', 'ano', 'modelo','estado','cidade','quilometragem']).size().reset_index(name='Total')
data_group.sort_values('Total', ascending=True, inplace=True)
data_cars = data[['ano', 'preco', 'marca', 'modelo','estado','cidade']]



data_filtered= top_categories(
    data=data,
    top= 10,
    label='marca'
)


boxplot(
    data= data_filtered,
    title='BoxPlot da Marca por Quilometragem',
    x='marca',
    y='quilometragem',
    p='''<p style='text-align:justify;'> As marcas tem uma concetração entre 20k e 60k  quilometros rodados, 
    algumas marcas como Hyundai, Nissan, jeep e BMW tem veiculos passandos dos 100k de quilometragem </p>'''
)



hist(
    title='HISTOGRAMA DA MARCA',
    data = data,
    x='marca'
)


#grafico de barras
data_ano = data.groupby(['ano'])['preco'].size().reset_index()
bar(
    title='GRAFICO DE BARRAS, PRECO X ANO',
    data = data_ano,
    x='ano',
    y='preco',
    p=''' <p> Observamos que os precos dos veiculos tendem a ser mais caros com a variacao de ano entre 2014 e 2016</p>'''
)



select_chart(
  data,
  x = 'preco',
  options = data.columns,
  type_graph=px.histogram,
  type_txt=f'Distribuição da',
)





select_chart(
  data_group,
  x = 'quilometragem',
  options = data.columns,
  type_graph=px.scatter,
  type_txt=f'Distribuição da'
)


    