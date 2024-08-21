import streamlit as st
import pandas as pd
import numpy as np
from utils.build import build_header
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.graph import boxplot,scatter,treemap,hist,bar,select_chart,line_graph
from utils.build import  top_categories


build_header(
    title='Analise Da Quilometragem',
    hdr='# ANALISE DA QUILOMETRAGEM DOS VEICULOS',
    p='''
        <p> Aqui vamos realizar as primeiras observações dos dados e correlações entre algumas variaveis</p>
    '''
)

st.divider()

data = pd.read_parquet('data/price_cars.parquet')
new_data = pd.read_csv('data/price_cars.csv')

data_filtered= top_categories(
    data=data,
    top= 10,
    label='marca'
)

boxplot(
    data=data,
    x='quilometragem',
    title='Boxplot - Quilometragem dos veiculos',
    p="""<p style= 'text-align:justify';> Nesse grafico de boxplot é possivel veririficar que os veiculos tem uma concetração de dados entre 23 mil quilometros e 70 mil  </p>"""
)



boxplot(
    data= data_filtered,
    title='BoxPlot da Marca por Quilometragem',
    x='marca',
    y='quilometragem',
    p='''<p style='text-align:justify;'> As marcas tem uma concetração entre 20k e 60k  quilometros rodados, 
    algumas marcas como Hyundai, Nissan, jeep e BMW tem veiculos passandos dos 100k de quilometragem </p>'''
)



# Criando faixas de quilometragem
new_data['km_intervalo'] = pd.cut(new_data['Mileage'], bins=[0, 50000, 100000, 150000, 200000, 300000], 
                            labels=['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k+'])

# Preço médio por faixa de quilometragem
average_price_by_mileage = new_data.groupby('km_intervalo')['Price'].mean().reset_index()
average_price_by_mileage.columns = ['KM-intervalo', 'media-preco']


line_graph(
    data=average_price_by_mileage,
    x='KM-intervalo',
    y='media-preco',
    title='Grafico de linha da relacao Preco X Quilometragem',
    p="""<p style='text-align=justify;'>Vemos nesse grafico de linha que quanto maior a Km do veiculo o preco diminui na proporcao
    indicando que ha uma relacao inversao .
    </p>"""
)


