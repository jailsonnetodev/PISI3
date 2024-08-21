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
    title='Analise Dos Precos',
    hdr='# ANALISE DOS PRECOS E SUAS CORRELACOES',
    p='''
        <p> Aqui vamos realizar as primeiras observações dos dados e correlações entre algumas variaveis</p>
    '''
)
data = pd.read_parquet('data/price_cars100k.parquet')



price_by_years = data.groupby(by='ano')['preco'].mean()
price_by_years = pd.DataFrame(price_by_years).reset_index()
arr = np.array(price_by_years['ano'])
fig = px.line(data_frame=price_by_years,x='ano',y='preco')

line_graph(
    data=price_by_years,
    x='ano',
    y='preco',
    title='Grafico da Evolucao do Preco ao Longo dos Anos',
    p='''existe uma tendencia natural dos precos dos veiculos serem mais caros com passar dos anos, isso auja a entender que o veiculo mais novo pode ser ofertado com preco maior visto que depreciou menos com passar do tempo.'''
)


boxplot(
    data= data,
    x='preco',
    title='BoxPlot- Precos dos Veiculos',
    p='''Existe uma concetracao de veiculos entre 13 mil e 26mil , e possivel obter insights relevantes com essa analise'''
    
)

veiculos_anos = data['ano'].value_counts().sort_values().reset_index(name='Total')


bar(
    data= veiculos_anos,
    x='ano',
    y='Total',
    title='Total de veiculos x Ano de Fabricacao',
    p=''' Analise aqui'''
)





select_chart(
  data,
  x = 'preco',
  options = data.columns,
  type_graph=px.histogram,
  type_txt=f'Distribuição da',
)

# Preço médio por marca
average_price_by_make = data.groupby('marca')['preco'].mean().reset_index()
average_price_by_make.columns = ['marca', 'media_preco']
average_price_by_make.sort_values(by='media_preco',ascending=False)

top_avg_price_by_make = top_categories(
    data=average_price_by_make,
    top=10,
    label='marca'
)

bar(
    data=top_avg_price_by_make,
    x='marca',
    y='media_preco'
)

