import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.build import build_header, top_categories
from utils.graph import boxplot, line_graph


build_header(
    title='Analise Dos Precos',
    hdr='# ANALISE DA QUILOMETRAGEM ',
    p='''
        <p> Aqui vamos realizar as primeiras observações dos dados e correlações entre algumas variaveis</p>
    '''

)

data = pd.read_parquet('data/usedcars_usa.parquet')

fig8 = px.box(data,x='quilometragem',y='dias_no_mercado_label')
fig8

group_km =  data.groupby(['dias_no_mercado'])['quilometragem'].size().reset_index()
fig10 = px.line(group_km, x='dias_no_mercado',y='quilometragem')
fig10




group_ano_km_preco = data.groupby(['ano','quilometragem'])['preco'].mean().reset_index()
fig12 = px.bar(group_ano_km_preco, x="ano", y=["quilometragem", "preco"], barmode='group',
            title="Total de Veículos e Preço Médio por Ano")
st.plotly_chart(fig12)
