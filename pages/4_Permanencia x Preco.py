import streamlit as st
import pandas as pd
import numpy as np
from utils.build import build_header
import plotly.express as px
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
data = pd.read_parquet('data/usedcars_usa100k.parquet')




df_preco_ano = data.groupby('ano')['preco'].mean().reset_index()
fig6 = px.line(df_preco_ano, x='ano', y='preco', title='Preço médio por ano')
st.plotly_chart(fig6)



group_make =  data.groupby(['nome_marca'])['preco'].size().reset_index().sort_values(by='preco',ascending=False)

fig11 = px.bar(group_make, x='nome_marca', y='preco')
fig11

group_by_year = data.groupby(['ano','dias_no_mercado_label']).agg({'preco': np.mean}).reset_index()
fig9 = px.line(group_by_year, x='ano', y='preco', color='dias_no_mercado_label')
fig9
