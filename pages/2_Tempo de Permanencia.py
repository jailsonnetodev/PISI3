import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.build import build_header
from utils.graph import boxplot, scatter, hist

build_header(
    title='Primeiras Análises',
    hdr='# PRIMEIRAS ANÁLISES E VISUALIZAÇÕES',
    p='<p>Aqui vamos realizar as primeiras observações dos dados e correlações entre algumas variáveis.</p>'
)
plt.figure(figsize = (25 , 10))
sns.heatmap(df.select_dtypes('float' , 'int').corr() , annot = True)
plt.xticks(rotation = 45);

fig = px.scatter(df, x="preco", y="dias_no_mercado", color="qtd_proprietarios",
                 hover_data=["nome_modelo", "ano"], title="Tempo de Venda vs. Preço")
fig.show()
fig = px.box(df, x="qtd_proprietarios", y="dias_no_mercado",
            color="qtd_proprietarios", title="Dias no Mercado por Quantidade de Proprietários")
fig.show()
ig = px.histogram(df, x="dias_no_mercado", color="preco",
                  nbins=20, title="Distribuição de Dias no Mercado por Faixa de Preço")
fig.show()
fig = px.scatter(df, x="preco", y="dias_no_mercado", color="qtd_proprietarios",
                 hover_data=["nome_modelo", "ano"], title="Tempo de Venda vs. Preço")
fig.show()

px.box(df,x='quilometragem',y='dias_no_mercado_label')
group_by_year = df.groupby(['ano','dias_no_mercado_label']).agg({'preco': np.mean}).reset_index()
