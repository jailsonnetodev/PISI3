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

px.line(group_by_year, x='ano', y='preco', color='dias_no_mercado_label')
group_km =  df.groupby(['dias_no_mercado_label'])['quilometragem'].size().reset_index()
px.line(group_km, x='dias_no_mercado_label',y='quilometragem')
group_km =  df.groupby(['dias_no_mercado'])['quilometragem'].size().reset_index()
px.line(group_km, x='dias_no_mercado',y='quilometragem')
group_by_price_year =  df.groupby(['ano'])['preco'].mean().reset_index()
px.line(group_by_price_year, x='ano',y='preco')
group_km =  df.groupby(['dias_no_mercado'])['quilometragem'].size().reset_index()
px.line(group_km, x='dias_no_mercado',y='quilometragem')
px.scatter(df, x='dias_no_mercado',y='quilometragem', color='ee_novo')
group_make =  df.groupby(['nome_marca'])['preco'].size().reset_index().sort_values(by='preco',ascending=False)
px.bar(group_make, x='nome_marca', y='preco')
fig = px.scatter(df[['dias_no_mercado','preco']], x="dias_no_mercado", y="preco",size="preco",size_max=60)
fig.show()
group_cor =  df.groupby(['cor_listagem']).size().reset_index(name='Total')
px.bar(group_cor, x='cor_listagem', y='Total')
