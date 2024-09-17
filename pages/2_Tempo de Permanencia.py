import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.build import *
from utils.graph import boxplot, scatter, hist

build_header(
    title='Primeiras Análises',
    hdr='# Tempo de Permanencia do Veiculo no Mercado de Usados',
    p='<p>Neste trabalho será possivel identificar as caracteristicas fisicas, funcionais e nao funcionais do veiculo com objetivo indeiticar esse fatores que influencia o tempo de permancencia do veiculo listado no site e disponivel no mercado.</p>'
)

data = pd.read_parquet('data/usedcars_usa.parquet')


col1,col2 = st.columns([0.3,0.8])

with col1:
    top_categorics = st.slider('Selecione quantidade de marcas que deseja visualizar:', 0, 20, 5)
with col2:
    data_filtered = top_categories(
    data = data,
    top=top_categorics,
    label='nome_marca'
    )
    new_fig = px.bar(data_filtered, x='nome_marca', y='preco')
    new_fig


group_make =  data.groupby(['nome_marca'])['preco'].mean().reset_index().sort_values(by='preco',ascending=False)



fig =plt.figure(figsize = (25 , 10))
sns.heatmap(data.select_dtypes('float' , 'int').corr() , annot = True)
plt.xticks(rotation = 45);
st.pyplot(fig)


df_ano = data.groupby('ano')['dias_no_mercado'].mean().reset_index()
fig = px.line(df_ano, x='ano', y='dias_no_mercado',
              title='Relação entre ano de fabricação e tempo de permanência')
st.plotly_chart(fig)


df_acidentes = data.groupby('historico_acidente')['dias_no_mercado'].mean().reset_index()
fig2 = px.bar(df_acidentes, x='historico_acidente', y='dias_no_mercado',
             title='Impacto de histórico de acidentes no tempo de venda')
st.plotly_chart(fig2)



df_cor = data.groupby('cor_exterior')['dias_no_mercado'].mean().reset_index()

datafiltered = top_categories(
    data=df_cor,
    top=10,
    label='cor_exterior'
)
fig3 = px.bar(datafiltered, x='cor_exterior', y='dias_no_mercado',
             title='Relação entre cor do veículo e tempo de venda')
st.plotly_chart(fig3)



df_combustivel = data.groupby('tipo_combustivel')['preco'].mean().reset_index()
fig4 = px.bar(df_combustivel, x='tipo_combustivel', y='preco', title='Preferência por tipo de combustível')
title=('Relação entre cor do veículo e tempo de venda')
st.plotly_chart(fig4)


df_veiculos_ano = data.groupby(['ano','dias_no_mercado_label']).size().reset_index(name='total_veiculos')
fig5 = px.bar(df_veiculos_ano, x='ano', y='total_veiculos', title='Total de veículos por ano', color='dias_no_mercado_label')
st.plotly_chart(fig5)



seats_df = data.groupby(['maximo_assentos' , 'tipo_carroceria'])['tipo_carroceria'].count().to_frame().rename(columns = {'tipo_carroceria':'Count'}).reset_index()
fig7 = px.bar(seats_df, x="tipo_carroceria", y="Count", animation_frame="maximo_assentos", animation_group="maximo_assentos",
            color="Count")
fig7["layout"].pop("updatemenus") # optional, drop animation buttons
st.plotly_chart(fig7)






data_make = top_categories(
    data=data,
    top=5,
    label='nome_marca'
)


fig = px.bar(data_make, x='nome_marca', y='dias_no_mercado', color='ano',
             title='Tempo de permanência no mercado por modelo e ano')
fig
