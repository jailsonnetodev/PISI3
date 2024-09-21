import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.build import *
from utils.graph import *

build_header(
    title='Primeiras Análises',
    hdr='# Tempo de Permanencia do Veiculo no Mercado de Usados',
    p='<p>Neste trabalho será possivel identificar as caracteristicas fisicas, funcionais e nao funcionais do veiculo com objetivo indeiticar esse fatores que influencia o tempo de permancencia do veiculo listado no site e disponivel no mercado.</p>'
)

data = pd.read_parquet('data/usedcars_usa100k.parquet')


col1,col2 = st.columns([0.3,0.8])

with col1:
    top_categorics = st.slider('Selecione quantidade de marcas que deseja visualizar:', 0, 20, 5)
with col2:
    data_filtered = top_categories(
    data = data,
    top=top_categorics,
    label='nome_marca'
    )
    bar(
      data=data_filtered,
      x='nome_marca',
      y='preco',
      title='Grafico de Barras da Media de Preco por Marca',
      p='<p></p>'
    )
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

