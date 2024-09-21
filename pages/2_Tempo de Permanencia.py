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



data_make = top_categories(
    data=data,
    top=5,
    label='nome_marca'
)



def category_label(list_classes, old_values):
  list_classes.sort()
  new_list = [None] * len(old_values)
  for i, old in enumerate(old_values):
    for index, value in enumerate(list_classes):
      if(len(list_classes)-1 == index):
        new_list[i] = f'mais de {value} dias'
      else:
        if(value <= old < list_classes[index+1]):
          new_list[i] = f'{value} a {list_classes[index+1]} dias'
          break
  return new_list

def group_label(data):
  test_lista= [x*10 for x in range(19)]
  lista_stay_category = category_label(test_lista, data['dias_no_mercado'].tolist())
  data['dias_no_mercado'] = lista_stay_category
  return data
data_modify = group_label(data)



select_graph(
  data_modify,
  x = 'quilometragem',
  options = data_modify.columns,
  type_graph=px.scatter,
  type_txt='Gráfico de Barras'
)


def filter_top_n(df, column1, top_n1, column2, top_n2, column3, top_n3):
    top_n_column1 = df[column1].value_counts().nlargest(top_n1).index
    filtered_df = df[df[column1].isin(top_n_column1)]
    top_n_column2 = filtered_df[column2].value_counts().nlargest(top_n2).index
    filtered_df = filtered_df[filtered_df[column2].isin(top_n_column2)]
    top_n_column3 = filtered_df[column3].value_counts().nlargest(top_n3).index
    filtered_df = filtered_df[filtered_df[column3].isin(top_n_column3)]

    return filtered_df
# Filtrar as top N cidades, vendedores e modelos
n_top_cidade = st.slider('Selecione a quantidade de cidades', 5,20, 5)
n_top_vendedor = st.slider('Selecione a quantidade de vendedores', 5,10, 5)
n_top_modelo = st.slider('Selecione a quantidade de modelos', 5,20,5)

top_filtered_df = filter_top_n(data, 'cidade',n_top_cidade, 'nome_modelo',n_top_modelo, 'nome_vendedor', n_top_vendedor)

fig = px.treemap(top_filtered_df, path=['cidade', 'nome_vendedor', 'nome_modelo'], values='preco',
                  color='dias_no_mercado',color_continuous_scale='Viridis',title='Treemap das Cidades, Vendedores e Modelos', hover_data=['ano'])
st.write(fig)



fig = px.strip(data, x="dias_no_mercado", y="chassi_danificado", orientation="h", color="frota")
st.plotly_chart(fig)