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

data = pd.read_parquet('data/price_cars_copy.parquet')

col1, _, col3 = st.columns(3)
st.divider()

num_registros = data.shape[0]
num_colunas = data.shape[1]
col1.metric("Registros", num_registros)
col3.metric("Colunas", num_colunas)

price_by_years = data.groupby(by='Year')['Price'].mean().reset_index()
fig_price_by_years = px.line(price_by_years, x='Year', y='Price', title='Preço Médio por Ano')
st.plotly_chart(fig_price_by_years)

st.subheader('Distribuição dos Preços')
boxplot(data=data, x='Price')
st.pyplot()

st.subheader('Preço vs Quilometragem')
scatter(data=data, x='Price', y='Mileage', title='Gráfico de Preço x Quilometragem')
st.pyplot()

st.subheader('Distribuição dos Preços (até R$100.000)')
filtered_prices = data[data['Price'] < 100000]['Price']
hist(data=filtered_prices, x='Price')
st.pyplot()

st.subheader('Número de Veículos por Ano')
veiculos_anos = data['Year'].value_counts().sort_index().reset_index(name='Total')
fig_veiculos_anos = px.bar(veiculos_anos, x='index', y='Total', labels={'index': 'Ano', 'Total': 'Número de Veículos'}, title='Número de Veículos por Ano')
st.plotly_chart(fig_veiculos_anos)

st.subheader('Número de Veículos por Marca')
categorics_size = data['Make'].value_counts().reset_index(name='Total')
categorics_size.columns = ['Marca', 'Total']
categorics_sorted = categorics_size.sort_values(by='Total', ascending=False)
fig_categorics = px.bar(categorics_sorted, x='Marca', y='Total', title='Número de Veículos por Marca')
st.plotly_chart(fig_categorics)
