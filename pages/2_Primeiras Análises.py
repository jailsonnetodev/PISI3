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
    title='Prinmeiras Analises',
    hdr='# PRIMEIRAS ANALISES E VISUALIZACOES',
    p='''
        <p> Aqui vamos realizar as primeiras observações dos dados e correlações entre algumas variaveis</p>
    '''
)


data = pd.read_parquet('data/price_cars_copy.parquet')


col1,_,col3 = st.columns(3)
st.divider()
lin = data.shape[0]
col = data.shape[1]
col1.metric("Registros",lin)
col3.metric("Colunas",col)


price_by_years = data.groupby(by='Year')['Price'].mean()
price_by_years = pd.DataFrame(price_by_years).reset_index()
arr = np.array(price_by_years['Year'])
fig = px.line(data_frame=price_by_years,x='Year',y='Price')

fig


boxplot(
    data= data,
    x='Price',
    
)
scatter(
    data=data,
    x= 'Price',
    title='grafico de preco x quilometragem',
    y='Mileage'
)

x = data[data.Price < 100000].Price
hist(
    data=data,
    x='Price'
)


veiculos_anos = data['Year'].value_counts().sort_values().reset_index(name='Total')
fig2 = px.bar(veiculos_anos, x='Year',y='Total')
fig2





categorics_size = data.groupby(['Make']).size()
categorics_means = data.groupby('Make')['Price'].mean()
categorics_total = categorics_size.reset_index(name="Total")
categorics_sorted = categorics_total.sort_values(by='Total', ascending=False)



# temos uma concentracao maior de carros nas marcas populares conm ford chevrolet nissan, honda 
fig3 = px.bar(categorics_sorted,x='Make',y='Total')
fig3
