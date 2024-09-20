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
