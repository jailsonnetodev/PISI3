import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.build import build_header, top_categories
from utils.graph import boxplot, line_graph

def setup_analysis():
    build_header(
        title='Análise da Quilometragem',
        hdr='# Análise da Quilometragem dos Veículos',
        p='''
            <p>Aqui vamos realizar as primeiras observações dos dados e analisar correlações entre algumas variáveis.</p>
        '''
    )
    st.divider()

def load_data(parquet_path, csv_path):
    try:
        data = pd.read_parquet(parquet_path)
        new_data = pd.read_csv(csv_path)
        return data, new_data
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None, None

def display_boxplots(data, data_filtered):
    boxplot(
        data=data,
        x='quilometragem',
        title='Boxplot - Quilometragem dos Veículos',
        p="""<p style='text-align:justify;'>Esse gráfico de boxplot mostra que a maioria dos veículos tem quilometragem 
        concentrada entre 23 mil e 70 mil quilômetros.</p>"""
    )

    boxplot(
        data=data_filtered,
        title='BoxPlot da Marca por Quilometragem',
        x='marca',
        y='quilometragem',
        p="""<p style='text-align:justify;'>As marcas apresentam quilometragem concentrada entre 20 mil e 60 mil quilômetros. 
        Marcas como Hyundai, Nissan, Jeep e BMW têm veículos com quilometragem superior a 100 mil quilômetros.</p>"""
    )

def display_line_graph(new_data):
    new_data['km_intervalo'] = pd.cut(new_data['Mileage'], bins=[0, 50000, 100000, 150000, 200000, 300000], 
                                      labels=['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k+'])
    
    average_price_by_mileage = new_data.groupby('km_intervalo')['Price'].mean().reset_index()
    average_price_by_mileage.columns = ['KM-intervalo', 'media-preco']
    
    line_graph(
        data=average_price_by_mileage,
        x='KM-intervalo',
        y='media-preco',
        title='Gráfico de Linha - Relação Preço x Quilometragem',
        p="""<p style='text-align:justify;'>Este gráfico de linha mostra que, à medida que a quilometragem dos veículos 
        aumenta, o preço tende a diminuir, indicando uma relação inversa.</p>"""
    )

def main():
    setup_analysis()
    
    data, new_data = load_data('data/price_cars.parquet', 'data/price_cars.csv')
    
    if data is not None and new_data is not None:
        data_filtered = top_categories(data=data, top=10, label='marca')
        display_boxplots(data, data_filtered)
        display_line_graph(new_data)

if __name__ == "__main__":
    main()
