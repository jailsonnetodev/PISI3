import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
from utils.build import build_header

build_header(
    title='Análise Exploratória',
    hdr='# ANÁLISE EXPLORATÓRIA',
    p=(
        '<p>A análise exploratória visa identificar relações entre as variáveis, '
        'extrair insights preliminares e encaminhar a modelagem para os paradigmas mais comuns '
        'de machine learning. Essa etapa é considerada uma das mais importantes no processo de '
        'análise de dados, pois a partir dela entendemos como os dados estão relacionados, '
        'extraímos informações úteis e identificamos o que precisa ser tratado antes da modelagem.</p>'
    )
)

def build_profile(path, dataframe):
    if not os.path.exists(path):
        profile = ProfileReport(dataframe, title="Preço Veículos")
        profile.to_file(path)
    components.html(open(path, 'r').read(), height=1200, scrolling=True)

def open_profile(path):
    if os.path.exists(path):
        try:
            components.html(open(path, 'r').read(), height=1200, scrolling=True)
        except Exception as e:
            print(f"Erro ao abrir o perfil: {e}")
    else:
        build_profile(path, pd.read_parquet('data/price_cars10k.parquet'))

open_profile('price_cars.html')

def plot_visualizations(df):
    plt.figure(figsize=(8, 12))
    sns.boxplot(x='preco', y='nome_marca', data=df)
    plt.title('Boxplot do Preço por Marca')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(df['quilometragem'], bins=20)
    plt.xlabel('Quilometragem')
    plt.ylabel('Frequência')
    plt.title('Histograma da Quilometragem')
    plt.show()

    df['tipo_carroceria'].value_counts().plot(kind='bar', figsize=(10, 6))
    plt.title('Contagem de Tipos de Carroceria')
    plt.xlabel('Tipo de Carroceria')
    plt.ylabel('Contagem')
    plt.show()

    df_grouped = df.groupby('ano')['preco'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(df_grouped['ano'], df_grouped['preco'])
    plt.xlabel('Ano')
    plt.ylabel('Preço Médio')
    plt.title('Preço Médio por Ano')
    plt.show()

    correlation_matrix = df[['preco', 'quilometragem', 'cavalos_de_potencia']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.show()

plot_visualizations(df)
