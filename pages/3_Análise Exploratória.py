import os
import pandas as pd
from ydata_profiling import ProfileReport
import streamlit as st
import streamlit.components.v1 as components
from utils.build import build_header

def render_header():
    build_header(
        title='Análise Exploratória',
        hdr='# Análise Exploratória',
        p='''
            <p>A análise exploratória visa identificar relações entre as variáveis, extrair insights preliminares 
            e orientar a modelagem para os paradigmas mais comuns de machine learning. Esta etapa é considerada uma 
            das mais importantes no processo de análise de dados, pois permite entender como os dados estão relacionados, 
            extrair informações úteis e identificar o que precisa ser tratado antes de iniciar o processo de modelagem.</p>
        '''
    )

def build_profile(path, dataframe, report_title="Análise Exploratória de Dados"):
    if not os.path.exists(path):
        st.info(f"Gerando o relatório de perfil: {report_title}")
        profile = ProfileReport(dataframe, title=report_title)
        profile.to_file(path)
        st.success(f"Relatório salvo em: {path}")
    else:
        st.info(f"Relatório existente encontrado: {path}")
    render_profile(path)

def render_profile(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            report_html = file.read()
        components.html(report_html, height=1200, scrolling=True)
    except Exception as e:
        st.error(f"Erro ao carregar o relatório: {e}")

def open_profile(path, data_path='data/price_cars10k.parquet'):
    if os.path.exists(path):
        st.info(f"Carregando relatório de perfil de {path}...")
        render_profile(path)
    else:
        st.warning(f"Relatório não encontrado. Será gerado um novo.")
        try:
            dataframe = pd.read_parquet(data_path)
            build_profile(path, dataframe)
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")

def main():
    st.title('Dashboard de Análise Exploratória de Dados')
    render_header()
    open_profile('price_cars.html')

if __name__ == "__main__":
    main()
