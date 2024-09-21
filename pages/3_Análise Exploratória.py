import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
from utils.build import build_header

# build_header(
#     title='Análise Exploratória',
#     hdr='# ANÁLISE EXPLORATÓRIA',
#     p=(
#         '<p>A análise exploratória visa identificar relações entre as variáveis, '
#         'extrair insights preliminares e encaminhar a modelagem para os paradigmas mais comuns '
#         'de machine learning. Essa etapa é considerada uma das mais importantes no processo de '
#         'análise de dados, pois a partir dela entendemos como os dados estão relacionados, '
#         'extraímos informações úteis e identificamos o que precisa ser tratado antes da modelagem.</p>'
#     )
# )

def build_profile(path, dataframe):
    if not os.path.exists(path):
        profile = ProfileReport(dataframe, title="Tempo de Permancencia")
        profile.to_file(path)
    components.html(open(path, 'r').read(), height=1200, scrolling=True)

def open_profile(path):
    if os.path.exists(path):
        try:
            components.html(open(path, 'r').read(), height=1200, scrolling=True)
        except Exception as e:
            print(f"Erro ao abrir o perfil: {e}")
    else:
        build_profile(path, pd.read_parquet('data/usedcars_usa20k.parquet'))

open_profile('relatorio_analise_exploratoria.html')


