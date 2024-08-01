import pandas as pd
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from utils.build import build_header
import streamlit as st
import os



build_header(
    title='Analise Exploratoria',
    hdr='# ANALISE EXPLORATORIA',
    p='''
        <p>A análise exploratória visa identificar relações entre as variáveis, extrair insights preliminares e encaminhar a modelagem para os paradigmas mais comuns de machine learning, essa etapa e considerada uma das etapas mais importantes no processo de analise de dados pois e apartir dela que vemos o como os dados estao relacionados, extraimos informacoes uteis e o que teremos que tratar antes de iniciar o processo modelagem para uso de machine learning.</p>
    '''
)



def build_profile(path, dataframe):
  if not(os.path.exists(path)):
    profile = ProfileReport(dataframe, title=f"Preco Veiculos")
    profile.to_file(path)
  components.html(open(path, 'r').read(), height= 1200, scrolling=True)

def open_profile(path):
  if os.path.exists(path):
    try:
      components.html(open(path, 'r').read(), height= 1200, scrolling=True)
    except:
      print('invalid path.')
  else:
    build_profile(path, pd.read_parquet('data/price_cars10k.parquet'))


open_profile('price_cars.html')



