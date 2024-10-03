import shap
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import os
import streamlit as st
from utils.transform_pkl import main
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Configurações iniciais do Streamlit
st.set_page_config(page_title='Análise SHAP para Modelos', layout='wide')

st.title('Análise de SHAP para Modelos de Classificação')

# Função para carregar dados com caching
@st.cache_data
def load_data(data_path='data/usedcars_usa.pkl'):
    if not os.path.isfile(data_path):
        st.warning('Dados não encontrados. Iniciando transformação dos dados...')
        main()
    with open(data_path, 'rb') as f:
        X_training, X_test, y_training, y_test = pickle.load(f)
    return X_training, X_test, y_training, y_test
