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
    # Carregar os dados
X_train, X_test, y_train, y_test = load_data()

# Definir os nomes das features (substitua pela sua lista real de nomes de features)
feature_names = [
    'espaco_banco_traseiro', 'tipo_carroceria', 'consumo_cidade',
    'cilindros_motor', 'cilindradas_motor', 'tipo_motor',
    'cor_exterior', 'frota', 'chassi_danificado',
    'concessionaria_franqueada', 'marca_da_franquia',
    'espaco_banco_dianteiro', 'volume_tanque', 'tipo_combustivel',
    'historico_acidente', 'altura', 'consumo_estrada', 'cavalo_de_potencia',
    'cor_interior', 'ee_cabine', 'ee_novo', 'comprimento',
    'cor_listagem', 'nome_marca', 'maximo_assentos', 'quilometragem',
    'nome_modelo', 'qtd_proprietarios', 'potencia', 'preco', 'recuperado',
    'valor_economizado', 'avaliacao_vendedor', 'nome_vendedor',
    'titulo_roubo', 'torque', 'transmissao', 'exibicao_transmissao',
    'nome_versao', 'sistema_rodas', 'exibicao_sistema_rodas', 'entre_eixos',
    'largura', 'ano'
]
# Verificar se o número de features corresponde
if X_train.shape[1] == len(feature_names):
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
else:
    st.error(
        f"O número de features no X_train ({X_train.shape[1]}) não corresponde ao número de nomes de features ({len(feature_names)})."
    )
    st.stop()
# Função para treinar modelos com caching
@st.cache_resource
def train_models(X_train, y_train):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    trained_models = {}
    for name, model in models.items():
        st.write(f'Treinando {name}...')
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models
# Treinar os modelos
trained_models = train_models(X_train_df, y_train)

# Seção de seleção de modelos
st.sidebar.header('Selecione os Modelos para Análise de SHAP')
selected_models = st.sidebar.multiselect(
    'Modelos',
    options=list(trained_models.keys()),
    default=list(trained_models.keys())
)

# Seleção do número de amostras para SHAP
st.sidebar.header('Configurações SHAP')
sample_size = st.sidebar.slider(
    'Número de amostras para SHAP',
    min_value=100,
    max_value=1000,
    value=200,
    step=100
)
# Informações sobre SHAP
st.sidebar.info(
    'SHAP (SHapley Additive exPlanations) é uma técnica para interpretar modelos de machine learning, atribuindo a cada feature uma importância para a previsão.'
)

if not selected_models:
    st.sidebar.error("Por favor, selecione pelo menos um modelo para análise.")
    st.stop()
# Função para calcular SHAP e limitar o número de amostras com caching
@st.cache_data
def compute_shap_values(_model, explainer_type, X_sample):
    if explainer_type == 'Tree':
        explainer = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(X_sample)
    elif explainer_type == 'Kernel':
        # Limitar o número de amostras para KernelExplainer
        background = shap.sample(X_sample, 100, random_state=42)
        explainer = shap.KernelExplainer(_model.predict_proba, background)
        # Limitar ainda mais as amostras para KNN
        shap_values = explainer.shap_values(X_sample[:100])
    else:
        shap_values = None
    return shap_values

# Função para calcular a importância média absoluta das características
def calculate_mean_shap_importance(shap_values, feature_names):
    if isinstance(shap_values, list):
        # Para classificação binária, usar apenas a classe positiva (assumindo que é a segunda classe)
        shap_values = shap_values[1]
    mean_shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap_importance
    }).sort_values(by='importance', ascending=False)
    return importance_df

# Função para plotar a importância usando Plotly no Streamlit
def plot_feature_importance_plotly(importance_df, model_name):
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Importância das Features - {model_name}',
        labels={'importance': 'Importância', 'feature': 'Feature'},
        height=600
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

