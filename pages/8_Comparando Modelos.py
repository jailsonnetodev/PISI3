import pandas as pd
import numpy as np
import plotly.express as px
import streamlit  as st
import os 
import shap
import pickle
from utils.transform_pkl import main
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.classifier import ConfusionMatrix
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if not(os.path.isfile('data/usedcars_usa.pkl')):
    print('iniciando...')
    main()

with open('data/usedcars_usa.pkl', 'rb') as f:
    X_training, X_test, y_training, y_test = pickle.load(f)



import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# Função para calcular métricas
def calculate_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics_train = {
        'Acurácia': accuracy_score(y_train, y_train_pred),
        'Precisão': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    }

    metrics_test = {
        'Acurácia': accuracy_score(y_test, y_test_pred),
        'Precisão': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    }

    return metrics_train, metrics_test

# Função para exibir métricas no Streamlit
def display_metrics(model_name, metrics_train, metrics_test):
    metrics_df = pd.DataFrame({
        'Métrica': list(metrics_train.keys()),
        'Treinamento': list(metrics_train.values()),
        'Teste': list(metrics_test.values())
    })

    st.subheader(f'Métricas do Modelo: {model_name}')
    st.table(metrics_df)

# Função para plotar a importância das features
def plot_feature_importance(model, model_name, X_train, y_train, X_test, y_test, feature_names):
    st.markdown(f"### Importância das Features: {model_name}")
    
    if hasattr(model, 'feature_importances_'):
        # Modelos como Random Forest e Decision Tree possuem feature_importances_
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importância': importances
        }).sort_values(by='Importância', ascending=False)
    else:
        # Para modelos como KNN, utilizamos a importância por permutação
        st.info(f"O modelo {model_name} não possui feature_importances_. Usando Importância por Permutação.")
        result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring='accuracy')
        importances = result.importances_mean
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importância': importances
        }).sort_values(by='Importância', ascending=False)
    
    # Plotando com Plotly
    fig = px.bar(
        importance_df,
        x='Importância',
        y='Feature',
        orientation='h',
        title=f'Importância das Features: {model_name}',
        labels={'Importância': 'Importância', 'Feature': 'Feature'},
        height=400
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig)




# Lista de modelos
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Título da aplicação
st.title('Comparação de Modelos de Classificação')
feature_names = ['espaco_banco_traseiro', 'tipo_carroceria', 'consumo_cidade',
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
       'largura', 'ano']

# Loop para exibir métricas e importâncias para cada modelo
for model_name, model in models.items():
    with st.expander(f"🔍 {model_name}"):
        metrics_train, metrics_test = calculate_metrics(model, X_training, X_test, y_training, y_test)
        display_metrics(model_name, metrics_train, metrics_test)
        plot_feature_importance(model, model_name, X_training, y_training, X_test, y_test, feature_names)