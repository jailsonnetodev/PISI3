import os
import glob
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import shutil
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px

def delete_pkl_files(model_dirs):
    """
    Exclui todos os arquivos .pkl em diretórios específicos com confirmação.

    Parameters:
    - model_dirs (list): Lista de diretórios onde os arquivos .pkl estão armazenados.
    """
    pkl_files = []
    for dir in model_dirs:
        pattern = os.path.join(dir, '*', '.pkl')
        pkl_files.extend(glob.glob(pattern, recursive=True))
    
    if not pkl_files:
        st.warning("Nenhum arquivo .pkl encontrado para excluir nos diretórios especificados.")
        return
    
    confirm = st.checkbox("Confirma a exclusão de todos os arquivos .pkl?", key='delete_confirm')
    if confirm:
        for file_path in pkl_files:
            try:
                os.remove(file_path)
                st.success(f'Arquivo excluído: {file_path}')
            except Exception as e:
                st.error(f'Erro ao excluir {file_path}: {e}')
    else:
        st.info("Exclusão de arquivos .pkl cancelada.")

def backup_and_delete_pkl_files(model_dirs):
    """
    Move todos os arquivos .pkl para uma pasta de backup e depois os exclui com confirmação.

    Parameters:
    - model_dirs (list): Lista de diretórios onde os arquivos .pkl estão armazenados.
    """
    backup_dir = 'backup_pkl'
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f'backup_{timestamp}')
    os.makedirs(backup_path, exist_ok=True)

    pkl_files = []
    for dir in model_dirs:
        pattern = os.path.join(dir, '*', '.pkl')
        pkl_files.extend(glob.glob(pattern, recursive=True))
    
    if not pkl_files:
        st.warning("Nenhum arquivo .pkl encontrado para backup e exclusão.")
        return
    
    # Adiciona uma caixa de seleção para confirmação
    confirm = st.checkbox("Confirma o backup e exclusão de todos os arquivos .pkl?", key='backup_delete_confirm')
    if confirm:
        for file_path in pkl_files:
            try:
                # Define o caminho de destino no backup
                relative_path = os.path.relpath(file_path, dir)
                dest_path = os.path.join(backup_path, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Move o arquivo
                shutil.move(file_path, dest_path)
                st.success(f'Arquivo movido para backup: {dest_path}')
            except Exception as e:
                st.error(f'Erro ao mover {file_path}: {e}')
    else:
        st.info("Backup e exclusão de arquivos .pkl cancelada.")

def train_and_save_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Treina diferentes modelos e os salva como arquivos .pkl usando pickle.

    Parameters:
    - X_train, X_test, y_train, y_test: Dados para treinamento e teste.
    - feature_names (list): Lista de nomes das features.
    """
    models = {
        'Random_Forest': RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0),
        'Decision_Tree': DecisionTreeClassifier(criterion='entropy'),
        'KNN': KNeighborsClassifier(n_neighbors=10, weights='distance', p=1)
    }

    for model_name, model in models.items():
        # Treinar o modelo
        model.fit(X_train, y_train)
        
        # Salvar o modelo usando pickle
        model_dir = 'models'  # Diretório onde os modelos serão salvos
        os.makedirs(model_dir, exist_ok=True)  # Cria o diretório se não existir
        model_path = os.path.join(model_dir, f'{model_name}.pkl')
        try:
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            st.success(f'Modelo {model_name} treinado e salvo em {model_path}')
        except Exception as e:
            st.error(f'Erro ao salvar o modelo {model_name}: {e}')

def load_model(model_path):
    """
    Carrega um modelo salvo usando pickle.

    Parameters:
    - model_path (str): Caminho para o arquivo .pkl do modelo.

    Returns:
    - model: Modelo carregado.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f'Erro ao carregar o modelo de {model_path}: {e}')
        return None

def calculate_metrics(model, X_train, X_test, y_train, y_test):
    """
    Calcula as métricas de desempenho para o modelo nos conjuntos de treinamento e teste.

    Parameters:
    - model: Modelo treinado.
    - X_train, X_test: Dados de features para treinamento e teste.
    - y_train, y_test: Dados de target para treinamento e teste.

    Returns:
    - metrics_train (dict): Métricas no conjunto de treinamento.
    - metrics_test (dict): Métricas no conjunto de teste.
    """
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

def display_metrics(model_name, metrics_train, metrics_test):
    """
    Exibe as métricas em formato de tabela no Streamlit.

    Parameters:
    - model_name (str): Nome do modelo.
    - metrics_train (dict): Métricas no conjunto de treinamento.
    - metrics_test (dict): Métricas no conjunto de teste.
    """
    metrics_df = pd.DataFrame({
        'Métrica': list(metrics_train.keys()),
        'Treinamento': list(metrics_train.values()),
        'Teste': list(metrics_test.values())
    })

    st.subheader(f'Métricas do Modelo: {model_name}')
    st.table(metrics_df)

def plot_feature_importance(model, feature_names, model_name, X_test=None, y_test=None):
    """
    Plota a importância das features usando Plotly.

    Parameters:
    - model: Modelo treinado.
    - feature_names (list): Lista de nomes das features.
    - model_name (str): Nome do modelo.
    - X_test, y_test: Dados de teste para modelos que não possuem feature_importances_.
    """
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
        if X_test is None or y_test is None:
            st.error(f"Dados de teste não fornecidos para calcular a importância por permutação do modelo {model_name}.")
            return
        st.info(f"O modelo {model_name} não possui feature_importances_. Usando Importância por Permutação.")
        from sklearn.inspection import permutation_importance
        try:
            result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring='accuracy')
            importances = result.importances_mean
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importância': importances
            }).sort_values(by='Importância', ascending=False)
        except Exception as e:
            st.error(f"Erro ao calcular a importância por permutação para {model_name}: {e}")
            return
    
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