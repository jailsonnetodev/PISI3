# app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from utils.model_utils import (
    delete_pkl_files, 
    backup_and_delete_pkl_files, 
    train_and_save_models, 
    load_model, 
    calculate_metrics, 
    display_metrics, 
    plot_feature_importance
)
import os


def label_encoder(x_data):
  le=LabelEncoder()
  for col in x_data:
    if x_data[col].dtypes == 'object' or x_data[col].dtypes == 'category':
      x_data[col] = pd.DataFrame(le.fit_transform(x_data[col]))
  x_data = x_data.values
  return x_data


def standard(x_data):
  scaler = StandardScaler()
  x_data = scaler.fit_transform(x_data)

  return x_data

def main():
    st.set_page_config(page_title='Treinamento e Gerenciamento de Modelos', layout='wide')
    st.title('Treinamento e Gerenciamento de Modelos de Classificação')
    
    # Seção de Upload e Preparação dos Dados
    st.header('Upload de Dados Pré-Processados')
    uploaded_file = st.file_uploader("Escolha um arquivo .parquet com os dados pré-processados", type="parquet")
    if uploaded_file is not None:
        try:
            df = pd.read_parquet(uploaded_file)
            if isinstance(df, pd.DataFrame):
                st.write("**Dados Carregados:**")
                st.dataframe(df.head())
                
                all_columns = df.columns.tolist()
                target_column = st.selectbox("Selecione a coluna alvo (target)", all_columns)
                feature_columns = st.multiselect("Selecione as colunas de features", [col for col in all_columns if col != target_column])
                
                if st.button("Treinar Modelos"):
                    if not feature_columns:
                        st.error("Por favor, selecione pelo menos uma feature.")
                    else:

                        X = df[feature_columns]
                        y = df[target_column]
                        X = label_encoder(X)
                        X = standard(X)
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.3, random_state=42
                        )
                        st.write("**Features Selecionadas:**", feature_columns)
                        st.write("**Coluna Alvo:**", target_column)
                        st.warning("Arquivos de modelos antigos serão excluídos antes do treinamento.")
                    
                        model_directories = ['models']  # Adicione outros diretórios, se necessário
                        

                        delete_pkl_files(model_directories)
                        

                        train_and_save_models(X_train, X_test, y_train, y_test, feature_columns)
                        
                        st.success("Modelos treinados e salvos com sucesso!")

                        st.header('Métricas de Desempenho dos Modelos')
                        for model_name in ['Random_Forest', 'Decision_Tree', 'KNN']:
                            model_path = os.path.join('models', f'{model_name}.pkl')
                            if os.path.exists(model_path):
                                model = load_model(model_path)
                                if model:
                                    metrics_train, metrics_test = calculate_metrics(model, X_train, X_test, y_train, y_test)
                                    display_metrics(model_name, metrics_train, metrics_test)
                                    plot_feature_importance(model, feature_columns, model_name, X_test, y_test)
                            else:
                                st.warning(f"Modelo {model_name} não encontrado em {model_path}.")
            else:
                st.error("O arquivo .parquet carregado não contém um DataFrame do Pandas.")
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo .parquet: {e}")
    
    st.markdown("---")
    

    st.header('Gerenciamento de Modelos')

    st.subheader('Excluir Modelos Salvos')
    st.write("Clique no botão abaixo para excluir todos os modelos salvos (.pkl). **Atenção:** Esta ação não pode ser desfeita.")

    if st.button("Excluir Modelos"):
        st.warning("Você está prestes a excluir todos os arquivos .pkl nos diretórios especificados.")
        model_directories = ['models']  # Adicione outros diretórios, se necessário
        delete_pkl_files(model_directories)
    
    st.markdown("---")
    
    
    st.subheader('Backup e Excluir Modelos Salvos')
    st.write("Clique no botão abaixo para mover todos os modelos salvos (.pkl) para uma pasta de backup antes de excluí-los.")
    
    
    if st.button("Backup e Excluir Modelos"):
        st.warning("Você está prestes a mover todos os arquivos .pkl para a pasta de backup e excluí-los dos diretórios originais.")
        model_directories = ['models']  
        backup_and_delete_pkl_files(model_directories)
    
    st.markdown("---")
    
    
    st.header('Carregar e Usar Modelos Salvos')
    model_path_input = st.text_input("Digite o caminho para o modelo .pkl que deseja carregar:", value='models/random_forest.pkl')
    
    if st.button("Carregar Modelo"):
        if not model_path_input:
            st.error("Por favor, insira o caminho para o arquivo .pkl do modelo.")
        elif not os.path.exists(model_path_input):
            st.error("O arquivo especificado não existe.")
        else:
            model = load_model(model_path_input)
            if model:
                st.success(f'Modelo carregado com sucesso: {model_path_input}')
                
                st.write("**Faça previsões com o modelo carregado:**")
                user_input = st.text_area("Insira os valores das features separados por vírgula:", "")
                if st.button("Prever"):
                    try:
                        input_values = [float(x.strip()) for x in user_input.split(',')]
                        import numpy as np
                        input_array = np.array(input_values).reshape(1, -1)
                        prediction = model.predict(input_array)
                        st.write(f"**Predição:** {prediction[0]}")
                    except Exception as e:
                        st.error(f"Erro ao fazer a predição: {e}")
    else:
        st.info("Insira o caminho para o modelo e clique em 'Carregar Modelo'.")

if __name__ == '__main__':
    main()
