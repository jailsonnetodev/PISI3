import streamlit as st
import shap
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(data_path):
    with open(data_path, 'rb') as file:
        X_training, X_test, y_training, y_test = pickle.load(file)
    return X_training, X_test, y_training, y_test

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def plot_shap_beeswarm(model, X_test, model_type):
    if model_type in ['RandomForest', 'DecisionTree']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    elif model_type == 'KNN':
        explainer = shap.KernelExplainer(model.predict, X_test)
        shap_values = explainer.shap_values(X_test)
    
    st.write(f"Beeswarm Plot para o modelo {model_type}")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="beeswarm", show=False)
    st.pyplot(fig)

st.title('Visualização SHAP para Modelos')


data_path = 'data/usedcars_usa.pkl'
X_training, X_test, y_training, y_test = load_data(data_path)

st.write("Dados de teste carregados com sucesso!")
st.write("Exemplo de X_test:")

model_choice = st.sidebar.selectbox("Escolha o modelo", ["RandomForest", "DecisionTree", "KNN"])

model_paths = {
    'RandomForest': 'models/Random_Forest.pkl',
    'DecisionTree': 'models/Decision_Tree.pkl',
    'KNN': 'models/KNN.pkl'
}

if st.sidebar.button("Carregar Modelo e Gerar SHAP"):
    model_path = model_paths[model_choice]
    model = load_model(model_path)
    plot_shap_beeswarm(model, X_test, model_choice)
else:
    st.write("Por favor, selecione um modelo para visualizar o gráfico SHAP.")
