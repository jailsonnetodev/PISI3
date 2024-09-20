import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import pickle
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from utils.transform_regressor_pkl import main

# Função para avaliar os modelos
def evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Avalia múltiplos modelos de regressão e retorna uma tabela com as métricas de desempenho.

    :param models: Dicionário de modelos para avaliar { 'nome_modelo': modelo }
    :param X_train: Conjunto de treino (features)
    :param y_train: Conjunto de treino (target)
    :param X_test: Conjunto de teste (features)
    :param y_test: Conjunto de teste (target)
    :return: DataFrame com as métricas de avaliação (MSE, RMSE, MAE, R²)
    """
    results = []
