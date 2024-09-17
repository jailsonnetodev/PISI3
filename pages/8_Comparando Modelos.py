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



def calculate_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics_train = {
        'Acurácia': accuracy_score(y_train, y_train_pred),
        'Precisão': precision_score(y_train, y_train_pred, average='weighted'),
        'Recall': recall_score(y_train, y_train_pred, average='weighted'),
        'F1 Score': f1_score(y_train, y_train_pred, average='weighted')
    }

    metrics_test = {
        'Acurácia': accuracy_score(y_test, y_test_pred),
        'Precisão': precision_score(y_test, y_test_pred, average='weighted'),
        'Recall': recall_score(y_test, y_test_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_test_pred, average='weighted')
    }

    return metrics_train, metrics_test
