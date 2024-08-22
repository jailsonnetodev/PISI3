import os
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.build import build_header
from utils.transform_pkl import main

def table_report(y_test: np.ndarray, predictions: np.ndarray, method: str = '') -> None:
    st.markdown(f'##### Classification report do método: <span style="color: blue">{method}</span>', unsafe_allow_html=True)
    report = classification_report(y_test, predictions, output_dict=True)
    classification_data = pd.DataFrame(report).transpose()
    st.table(classification_data)

def confusion_matrix_display(y_test: np.ndarray, predictions: np.ndarray, method: str = '') -> None:
    st.markdown(f'##### Matriz de Confusão do método: <span style="color: blue">{method}</span>', unsafe_allow_html=True)
    labels = sorted(list(set(y_test) | set(predictions)))
    cm = pd.DataFrame(0, index=labels, columns=labels)
    for true_label, predicted_label in zip(y_test, predictions):
        cm.loc[true_label, predicted_label] += 1
    st.table(cm)

def naive_bayes(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
    st.markdown('### Resultado do machine learning usando o método Naive Bayes')
    model_path = 'data/naive_bayes.pkl'
    
    if not os.path.isfile(model_path):
        model = GaussianNB()
        model.fit(x_train, y_train)
        with open(model_path, mode='wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    predictions = model.predict(x_test)
    table_report(y_test, predictions, 'Naive Bayes')
    confusion_matrix_display(y_test, predictions, 'Naive Bayes')

def decision_tree(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
    st.markdown('### Resultado do machine learning usando o método Árvore de Decisão')
    model_path = 'data/tree_decision.pkl'
    
    if not os.path.isfile(model_path):
        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(x_train, y_train)
        with open(model_path, mode='wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    predictions = model.predict(x_test)
    table_report(y_test, predictions, 'Árvore de Decisão')
    confusion_matrix_display(y_test, predictions, 'Árvore de Decisão')

def random_forest(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
    st.markdown('### Resultado do machine learning usando o método Random Forest')
    model_path = 'data/random_forest.pkl'
    
    if not os.path.isfile(model_path):
        model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        model.fit(x_train, y_train)
        with open(model_path, mode='wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    predictions = model.predict(x_test)
    importances = pd.Series(data=model.feature_importances_, index=['ano', 'cidade', 'estado', 'marca', 'modelo'])
    importance_df = importances.reset_index()
    importance_df.columns = ['Feature', 'Importância']
    
    st.markdown('##### Gráfico de Importância dos Parâmetros')
    st.plotly_chart(px.bar(data_frame=importance_df, x='Feature', y='Importância', orientation='h', template='plotly_dark'))
    table_report(y_test, predictions, 'Random Forest')
    confusion_matrix_display(y_test, predictions, 'Random Forest')

def knn(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
    st.markdown('### Resultado do machine learning usando o método KNN')
    model_path = 'data/KNN_data.pkl'
    
    if not os.path.isfile(model_path):
        model = KNeighborsClassifier(n_neighbors=10, weights='distance', p=1)
        model.fit(x_train, y_train)
        with open(model_path, mode='wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    predictions = model.predict(x_test)
    table_report(y_test, predictions, 'KNN')
    confusion_matrix_display(y_test, predictions, 'KNN')

if not os.path.isfile('data/price_cars.pkl'):
    print('Iniciando...')
    main()

with open('data/price_cars.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

naive_bayes(X_train, y_train, X_test, y_test)
decision_tree(X_train, y_train, X_test, y_test)
random_forest(X_train, y_train, X_test, y_test)
knn(X_train, y_train, X_test, y_test)
