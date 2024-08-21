import pandas as pd
import plotly.express as px
import streamlit as st
import pickle
import numpy as np

import os
from utils.build import build_header
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.transform_pkl import main
from yellowbrick.classifier import ConfusionMatrix
import streamlit_yellowbrick as sty


def table_report(y_test: np.ndarray, previsao: np.ndarray, method:str =''):
  st.markdown(f'##### Classification report do metodo:  <span style="color: blue">{method}</span>', unsafe_allow_html=True)
  report = classification_report(y_test, previsao, output_dict=True)
  classification_data = pd.DataFrame(report).transpose()
  st.table(classification_data)



def confusion_graph(y_test, previsao, method:str = ''):
  st.markdown(f'##### Matriz de Confução do metodo: <span style="color: blue">{method}</span>', unsafe_allow_html=True)
  labels = sorted(list(set(y_test) | set(previsao)))
  cm = pd.DataFrame(0, index=labels, columns=labels)
  for true_label, predicted_label in zip(y_test, previsao):
      cm.loc[true_label, predicted_label] += 1
  st.table(cm)


# def matrix(x_training, x_test, y_training, y_test, mtd):
#   cm = ConfusionMatrix(mtd)
#   cm.fit(x_training, y_training)
#   cm_score = cm.score(x_test, y_test)
#   sty.st_yellowbrick(cm_score)
  

def naive_bayes(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, obj_naive= None
  )-> None:
  st.markdown('### Resultado do machine learning usando o método Naive Bayes')
  if not(os.path.isfile('data/naive_bayes.pkl')):
    obj_naive = GaussianNB()
    obj_naive.fit(X_training, y_training)
    with open('data/naive_bayes.pkl', mode='wb') as f:
      pickle.dump(obj_naive, f)
  else:
    with open('data/naive_bayes.pkl', 'rb') as f:
      obj_naive = pickle.load(f)
  previsor = obj_naive.predict(X_test)
  table_report(y_test, previsor, 'Naive Bayes')
  confusion_graph(y_test, previsor, 'Naive Bayes')
  #matrix(X_training, X_test, y_training, y_test, mtd=obj_naive)


def tree_decision(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
  )-> None:
  st.markdown('### Resultado do machine learning usando o método Árvore de decisão')
  if not(os.path.isfile('data/tree_decision.pkl')):
    obj_tree_decision = DecisionTreeClassifier(criterion='entropy')
    obj_tree_decision.fit(X_training, y_training)
    with open('data/tree_decision.pkl', mode='wb') as f:
      pickle.dump(obj_tree_decision, f)
  else:
    with open('data/tree_decision.pkl', 'rb') as f:
      obj_tree_decision = pickle.load(f)
  prevision_tree = obj_tree_decision.predict(x_test)
  table_report(y_test, prevision_tree,'Árvore de decisão')
  confusion_graph(y_test, prevision_tree, 'Árvore de decisão')

  
def random_forest(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
  )-> None:
  st.markdown('### Resultado do machine learning usando o método Random Forest')

  if not(os.path.isfile('data/random_forest.pkl')):
    obj_random_forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    obj_random_forest.fit(x_training, y_training)
    with open('data/random_forest.pkl', mode='wb') as f:
      pickle.dump(obj_random_forest, f)
  else:
    with open('data/random_forest.pkl', 'rb') as f:
      obj_random_forest = pickle.load(f)

  prevision_random_forest = obj_random_forest.predict(x_test)
  importances = pd.Series(
    data=obj_random_forest.feature_importances_,
    index=['ano', 'cidade', 'estado', 'marca', 'modelo']
  )
  important = importances.to_frame()
  important.reset_index(inplace=True)
  important.columns = ['Importância','Feature', ]
  st.markdown('##### Gráfico de Importância de parametros')
  st.plotly_chart(px.bar(data_frame=important, x='Feature', y='Importância', orientation='h', template='plotly_dark'))
  table_report(y_test, prevision_random_forest, 'Random Forest')
  confusion_graph(y_test, prevision_random_forest, 'Random Forest')
  #matrix(X_training, X_test, y_training, y_test, mtd=obj_random_forest)



def KNN(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
  )-> None:
  st.markdown('### Resultado do machine learning usando o método KNN')
  if not(os.path.isfile('data/KNN_data.pkl')):
    obj_knn = KNeighborsClassifier(n_neighbors=10, weights='distance', p=1)
    obj_knn.fit(x_training, y_training)
    with open('data/KNN_data.pkl', mode='wb') as f:
      pickle.dump(obj_knn, f)
  else:
    with open('data/KNN_data.pkl', 'rb') as f:
      obj_knn = pickle.load(f)
  prevision_knn = obj_knn.predict(x_test)
  table_report(y_test, prevision_knn, 'KNN')
  confusion_graph(y_test, prevision_knn, 'KNN')






if not(os.path.isfile('data/price_cars.pkl')):
  print('iniciando...')
  main()

with open('data/price_cars.pkl', 'rb') as f:
    X_training, X_test, y_training, y_test = pickle.load(f)

    
naive_bayes(X_training, y_training, X_test, y_test)
tree_decision(X_training, y_training, X_test, y_test)
random_forest(X_training, y_training, X_test, y_test)
KNN(X_training, y_training, X_test, y_test)