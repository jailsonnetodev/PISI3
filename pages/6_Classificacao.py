import pandas as pd
import numpy as np
import plotly.express as px
import streamlit  as st
import os
import pickle
from utils.transform_pkl import main
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.classifier import ConfusionMatrix


def table_report(y_test: np.ndarray, previsao: np.ndarray, method:str =''):
  #st.markdown(f'##### Classification report do metodo:  <span style="color: blue">{method}</span>', unsafe_allow_html=True)
  report = classification_report(y_test, previsao, output_dict=True)
  classification_data = pd.DataFrame(report).transpose()
  print(classification_data)

def table_report(y_test: np.ndarray, previsao: np.ndarray, method:str =''):
  #st.markdown(f'##### Classification report do metodo:  <span style="color: blue">{method}</span>', unsafe_allow_html=True)
  report = classification_report(y_test, previsao, output_dict=True)
  classification_data = pd.DataFrame(report).transpose()
  #st.table(classification_data)
  print(classification_data)


def confusion_graph(y_test, previsao, method:str = ''):
  #st.markdown(f'##### Matriz de Confução do metodo: <span style="color: blue">{method}</span>', unsafe_allow_html=True)
  labels = sorted(list(set(y_test) | set(previsao)))
  cm = pd.DataFrame(0, index=labels, columns=labels)
  for true_label, predicted_label in zip(y_test, previsao):
      cm.loc[true_label, predicted_label] += 1
  #st.table(cm)
  print(cm)


def tree_decision(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
  )-> None:
  #st.markdown('### Resultado do machine learning usando o método Árvore de decisão')
  if not(os.path.isfile('tree_decision.pkl')):
    obj_tree_decision = DecisionTreeClassifier(criterion='entropy')
    obj_tree_decision.fit(X_training, y_training)
    with open('tree_decision.pkl', mode='wb') as f:
      pickle.dump(obj_tree_decision, f)
  else:
    with open('tree_decision.pkl', 'rb') as f:
      obj_tree_decision = pickle.load(f)
  prevision_tree = obj_tree_decision.predict(x_test)
  table_report(y_test, prevision_tree,'Árvore de decisão')
  confusion_graph(y_test, prevision_tree, 'Árvore de decisão')

