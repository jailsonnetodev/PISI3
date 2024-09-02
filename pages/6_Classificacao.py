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
def random_forest(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
  )-> None:
  st.markdown('### Resultado do machine learning usando o método Random Forest')

  if not(os.path.isfile('random_forest.pkl')):
    obj_random_forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    obj_random_forest.fit(x_training, y_training)
    with open('random_forest.pkl', mode='wb') as f:
      pickle.dump(obj_random_forest, f)
  else:
    with open('random_forest.pkl', 'rb') as f:
      obj_random_forest = pickle.load(f)

  prevision_random_forest = obj_random_forest.predict(x_test)
  importances = pd.Series(
    data=obj_random_forest.feature_importances_,
    index= ['espaco_banco_traseiro', 'tipo_carroceria', 'cidade', 'consumo_cidade',
       'cilindros_motor', 'cilindradas_motor', 'tipo_motor',
       'cor_exterior', 'frota', 'chassi_danificado',
       'concessionaria_franqueada', 'marca_da_franquia',
       'espaco_banco_dianteiro', 'volume_tanque', 'tipo_combustivel',
       'historico_acidente', 'altura', 'consumo_estrada', 'cavalo_de_potencia',
       'cor_interior', 'ee_cabine', 'ee_novo', 'comprimento', 'data_listagem',
       'cor_listagem', 'nome_marca', 'maximo_assentos', 'quilometragem',
       'nome_modelo', 'qtd_proprietarios', 'potencia', 'preco', 'recuperado',
       'valor_economizado', 'avaliacao_vendedor', 'nome_vendedor',
       'titulo_roubo', 'torque', 'transmissao', 'exibicao_transmissao',
       'nome_versao', 'sistema_rodas', 'exibicao_sistema_rodas', 'entre_eixos',
       'largura', 'ano']
  )
  important = importances.to_frame()
  important.reset_index(inplace=True)
  important.columns = ['Importância','Feature', ]
  st.markdown('##### Gráfico de Importância de parametros')
  px.bar(data_frame=important, x='Feature', y='Importância', orientation='h', template='plotly_dark')
  table_report(y_test, prevision_random_forest, 'Random Forest')
  confusion_graph(y_test, prevision_random_forest, 'Random Forest')


def KNN(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
  )-> None:
  st.markdown('### Resultado do machine learning usando o método KNN')
  if not(os.path.isfile('KNN_data.pkl')):
    obj_knn = KNeighborsClassifier(n_neighbors=10, weights='distance', p=1)
    obj_knn.fit(x_training, y_training)
    with open('KNN_data.pkl', mode='wb') as f:
      pickle.dump(obj_knn, f)
  else:
    with open('KNN_data.pkl', 'rb') as f:
      obj_knn = pickle.load(f)
  prevision_knn = obj_knn.predict(x_test)
  table_report(y_test, prevision_knn, 'KNN')
  confusion_graph(y_test, prevision_knn, 'KNN')


if not(os.path.isfile('data/usedcars_usa.pkl')):
  print('iniciando...')
  main()

with open('data/usedcars_usa.pkl', 'rb') as f:
    X_training, X_test, y_training, y_test = pickle.load(f)


tree_decision(X_training, y_training, X_test, y_test)
random_forest(X_training, y_training, X_test, y_test)
KNN(X_training, y_training, X_test, y_test)

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

