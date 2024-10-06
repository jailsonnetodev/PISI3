import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

with open('data/usedcars_usa.pkl', 'rb') as f:
    X_training, X_test, y_training, y_test = pickle.load(f)
    

def calcular_metricas(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average=None)
    train_recall = recall_score(y_train, y_train_pred, average=None)
    train_f1 = f1_score(y_train, y_train_pred, average=None)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average=None)
    test_recall = recall_score(y_test, y_test_pred, average=None)
    test_f1 = f1_score(y_test, y_test_pred, average=None)

    return train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1


def gerar_tabela_html(nome_modelo, train_acc, train_prec, train_rec, train_f1, test_acc, test_prec, test_rec, test_f1):
    html_table = f"""
    <table style="width:80%">
      <tr>
        <th rowspan="6">{nome_modelo}</th>
        <th colspan="7">Treino</th>
        <th colspan="7">Teste</th>
      </tr>
        <tr>
        <th colspan="2"></th>
        <th>ate-36 dias</th>
        <th>36-83 dias</th>
        <th>83-185 dias</th>
        <th>185-365 dias</th>
        <th>> 365 dias</th>
        <th>ate-36 dias</th>
        <th>36-83 dias</th>
        <th>83-185 dias</th>
        <th>185-365 dias</th>
        <th>> 365 dias</th>
      </tr>
      <tr>
        <th>accuracy</th>
        <td>{train_acc:.2f}</td>
        <td>{train_acc:.2f}</td>
        <td>{train_acc:.2f}</td>
        <td>{train_acc:.2f}</td>
        <td>{train_acc:.2f}</td>
        <td>{test_acc:.2f}</td>
        <td>{test_acc:.2f}</td>
        <td>{test_acc:.2f}</td>
        <td>{test_acc:.2f}</td>
        <td>{test_acc:.2f}</td>
      </tr>
       <tr>
        <th>precision</th>
        <td>{train_prec[0]:.2f}</td>
        <td>{train_prec[1]:.2f}</td>
        <td>{train_prec[2]:.2f}</td>
        <td>{train_prec[3]:.2f}</td>
        <td>{train_prec[4]:.2f}</td>
        <td>{test_prec[0]:.2f}</td>
        <td>{test_prec[1]:.2f}</td>
        <td>{test_prec[2]:.2f}</td>
        <td>{test_prec[3]:.2f}</td>
        <td>{test_prec[4]:.2f}</td>
      </tr>
        <tr>
        <th>recall</th>
        <td>{train_rec[0]:.2f}</td>
        <td>{train_rec[1]:.2f}</td>
        <td>{train_rec[2]:.2f}</td>
        <td>{train_rec[3]:.2f}</td>
        <td>{train_rec[4]:.2f}</td>
        <td>{test_rec[0]:.2f}</td>
        <td>{test_rec[1]:.2f}</td>
        <td>{test_rec[2]:.2f}</td>
        <td>{test_rec[3]:.2f}</td>
        <td>{test_rec[4]:.2f}</td>
      </tr>
        <tr>
        <th>f1</th>
        <td>{train_f1[0]:.2f}</td>
        <td>{train_f1[1]:.2f}</td>
        <td>{train_f1[2]:.2f}</td>
        <td>{train_f1[3]:.2f}</td>
        <td>{train_f1[4]:.2f}</td>
        <td>{test_f1[0]:.2f}</td>
        <td>{test_f1[1]:.2f}</td>
        <td>{test_f1[2]:.2f}</td>
        <td>{test_f1[3]:.2f}</td>
        <td>{test_f1[4]:.2f}</td>
      </tr>
    </table>
    """
    return html_table

modelos = ['Random_Forest.pkl', 'Decision_Tree.pkl', 'KNN.pkl']
nomes_modelos = ['Random Forest', 'Decision Tree', 'KNN']


import os
caminho_modelos = 'models/'

tabelas_html = []
for modelo_pkl, nome_modelo in zip(modelos, nomes_modelos):
    caminho_completo = os.path.join(caminho_modelos, modelo_pkl) 
    with open(caminho_completo, 'rb') as file:
        model = pickle.load(file)

    train_acc, train_prec, train_rec, train_f1, test_acc, test_prec, test_rec, test_f1 = calcular_metricas(model, X_training, y_training, X_test, y_test)

    tabela_html = gerar_tabela_html(nome_modelo, train_acc, train_prec, train_rec, train_f1, test_acc, test_prec, test_rec, test_f1)
    tabelas_html.append(tabela_html)


st.title("Comparação de Modelos")
for tabela in tabelas_html:
    st.markdown(tabela, unsafe_allow_html=True)