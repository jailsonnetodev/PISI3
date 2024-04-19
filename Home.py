import streamlit as st
import pandas as pd


st.set_page_config(
        page_title='Home',
        initial_sidebar_state='expanded',
    )
st.header('Predição de Preço e Análise de Dados de Veículos Usados com Uso de Aprendizado de Máquina')
st.write('''<p> Neste trabalho será desenvolvido um modelo de predição de preços de carros usados, com base no uso de técnicas de aprendizado de máquina, isto permitirá a formação de preços justos e competitivos. </p>
    ''', unsafe_allow_html=True)

if "data" not in st.session_state:
    data = pd.read_csv('data/price_cars.csv')
    st.session_state["data"]=data


st.markdown(f'''
    O objetivo deste projeto é identificar e analisar as correlações entre as variáveis para prever o  preço do veículo usado com aplicação de diversos algoritmos de aprendizado de máquina.

            
    <br>
    Passo a Passo desse projeto:
    <ul>
            <li>Exploração e visualização dos dados</li>
            <li>Analise Exploratoria</li>
            <li>Tratamento dos dados para uso dos algoritimos ML</li>
            <li>Aplicação de Marchine Learning</li>
            <li>Metricas</li>
            <li>Comparando modelos</li>
    </ul>
    Dataset: <a href="https://www.kaggle.com/datasets/harikrishnareddyb/used-car-price-predictions?resource=download"</a><br>
    Contribuidores:<br>Silas Ribeiro<br>Ronald Barbosa <br>Joao Gabriel Nascimento<br>Jailson Neto <br> Yrlar Kleber
''', unsafe_allow_html=True)



