import streamlit as st
import pandas as pd


st.set_page_config(
        page_title='Home',
        initial_sidebar_state='expanded',
    )
st.header('Predição de Preço e Análise de Dados de Veículos Usados com Uso de Aprendizado de Máquina')

st.write('''<p> Neste trabalho será desenvolvido um modelo de predição de preços de carros usados, com base no uso de técnicas de aprendizado de máquina, isto permitirá a formação de preços justos e competitivos. </p>
    ''', unsafe_allow_html=True)


