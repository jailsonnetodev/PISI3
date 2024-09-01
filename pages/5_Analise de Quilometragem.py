import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.build import build_header, top_categories
from utils.graph import boxplot, line_graph

def setup_analysis():
    build_header(
        title='Análise da Quilometragem',
        hdr='# Análise da Quilometragem dos Veículos',
        p='''
            <p>Aqui vamos realizar as primeiras observações dos dados e analisar correlações entre algumas variáveis.</p>
        '''
    )
    st.divider()
