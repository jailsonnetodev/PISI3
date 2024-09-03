import streamlit as st
import pandas as pd
import numpy as np
from utils.build import build_header
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.graph import boxplot,scatter,treemap,hist,bar,select_chart,line_graph
from utils.build import  top_categories


build_header(
    title='Analise Da Quilometragem',
    hdr='# ANALISE DA QUILOMETRAGEM DOS VEICULOS',
    p='''
        <p> Aqui vamos realizar as primeiras observações dos dados e correlações entre algumas variaveis</p>
    '''
)
