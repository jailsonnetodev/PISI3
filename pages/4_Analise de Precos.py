import streamlit as st
import pandas as pd
import numpy as np
from utils.build import build_header
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.graph import *
from utils.build import  top_categories



build_header(
    title='Analise Dos Precos',
    hdr='# ANALISE DOS PRECOS E SUAS CORRELACOES',
    p='''
        <p> Aqui vamos realizar as primeiras observações dos dados e correlações entre algumas variaveis</p>
    '''
)