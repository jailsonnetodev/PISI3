import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.build import build_header
from utils.graph import boxplot, scatter, hist

build_header(
    title='Primeiras Análises',
    hdr='# PRIMEIRAS ANÁLISES E VISUALIZAÇÕES',
    p='<p>Aqui vamos realizar as primeiras observações dos dados e correlações entre algumas variáveis.</p>'
)
