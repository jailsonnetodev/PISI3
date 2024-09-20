import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.build import *
from utils.graph import *

build_header(
    title='Primeiras Análises',
    hdr='# Tempo de Permanencia do Veiculo no Mercado de Usados',
    p='<p>Neste trabalho será possivel identificar as caracteristicas fisicas, funcionais e nao funcionais do veiculo com objetivo indeiticar esse fatores que influencia o tempo de permancencia do veiculo listado no site e disponivel no mercado.</p>'
)

