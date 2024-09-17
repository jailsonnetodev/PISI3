import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score


columns_drop =  ['nome_modelo','nome_marca','chassi_danificado','data_listagem','nome_vendedor','cidade']
features = df.drop(columns_drop, axis=1)

def label_encoder(x_data):
  le=LabelEncoder()
  for col in x_data:
    if x_data[col].dtypes == 'object' or x_data[col].dtypes == 'category':
      x_data[col] = pd.DataFrame(le.fit_transform(x_data[col]))
  x_data = x_data.values
  return x_data

X_labels = label_encoder(features)


def standard(x_data):
  scaler = StandardScaler()
  x_data = scaler.fit_transform(x_data)

  return x_data
X_standard = standard(X_labels)



# 2. Determinação do Número de Clusters usando o Método do Cotovelo
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_standard)
    sse.append(kmeans.inertia_)

# Gráfico do Método do Cotovelo
fig_elbow = go.Figure(data=go.Scatter(x=list(range(1, 11)), y=sse, mode='lines+markers'))
fig_elbow.update_layout(title='Método do Cotovelo para Determinação do Número de Clusters',
                        xaxis_title='Número de Clusters',
                        yaxis_title='Soma dos Quadrados das Distâncias')
fig_elbow.show()
