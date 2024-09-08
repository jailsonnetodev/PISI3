import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


X_clusters = df.iloc[:,[4,28,32]].values

scaler = StandardScaler()
X_clusters = scaler.fit_transform(X_clusters)



wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_clusters)
    wcss.append(kmeans.inertia_)



graph_wcss = px.line(x = range(1,11), y=wcss)
graph_wcss



kmeans_usedcars = KMeans(n_clusters=4, random_state=0)
labels = kmeans_usedcars.fit_predict(X_clusters)


pca = PCA(n_components=2)
X_clusters_pca = pca.fit_transform(X_clusters)


graph_clusters = px.scatter(x=X_clusters_pca[:,0], y= X_clusters_pca[:,1], color=labels)
graph_clusters.show()

colunas_cluster = ['preco', 'quilometragem', 'cavalo_de_potencia', 'consumo_cidade', 'dias_no_mercado']
df_cluster = df[colunas_cluster]
