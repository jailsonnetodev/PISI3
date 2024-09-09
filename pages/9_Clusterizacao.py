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

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Ajuste eps e min_samples conforme necessário
clusters_dbscan = dbscan.fit_predict(df_scaled)

# Adicionar os clusters ao dataset original
df['cluster_dbscan'] = clusters_dbscan

# Contar o número de clusters gerados (-1 é o ruído)
num_clusters = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
print(f"Número de clusters encontrados: {num_clusters}")

# Avaliar a qualidade dos clusters (somente se houver mais de um cluster)
if num_clusters > 1:
    silhouette_avg_dbscan = silhouette_score(df_scaled, clusters_dbscan)
    print(f"Índice de Silhueta para DBSCAN: {silhouette_avg_dbscan}")

# Visualização dos clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['preco'], df['quilometragem'], c=df['cluster_dbscan'], cmap='viridis', marker='o', s=50)
plt.xlabel('Preço')
plt.ylabel('Quilometragem')
plt.title('Clusters usando DBSCAN')
plt.show()
