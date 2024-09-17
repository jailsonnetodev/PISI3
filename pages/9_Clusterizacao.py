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

# 2. Clusterização com K-Means
optimal_clusters = 4  # Defina o número de clusters com base no método do cotovelo
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_standard)


# 3. Reduzindo a Dimensionalidade com PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_standard)


# Adicionando as componentes principais ao DataFrame
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['cluster'] = cluster_labels  # Adicionando os rótulos dos cluster


# 4. Visualização dos Clusters com scatter_matrix
fig = px.scatter_matrix(df_pca, dimensions=['PC1', 'PC2', 'PC3'], color='cluster',
                        title='Matriz de Dispersão dos Clusters usando PCA com K-Means')
fig.update_traces(diagonal_visible=False)  # Oculta os histogramas da diagonal
fig.show()

# 3. Cálculo da Pontuação de Silhueta
silhouette_avg = silhouette_score(X_standard, cluster_labels)
print(f'Coeficiente de Silhueta Médio para K-Means: {silhouette_avg:.2f}')


# Cálculo da pontuação de silhueta para cada ponto
sample_silhouette_values = silhouette_samples(X_standard, cluster_labels)

# 4. Gráfico de Silhueta
fig, ax = plt.subplots(figsize=(8, 6))

y_lower = 10
for i in range(optimal_clusters):
    # Agregar as pontuações de silhueta para as amostras que pertencem ao cluster i
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / optimal_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.7)

    # Rotular o gráfico com o número do cluster no meio
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    y_lower = y_upper + 10  # 10 para espaço entre os clusters

ax.set_title("Gráfico de Silhueta para os Clusters K-Means")
ax.set_xlabel("Valor da Silhueta")
ax.set_ylabel("Cluster")

# Linha vertical para a média da pontuação de silhueta de todos os valores
ax.axvline(x=silhouette_avg, color="red", linestyle="--")

ax.set_yticks([])  # Ocultar as etiquetas do eixo y
ax.set_xticks(np.arange(-0.1, 1.1, 0.2))

plt.show()
