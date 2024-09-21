import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from itertools import product
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples, silhouette_score

# Carregar os dados (substitua pelo seu próprio caminho de arquivo)
df = pd.read_parquet('data/usedcars_usa.parquet')
#df = df.drop('Unnamed: 0',axis=1)


def processing_data(X):
            
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    few_unique = [col for col in categorical_cols if X[col].nunique() <= 10]
    many_unique = [col for col in categorical_cols if X[col].nunique() > 10]
    if few_unique:
        X = pd.get_dummies(X, columns=few_unique, drop_first=True)
    for col in many_unique:
        freq_encoding = X[col].value_counts() / len(X)
        X[col] = X[col].map(freq_encoding)
    X.fillna(0, inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled

def graph_silhouette(X, optimal_clusters, clusters):

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    sample_silhouette_values = silhouette_samples(X, clusters)
    silhouette_avg = silhouette_score(X, clusters)
    for i in range(optimal_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
  
        color = cm.nipy_spectral(float(i) / optimal_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        y_lower = y_upper + 10 
    ax.set_title("Gráfico de Silhueta para os Clusters K-Means")
    ax.set_xlabel("Valor da Silhueta")
    ax.set_ylabel("Cluster")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([]) 
    ax.set_xticks(np.arange(-0.1, 1.1, 0.2))
    st.pyplot(fig)

columns = ['espaco_banco_traseiro', 'tipo_carroceria', 'cidade', 'consumo_cidade',
           'cilindros_motor', 'cilindradas_motor', 'tipo_motor', 'cor_exterior', 
           'frota', 'chassi_danificado', 'dias_no_mercado', 'concessionaria_franqueada', 
           'marca_da_franquia', 'espaco_banco_dianteiro', 'volume_tanque', 
           'tipo_combustivel', 'historico_acidente', 'altura', 'consumo_estrada', 
           'cavalo_de_potencia', 'cor_interior', 'ee_cabine', 'ee_novo', 
           'comprimento', 'data_listagem', 'cor_listagem', 'nome_marca', 
           'maximo_assentos', 'quilometragem', 'nome_modelo', 'qtd_proprietarios', 
           'potencia', 'preco', 'recuperado', 'valor_economizado', 
           'avaliacao_vendedor', 'nome_vendedor', 'titulo_roubo', 
           'torque', 'transmissao', 'exibicao_transmissao', 'nome_versao', 
           'sistema_rodas', 'exibicao_sistema_rodas', 'entre_eixos', 
           'largura', 'ano', 'dias_no_mercado_label']


st.title('Clusterização de Veículos')
selected_columns = st.multiselect('Selecione as colunas para clusterização:', columns)

if selected_columns:
    if df.empty:
        st.warning('O DataFrame está vazio. Por favor, carregue os dados corretamente.')
    else:
        X = df[selected_columns].copy()
        
        X_scaled = processing_data(X)
        num_clusters = st.slider('Selecione o número de clusters:', min_value=1, max_value=10, value=3)
        
        if num_clusters >= 1:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            clusters = kmeans.fit_predict(X_scaled)
            sse = []
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                sse.append(kmeans.inertia_)
            fig_elbow = go.Figure(data=go.Scatter(x=list(range(1, 11)), y=sse, mode='lines+markers'))
            fig_elbow.update_layout(title='Método do Cotovelo para Determinação do Número de Clusters',
                        xaxis_title='Número de Clusters',
                        yaxis_title='Soma dos Quadrados das Distâncias')
            st.plotly_chart(fig_elbow)
            graph_silhouette(X_scaled,num_clusters,clusters)
            
            
            df['cluster'] = (clusters + 1).astype(int)  
            color_palette = ['yellow', 'blue', 'red', 'green', 'purple', 
                             'orange', 'cyan', 'magenta', 'lime', 'pink']
            color_sequence = color_palette[:num_clusters]
            
            st.subheader('Gráficos de Dispersão dos Clusters')
            ordered_pairs = [(x, y) for x, y in product(selected_columns, selected_columns) if x != y]
  
            max_plots = 20  
            if len(ordered_pairs) > max_plots:
                st.warning(f'Muitos pares de colunas selecionados. Apenas os primeiros {max_plots} pares serão exibidos.')
                ordered_pairs = ordered_pairs[:max_plots]
          
            for i in range(0, len(ordered_pairs), 2): 
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(ordered_pairs):
                        x_col, y_col = ordered_pairs[i + j]
                        fig = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col, 
                            color=df['cluster'].astype(str), 
                            color_discrete_sequence=color_sequence,
                            title=f'Cluster: {x_col} vs {y_col}',
                            labels={'color': 'Cluster'},
                            hover_data=selected_columns 
                        )
                        
                        fig.update_layout(
                            margin=dict(l=120, r=50, t=100, b=150), 
                            legend_title_text='Cluster',
                            title=dict(
                                text=f'Cluster: {x_col} vs {y_col}',
                                x=0,  
                                xanchor='left'
                            ),
                            xaxis_title_font=dict(size=14),
                            yaxis_title_font=dict(size=14),
                            legend_font=dict(size=12),
                            plot_bgcolor='rgba(0,0,0,0)' 
                        )
                        fig.update_xaxes(tickfont=dict(size=12), tickangle=-45)
                        fig.update_yaxes(tickfont=dict(size=12))
                        
                        cols[j].plotly_chart(fig, use_container_width=True)
        else:
            st.warning('O número de clusters deve ser pelo menos 1.')
else:
    st.warning('Selecione ao menos uma coluna para realizar a clusterização.')
