import pandas as pd
import streamlit as st
import plotly.express as px

## Reservados para codigos temp e outros nao executaveis##



#util.py

def load_dataset(path: str):

    data = pd.read_csv(path)
    return data

df = load_dataset('data/price_cars.csv')
df_copy = df.copy()
df_copy = df_copy.to_parquet('data/price_cars_copy.parquet')
#duplicatas

#create new column with age of vehicle.
df['idade_carro']=df['Year'].max()-df['Year']


# CPIs years 
cpi_data = {

    2014: 236.7,
    2013: 232.9,
    2016: 240.0,
    2012: 229.6,
    2009: 214.5,
    2015: 237.0,
    2010: 218.1,
    2011: 224.9,
    2007: 207.3,
    2006: 201.6,
    2008: 215.3,
    2004: 188.9,
    2017: 245.1,
    2005: 195.3,
    2003: 184.0,
    2002: 179.9,
    1999: 166.6,
    2001: 177.1,
    2000: 172.2,
    1998: 163.0,
    2018: 251.1,
    1997: 160.5,
    2024: 310.0
}


# Função para ajustar o preço de acordo com a inflação
def adjust_price(row):
    original_price = row['Price']
    original_year = row['Year']
    cpi_original = cpi_data[original_year]
    cpi_current = cpi_data[2024]  # Usando CPI de 2024 como referência
    adjusted_price = original_price * (cpi_current / cpi_original)
    return adjusted_price

# Aplicando o ajuste de preço com uma função lambda
df['preco_ajustado'] = df.apply(lambda row: adjust_price(row), axis=1)
df['preco_ajustado'] = df['preco_ajustado'].apply(lambda x: round(x, 2))


#functions for depreciate price 
def depreciate_price(price, years, depreciation_rate):
    """  informe os dados para depreciar o valor    """
    return price * (1 - depreciation_rate) ** years

df['preco_depreciado'] = df.apply(lambda row: depreciate_price(row['preco_ajustado'], row['idade_carro'], 0.24), axis=1)
df['preco_depreciado'] = df['preco_depreciado'].apply(lambda x: round(x, 2))

def process_data(data):
  data.duplicated().sum()
  data.drop_duplicates()
  columns_drop = ['Vin']
  data = data.drop(columns= columns_drop, axis=1)
  data.rename(columns={'Price': 'preco','Year': 'ano','Mileage': 'quilometragem','City': 'cidade', 'State': 'estado', 'Make': 'marca', 'Model': 'modelo'}, inplace = True)
  data['modelo'] = data['modelo'].replace(['1','2','3','4','5','6','7','8'], ['Serie 1','Serie 2','Serie 3','Serie 4','Serie 5','Serie 6','Serie 7','Serie 8'])
  return data

df = process_data(df)




list_outliers = ['quilometragem']
def drop_outliers(df, columns, k=1.5):
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        df[column] = df[column].clip(lower=q1 - k * iqr, upper=q3 + k * iqr)
    return df
drop_outliers(df,list_outliers,k=1.5)



classes = [0, 15000, 35000, 499500]
labels = ['0 a 15k', '15k 35k', '+35k']
intervals = pd.cut(x=df.preco, bins=classes, labels=labels)
df['preco_intervalo'] = intervals





def transform_parquet(path, engine='auto'):
  path = path
  try:
    new_data = df.to_parquet(path)
    print('successful')
  except:
    print('error')
  return new_data

transform_parquet('data/price_cars.parquet')



def random_parquet(path: str, num: int) ->None:
  data = pd.read_parquet(path)
  new_data = data.sample(num, replace=False)
  num2 = ''.join(reversed(''.join(reversed(f'{num}')).replace('000','k')))
  new_data.to_parquet(path.replace('.',f'{num2}.'))


for i in [10000,100000,500000]:
  random_parquet('data/price_cars.parquet',i)

px.box(df,x='preco_depreciado')
px.box(df,x='Mileage')

t1=df.groupby(by='Year')['Price'].mean()
t1=pd.DataFrame(t1)
t1=t1.reset_index()
arr=np.array(t1['Year'])
plt.figure(figsize=(20,5))
a=sns.lineplot(data=t1,x='Year',y='Price')
plt.xticks(arr)
plt.ylabel('Average price by Year')
a


plt.figure(figsize=(20,5))
b=sns.countplot(data=df,x='Year', color='skyblue')
b.bar_label(b.containers[0])
plt.ylabel('Count of Car')
b


veiculos_anos = df['Year'].value_counts().sort_values().reset_index(name='Total')
px.bar(veiculos_anos, x='Year',y='Total')



import matplotlib.pyplot as plt
import seaborn as sns

# Configurando o estilo do Seaborn
sns.set(style="whitegrid")

# Distribuição dos preços
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=50, kde=True, color='blue')
plt.title('Distribuição dos Preços dos Veículos')
plt.xlabel('Preço ($)')
plt.ylabel('Frequência')
plt.xlim(0, 100000)
plt.show()




# Distribuição da quilometragem
plt.figure(figsize=(10, 6))
sns.histplot(df['Mileage'], bins=50, kde=True, color='green')
plt.title('Distribuição da Quilometragem dos Veículos')
plt.xlabel('Quilometragem (milhas)')
plt.ylabel('Frequência')
plt.xlim(0, 300000)
plt.show()





# Relação entre o Ano do Veículo e o Preço
plt.figure(figsize=(10, 6))
sns.boxplot(x='Year', y='Price', data=df)
plt.title('Relação entre o Ano do Veículo e o Preço')
plt.xlabel('Ano')
plt.ylabel('Preço ($)')
plt.xticks(rotation=45)
plt.ylim(0, 100000)
plt.show()



# Relação entre Quilometragem e Preço
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Mileage', y='Price', data=df, alpha=0.5)
plt.title('Relação entre Quilometragem e Preço')
plt.xlabel('Quilometragem (milhas)')
plt.ylabel('Preço ($)')
plt.xlim(0, 300000)
plt.ylim(0, 100000)
plt.show()



#df.loc[df['Price'] > 47987]
data_maior = df.loc[df['Price'] > 50000]
data_maior
# temos 28.000 registros acima do limite superior


# Criando faixas de quilometragem
df['MileageRange'] = pd.cut(df['Mileage'], bins=[0, 50000, 100000, 150000, 200000, 300000], 
                            labels=['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k+'])

# Preço médio por faixa de quilometragem
average_price_by_mileage = df.groupby('MileageRange')['Price'].mean().reset_index()
average_price_by_mileage.columns = ['MileageRange', 'AveragePrice']
px.line(average_price_by_mileage,x='MileageRange',y='AveragePrice')



def top_categories(data, top: int, label: str):
    '''DataFrame, Top: int, Label: str'''
    top_make_data = data[label].value_counts().nlargest(top).index
    filtered_data = data[data[label].isin(top_make_data)]
    return filtered_data

filtrado = top_categories(
    data=df,
    top=10,
    label='City'
)
px.bar(filtrado,x='City')





def top_cities_models_makes(df, top_cities_n=10, top_models_n=5):
    # Contar o número de veículos por cidade
    top_cities = df['City'].value_counts().head(top_cities_n).index

    # Filtrar o dataset para incluir apenas as top 10 cidades
    df_top_cities = df[df['City'].isin(top_cities)]

    # Criar uma nova coluna que combina Make e Model
    df_top_cities['MakeModel'] = df_top_cities['Make'] + ' ' + df_top_cities['Model']

    # Contar o número de veículos por MakeModel dentro dessas cidades
    top_make_models = df_top_cities['MakeModel'].value_counts().head(top_models_n).index

    # Filtrar o dataset para incluir apenas os top 5 make-models
    df_top_make_models = df_top_cities[df_top_cities['MakeModel'].isin(top_make_models)]

    return df_top_make_models

# Usando a função com o DataFrame carregado
df_filtrado = top_cities_models_makes(df, top_cities_n=20, top_models_n=10)
print(df_filtrado)




#CODIGO DA PAGINA DE CLASSIFICACAO 20/09


def table_report(y_test: np.ndarray, previsao: np.ndarray, method:str =''):
  st.markdown(f'##### Classification report do metodo:  <span style="color: blue">{method}</span>', unsafe_allow_html=True)
  report = classification_report(y_test, previsao, output_dict=True)
  classification_data = pd.DataFrame(report).transpose()
  st.table(classification_data)


  
def confusion_graph(y_test, previsao, method:str = ''):
  st.markdown(f'##### Matriz de Confução do metodo: <span style="color: blue">{method}</span>', unsafe_allow_html=True)
  labels = sorted(list(set(y_test) | set(previsao)))
  cm = pd.DataFrame(0, index=labels, columns=labels)
  for true_label, predicted_label in zip(y_test, previsao):
      cm.loc[true_label, predicted_label] += 1
  st.table(cm)

  
def random_forest(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
  )-> None:
  st.markdown('### Resultado do machine learning usando o método Random Forest')
  st.markdown('##### Execute o treinamento do modelo abaixo:')

  if not(os.path.isfile('data/random_forest.pkl')):
    obj_random_forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    obj_random_forest.fit(x_training, y_training)
    with open('data/random_forest.pkl', mode='wb') as f:
      pickle.dump(obj_random_forest, f)
  else:
    with open('data/random_forest.pkl', 'rb') as f:
      obj_random_forest = pickle.load(f)

  prevision_random_forest = obj_random_forest.predict(x_test)
  importances = pd.Series(
    data=obj_random_forest.feature_importances_,
    index= ['espaco_banco_traseiro', 'tipo_carroceria', 'consumo_cidade',
       'cilindros_motor', 'cilindradas_motor', 'tipo_motor',
       'cor_exterior', 'frota', 'chassi_danificado',
       'concessionaria_franqueada', 'marca_da_franquia',
       'espaco_banco_dianteiro', 'volume_tanque', 'tipo_combustivel',
       'historico_acidente', 'altura', 'consumo_estrada', 'cavalo_de_potencia',
       'cor_interior', 'ee_cabine', 'ee_novo', 'comprimento',
       'cor_listagem', 'nome_marca', 'maximo_assentos', 'quilometragem',
       'nome_modelo', 'qtd_proprietarios', 'potencia', 'preco', 'recuperado',
       'valor_economizado', 'avaliacao_vendedor', 'nome_vendedor',
       'titulo_roubo', 'torque', 'transmissao', 'exibicao_transmissao',
       'nome_versao', 'sistema_rodas', 'exibicao_sistema_rodas', 'entre_eixos',
       'largura', 'ano']
  )
  important = importances.to_frame()
  important.reset_index(inplace=True)
  important.columns = ['Importância','Feature', ]
  st.markdown('##### Gráfico de Importância de parametros')
  fig = px.bar(data_frame=important, x='Feature', y='Importância', orientation='h', template='plotly_dark')
  fig.update_layout(yaxis={'categoryorder':'total ascending'})
  table_report(y_test, prevision_random_forest, 'Random Forest')
  confusion_graph(y_test, prevision_random_forest, 'Random Forest')
  st.plotly_chart(fig)


def calculate_mean_shap_importance(shap_values, feature_names):
    mean_shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap_importance
    }).sort_values(by='importance', ascending=False)
    return importance_df


# Função para plotar a importância usando Plotly
def plot_feature_importance_plotly(importance_df, model_name):
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                 title=f'Feature Importance - {model_name}', height=600)
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig)

def KNN(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
  )-> None:
  #st.markdown('### Resultado do machine learning usando o método KNN')
  if not(os.path.isfile('data/KNN_data.pkl')):
    obj_knn = KNeighborsClassifier(n_neighbors=10, weights='distance', p=1)
    obj_knn.fit(x_training, y_training)
    with open('data/KNN_data.pkl', mode='wb') as f:
      pickle.dump(obj_knn, f)
  else:
    with open('data/KNN_data.pkl', 'rb') as f:
      obj_knn = pickle.load(f)
  prevision_knn = obj_knn.predict(x_test)
  table_report(y_test, prevision_knn, 'KNN')
  confusion_graph(y_test, prevision_knn, 'KNN')
  explainer_knn = shap.KernelExplainer(obj_knn.predict_proba, X_training)
  shap_values_knn = explainer_knn.shap_values(X_training)
  knn_importance_df = calculate_mean_shap_importance(shap_values_knn[1], X_training.columns)
  plot_feature_importance_plotly(knn_importance_df, 'KNN')




def tree_decision(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
  )-> None:
  st.markdown('### Resultado do machine learning usando o método Árvore de decisão')
  if not(os.path.isfile('data/tree_decision.pkl')):
    obj_tree_decision = DecisionTreeClassifier(criterion='entropy')
    obj_tree_decision.fit(X_training, y_training)
    with open('data/tree_decision.pkl', mode='wb') as f:
      pickle.dump(obj_tree_decision, f)
  else:
    with open('data/tree_decision.pkl', 'rb') as f:
      obj_tree_decision = pickle.load(f)
  prevision_tree = obj_tree_decision.predict(x_test)
  importances = pd.Series(
    data=obj_tree_decision.feature_importances_,
    index= ['espaco_banco_traseiro', 'tipo_carroceria', 'consumo_cidade',
       'cilindros_motor', 'cilindradas_motor', 'tipo_motor',
       'cor_exterior', 'frota', 'chassi_danificado',
       'concessionaria_franqueada', 'marca_da_franquia',
       'espaco_banco_dianteiro', 'volume_tanque', 'tipo_combustivel',
       'historico_acidente', 'altura', 'consumo_estrada', 'cavalo_de_potencia',
       'cor_interior', 'ee_cabine', 'ee_novo', 'comprimento',
       'cor_listagem', 'nome_marca', 'maximo_assentos', 'quilometragem',
       'nome_modelo', 'qtd_proprietarios', 'potencia', 'preco', 'recuperado',
       'valor_economizado', 'avaliacao_vendedor', 'nome_vendedor',
       'titulo_roubo', 'torque', 'transmissao', 'exibicao_transmissao',
       'nome_versao', 'sistema_rodas', 'exibicao_sistema_rodas', 'entre_eixos',
       'largura', 'ano']
  )
  important = importances.to_frame()
  important.reset_index(inplace=True)
  important.columns = ['Importância','Feature', ]
  st.markdown('##### Gráfico de Importância de parametros')
  fig = px.bar(data_frame=important, x='Feature', y='Importância', orientation='h', template='plotly_dark')
  table_report(y_test, prevision_tree,'Árvore de decisão')
  confusion_graph(y_test, prevision_tree, 'Árvore de decisão')
  #plot_feature_importance(dt_importances, feature_names, 'Decision Tree')
  fig.update_layout(yaxis={'categoryorder':'total ascending'})
  st.plotly_chart(fig)
  
if not(os.path.isfile('data/usedcars_usa.pkl')):
  print('iniciando...')
  main()

with open('data/usedcars_usa.pkl', 'rb') as f:
    X_training, X_test, y_training, y_test = pickle.load(f)



random_forest(X_training, y_training, X_test, y_test)
tree_decision(X_training, y_training, X_test, y_test)
KNN(X_training, y_training, X_test, y_test)