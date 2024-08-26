import pandas as pd
import streamlit as st
import plotly.express as px

## Reservados para codigos temp e outros nao executaveis##

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

  

def naive_bayes(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, obj_naive= None
  )-> None:
  st.markdown('### Resultado do machine learning usando o método Naive Bayes')
  if not(os.path.isfile('data/naive_bayes.pkl')):
    obj_naive = GaussianNB()
    obj_naive.fit(X_training, y_training)
    with open('data/naive_bayes.pkl', mode='wb') as f:
      pickle.dump(obj_naive, f)
  else:
    with open('data/naive_bayes.pkl', 'rb') as f:
      obj_naive = pickle.load(f)
  previsor = obj_naive.predict(X_test)
  table_report(y_test, previsor, 'Naive Bayes')
  confusion_graph(y_test, previsor, 'Naive Bayes')
  #matrix(X_training, X_test, y_training, y_test, mtd=obj_naive)


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
  table_report(y_test, prevision_tree,'Árvore de decisão')
  confusion_graph(y_test, prevision_tree, 'Árvore de decisão')

  
def random_forest(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
  )-> None:
  st.markdown('### Resultado do machine learning usando o método Random Forest')

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
    index=['ano', 'cidade', 'estado', 'marca', 'modelo']
  )
  important = importances.to_frame()
  important.reset_index(inplace=True)
  important.columns = ['Importância','Feature', ]
  st.markdown('##### Gráfico de Importância de parametros')
  st.plotly_chart(px.bar(data_frame=important, x='Feature', y='Importância', orientation='h', template='plotly_dark'))
  table_report(y_test, prevision_random_forest, 'Random Forest')
  confusion_graph(y_test, prevision_random_forest, 'Random Forest')
  #matrix(X_training, X_test, y_training, y_test, mtd=obj_random_forest)



def KNN(
  x_training: np.ndarray, y_training: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
  )-> None:
  st.markdown('### Resultado do machine learning usando o método KNN')
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






if not(os.path.isfile('data/price_cars.pkl')):
  print('iniciando...')
  main()

with open('data/price_cars.pkl', 'rb') as f:
    X_training, X_test, y_training, y_test = pickle.load(f)

    
naive_bayes(X_training, y_training, X_test, y_test)
tree_decision(X_training, y_training, X_test, y_test)
random_forest(X_training, y_training, X_test, y_test)
KNN(X_training, y_training, X_test, y_test)


#transform_pkl.py

columns_drop = ['preco','preco_intervalo','cidade','quilometragem','preco_ajustado','preco_depreciado']
# definindo as features e target
def features_and_target(data: pd.DataFrame, target: str, columns_drop):
  X_data= data.drop(columns_drop, axis=1)
  y_data= data[target]
  return X_data, y_data

def standard(x_data):
  scaler = StandardScaler()
  x_data = scaler.fit_transform(x_data)

  return x_data


def onehot_encoder(x_label: np.ndarray, columns: List[int]) -> np.ndarray:
  onehot = ColumnTransformer(transformers=[(
    'OneHot', 
    OneHotEncoder(), 
    columns,
    )], remainder='passthrough')
  x = onehot.fit_transform(x_label).toarray()
  return x


def pre_processing(data):
  X_data, y_data = features_and_target(data,'preco_intervalo',columns_drop)
  le=LabelEncoder()
  for col in X_data:
    if X_data[col].dtypes == 'object':
      X_data[col] = pd.DataFrame(le.fit_transform(X_data[col]))
  
  X_data = X_data.values
  y_data = y_data.values
  X_data = standard(X_data)

  return X_data, y_data

def save_pkl(
  x_data: np.ndarray, y_data: np.ndarray, path: str = 'data.pkl', per: int =0.2, random: int=0
  )-> None:
  '''save(x_training, y_training, x_teste, y_teste)'''
  X_training, X_test, y_training, y_test = train_test_split(
  X_data, y_data, test_size=per, random_state=random
    )
  with open(path, mode='wb') as f:
    pickle.dump([X_training, X_test, y_training, y_test], f)

  return X_training, X_test, y_training, y_test

X_data, y_data = pre_processing(data)

def main():
  if not(os.path.isfile('data/price_cars.pkl')):
    save_pkl(X_data, y_data,'data/price_cars.pkl')
  X_training, X_test, y_training, y_test = save_pkl(X_data, y_data,'price_cars.pkl')

main()

with open('data/price_cars.pkl', 'rb') as f:
    X_training, X_test, y_training, y_test = pickle.load(f)