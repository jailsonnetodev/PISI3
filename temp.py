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


import pandas as pd
import numpy as np
import os

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