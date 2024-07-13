import pandas as pd
import numpy as np
import os

def load_dataset(path: str):

    data = pd.read_csv(path)
    return data

df = load_dataset('data/price_cars.csv')
#duplicatas

def process_data(data):
  data.duplicated().sum()
  data.drop_duplicates()
  columns_drop = ['Vin']
  data = data.drop(columns= columns_drop, axis=1)
  data.rename(columns={'Price': 'preco','Year': 'ano','Mileage': 'quilometragem','City': 'cidade', 'State': 'estado', 'Make': 'marca', 'Model': 'modelo'}, inplace = True)
  data['modelo'] = data['modelo'].replace(['1','2','3','4','5','6','7','8'], ['Serie 1','Serie 2','Serie 3','Serie 4','Serie 5','Serie 6','Serie 7','Serie 8'])
  return data

df = process_data(df)




list_outliers = ['preco','ano','quilometragem']
def drop_outliers(df, columns, k=1.5):
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        df[column] = df[column].clip(lower=q1 - k * iqr, upper=q3 + k * iqr)
    return df
drop_outliers(df,list_outliers,k=1.5)


classes = [0, 13000, 18700, 26990, 47990]
labels = ['0 a 13k', '13k a 18k', '18k a 26k', '26k a 47k']
intervals = pd.cut(x=df.preco, bins=classes, labels=labels)
df['preco_categoria'] = intervals


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