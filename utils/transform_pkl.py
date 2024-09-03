import pandas as pd
import numpy as np
import pickle
import os
from typing import List
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

data = pd.read_parquet('data\dataprocess.parquet')

columns_drop = ['dias_no_mercado','dias_no_mercado_label']
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
  X_data, y_data = features_and_target(data,'dias_no_mercado_label',columns_drop)
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
  if not(os.path.isfile('data/usedcars_usa.pkl')):
    save_pkl(X_data, y_data,'data/usedcars_usa.pkl')
  X_training, X_test, y_training, y_test = save_pkl(X_data, y_data,'data/usedcars_usa.pkl')

main()

with open('data/usedcars_usa.pkl', 'rb') as f:
    X_training, X_test, y_training, y_test = pickle.load(f)

