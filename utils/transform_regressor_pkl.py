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
