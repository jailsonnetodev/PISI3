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


data = pd.read_parquet('data/usedcars_usa100k.parquet')
data = data.reset_index()
data = data.drop('index',axis=1)


columns_drop = ['dias_no_mercado','dias_no_mercado_label']
# definindo as features e target
def features_and_target(data: pd.DataFrame, target: str, columns_drop):
  X_data= data.drop(columns_drop, axis=1)
  y_data= data[target]
  return X_data, y_data
