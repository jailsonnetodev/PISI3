import pandas as pd
import numpy as np
import os


def extract_number_to_int(df, column):
  df[column] = df[column].str.extract('(\d+)').astype(float).fillna(0).astype(int)
  mean_value = df[column].mean()
  df[column] = df[column].replace(0, mean_value)
  df[column] = df[column].astype(int)
  return df
df = extract_number_to_int(df, 'maximum_seating')




extract_number = ['back_legroom','height','front_legroom','fuel_tank_volume','length','width','wheelbase']

# prompt: funcao para extrair apenas o numero de cada coluna acima da lista e converter pra float

def extract_number_to_float(df, columns, decimal_places=1):
    for column in columns:
        df[column] = (
            df[column]
            .str.extract('(\d+\.\d+|\d+)')
            .astype(float)
            .round(decimal_places)
        )
    drop_outliers(df, columns, k=1.5)
    return df
extract_number_to_float(df, extract_number, decimal_places=1)




#Converter para int , usar a media e tratar outliers
converter_int = ['owner_count','horsepower','city_fuel_economy','engine_displacement','highway_fuel_economy','seller_rating']


def transform_integer(df, colunas):
    for coluna in colunas:
        # Substituir valores nulos pela média da coluna
        media = df[coluna].mean()
        df[coluna].fillna(media, inplace=True)

        # Converter a coluna para inteiro
        df[coluna] = df[coluna].astype(int)

    return df

transform_integer(df, converter_int)