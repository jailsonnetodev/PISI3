import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
import plotly.express as px
import math
import os


def gerar_amostra_parquet(file_path, parquet_path, num_amostras=100000):
    if os.path.exists(parquet_path):
        print(f"O arquivo Parquet '{parquet_path}' já existe.")
    else:
        print(f"Gerando amostra de {num_amostras} registros e salvando como Parquet...")
        df = pd.read_csv(file_path, low_memory=False, skiprows=lambda i: i > 0 and i % (3000000 // num_amostras) != 0)
        df.to_parquet(parquet_path)
        print(f"Amostra gerada e salva como '{parquet_path}'.")


file_path = 'usedcars_usa.csv'
parquet_path = 'usedcars_usa.parquet'
gerar_amostra_parquet(file_path, parquet_path, num_amostras=100000)
df = pd.read_parquet('data/usedcars_usa.parquet')


drop_columns = ['vin','bed','bed_height','bed_length','cabin','combine_fuel_economy','dealer_zip','description','is_certified','is_cpo','is_oemcpo','latitude','listing_id','longitude','main_picture_url','sp_id','trimId','vehicle_damage_category','major_options']



#Dropar as colunas especificas
def drop_columns_from_df(df, columns_to_drop):
  df = df.drop(columns=columns_to_drop)
  return df
df = drop_columns_from_df(df, drop_columns)



#tratar outlier com o metodo IQR
def drop_outliers(df, columns, k=1.5):
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        df[column] = df[column].clip(lower=q1 - k * iqr, upper=q3 + k * iqr)
    return df

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

#INSERINDO A MEDIA NAS COLUNAS NUMERICAS 

col_media_process = ['back_legroom','front_legroom','fuel_tank_volume','height','length','mileage','seller_rating','width']


# prompt: funcao para preencher com o valor da media as colunas acima

def impute_mean(df, columns):
  for column in columns:
    mean_value = df[column].mean()
    df[column] = df[column].fillna(mean_value)
  return df

df = impute_mean(df, col_media_process)




#INSERINDO A MODA NAS COLUNAS CATEGORICAS



col_categories_process = ['has_accidents','body_type','engine_cylinders','engine_type','exterior_color','fleet','frame_damaged','franchise_make','fuel_type','has_accidents','interior_color','isCab','theft_title','torque','transmission','transmission_display','trim_name','wheel_system','wheel_system_display','wheelbase','power','salvage']

# prompt: funcao para usar a moda passando as colunas acima

def impute_mode(df, columns):
  for column in columns:
    mode_value = df[column].mode()[0]
    df[column] = df[column].fillna(mode_value)
  return df

df = impute_mode(df, col_categories_process)





#TRADUZINDO TODAS AS COLUNAS EM PT-BR PARA MELHOR ENTENDIMENTO NO PROCESSO DE ANALISE


# Dicionário de tradução das colunas
traducoes = {
    'back_legroom': 'espaco_banco_traseiro',
    'body_type': 'tipo_carroceria',
    'city': 'cidade',
    'city_fuel_economy': 'consumo_cidade',
    'daysonmarket': 'dias_no_mercado',
    'engine_cylinders': 'cilindros_motor',
    'engine_displacement': 'cilindradas_motor',
    'engine_type': 'tipo_motor',
    'exterior_color': 'cor_exterior',
    'fleet': 'frota',
    'frame_damaged': 'chassi_danificado',
    'franchise_dealer': 'concessionaria_franqueada',
    'franchise_make': 'marca_da_franquia',
    'front_legroom': 'espaco_banco_dianteiro',
    'fuel_tank_volume': 'volume_tanque',
    'fuel_type': 'tipo_combustivel',
    'has_accidents': 'historico_acidente',
    'height': 'altura',
    'highway_fuel_economy': 'consumo_estrada',
    'horsepower': 'cavalo_de_potencia',
    'interior_color': 'cor_interior',
    'isCab': 'ee_cabine',
    'is_new': 'ee_novo',
    'length': 'comprimento',
    'listed_date': 'data_listagem',
    'listing_color': 'cor_listagem',
    'make_name': 'nome_marca',
    'maximum_seating': 'maximo_assentos',
    'mileage': 'quilometragem',
    'model_name': 'nome_modelo',
    'owner_count': 'qtd_proprietarios',
    'power': 'potencia',
    'price': 'preco',
    'salvage': 'recuperado',
    'savings_amount': 'valor_economizado',
    'seller_rating': 'avaliacao_vendedor',
    'sp_name': 'nome_vendedor',
    'theft_title': 'titulo_roubo',
    'torque': 'torque',
    'transmission': 'transmissao',
    'transmission_display': 'exibicao_transmissao',
    'trim_name': 'nome_versao',
    'wheel_system': 'sistema_rodas',
    'wheel_system_display': 'exibicao_sistema_rodas',
    'wheelbase': 'entre_eixos',
    'width': 'largura',
    'year': 'ano',
}

# Renomear as colunas usando o dicionário de traduções
df = df.rename(columns=traducoes)
