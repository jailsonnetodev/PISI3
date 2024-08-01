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