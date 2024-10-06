import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def sim_nao(value):
    return 1 if value == "Sim" else 0


def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

classification_model = load_model('models/KNN.pkl') 


def label_encoder(x_data):
    le = LabelEncoder()
    for col in x_data:
        if x_data[col].dtype == 'object' or x_data[col].dtype.name == 'category':
            x_data[col] = le.fit_transform(x_data[col])
    return x_data


def standard(x_data):
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    return x_data


st.title("Predição de Classificação")

col1, _, col2 = st.columns([5,1,5])


with col1:
    espaco_banco_traseiro = st.slider('Espaço Banco Traseiro (polegadas)', 30.95, 44.95)
    tipo_carroceria = st.selectbox('Tipo de Carroceria', ['SUV / Crossover', 'Minivan', 'Van', 'Sedan', 'Pickup Truck', 'Wagon', 'Hatchback', 'Coupe', 'Convertible'])
    cidade = st.selectbox('Cidade', ['Tucson', 'Indianapolis', 'Houston', 'Phoenix', 'Tampa',
       'Columbia', 'Miami', 'San Antonio', 'Charlotte', 'Denver',
       'Oklahoma City', 'Orlando', 'Springfield', 'Jacksonville',
       'Cincinnati', 'Columbus', 'Dallas', 'Austin', 'Las Vegas',
       'Raleigh'])
    consumo_cidade = st.slider('Consumo na Cidade (mpg)', 8.0, 35.5)
    cilindros_motor = st.selectbox('Cilindros do Motor', ['V6', 'I4', 'I4 Hybrid', 'H4', 'V8', 'V8 Biodiesel', 'V6 Flex Fuel Vehicle', 'V8 Flex Fuel Vehicle', 'I6 Diesel', 'I6', 'I3', 'I4 Flex Fuel Vehicle', 'V8 Diesel', 'I4 Diesel', 'I5', 'V6 Diesel', 'V6 Biodiesel', 'H6', 'V6 Hybrid', 'W12', 'V12', 'V10', 'V8 Hybrid', 'I2', 'V8 Compressed Natural Gas', 'W12 Flex Fuel Vehicle', 'H4 Hybrid', 'I5 Biodiesel', 'V8 Propane', 'I3 Hybrid', 'R2'])
    cilindradas_motor = st.selectbox('Cilindradas do Motor (cc)', [3600, 3500, 2400, 2500, 2000, 1800, 4400, 5750, 3000, 1500, 5700, 5300, 1400, 4300, 2969, 3800, 1600, 2300, 4600, 5500, 1000, 5000, 4000, 3300, 3700, 1200, 5400, 2700, 3200, 5600, 1300, 3400, 4800, 4700, 2200, 4200, 3900, 2800, 1900, 2900, 1700, 900, 5200, 700, 2100, 3100, 4900, 4100, 4500, 2600])
    tipo_motor = st.selectbox('Tipo de Motor', ['V6', 'I4', 'I4 Hybrid', 'H4', 'V8', 'V8 Biodiesel', 'V6 Flex Fuel Vehicle', 'V8 Flex Fuel Vehicle', 'I6 Diesel', 'I6', 'I3', 'I4 Flex Fuel Vehicle', 'V8 Diesel', 'I4 Diesel', 'I5', 'V6 Diesel', 'V6 Biodiesel', 'H6', 'V6 Hybrid', 'W12', 'V12', 'V10', 'V8 Hybrid', 'I2', 'V8 Compressed Natural Gas', 'W12 Flex Fuel Vehicle', 'H4 Hybrid', 'I5 Biodiesel', 'V8 Propane', 'I3 Hybrid', 'R2'])
    cor_exterior = st.selectbox('Cor Exterior',['Silver', 'Red', 'Blue', 'Black', 'Oxford White', 'Summit White',
       'Gun Metallic', 'Diamond Black Crystal Pearlcoat', 'Gray',
       'Mosaic Black Metallic', 'White', 'BLACK', 'Crystal Black Pearl',
       'Granite Crystal Metallic Clearcoat', 'Satin Steel Metallic',
       'Modern Steel Metallic', 'Silver Ice Metallic',
       'Magnetic Metallic', 'Platinum White Pearl',
       'Bright White Clearcoat'])
    frota = sim_nao(st.radio('É de Frota?', ['Sim', 'Não']))
    chassi_danificado = sim_nao(st.radio('Chassi Danificado?', ['Sim', 'Não']))
    concessionaria_franqueada = sim_nao(st.radio('Concessionária Franqueada?', ['Sim', 'Não']))
    marca_da_franquia = st.selectbox('Marca da Franquia', ['Volkswagen', 'Nissan', 'RAM', 'Mazda', 'Ford', 'Dodge', 'Hyundai', 'Toyota', 'Honda', 'Jeep', 'Chevrolet', 'GMC', 'Mercedes-Benz', 'Subaru', 'Kia', 'Buick', 'BMW', 'Audi', 'Land Rover', 'Lexus', 'Mitsubishi', 'Chrysler', 'Acura', 'Bentley', 'INFINITI', 'Scion', 'Cadillac', 'Volvo', 'Lincoln', 'Maserati', 'Porsche', 'FIAT', 'Jaguar', 'MINI', 'Freightliner', 'Ferrari', 'Genesis', 'Lamborghini', 'Rolls-Royce', 'McLaren', 'Alfa Romeo', 'Lotus', 'Aston Martin', 'Pagani'])
    espaco_banco_dianteiro = st.slider('Espaço Banco Dianteiro (polegadas)', 38.25, 45.85)
    volume_tanque = st.slider('Volume do Tanque (gallons)', 5.5, 30.29)
    tipo_combustivel = st.selectbox('Tipo de Combustível', ['Gasoline', 'Hybrid', 'Biodiesel', 'Flex Fuel Vehicle', 'Diesel', 'Electric', 'Compressed Natural Gas', 'Propane'])
    historico_acidente = sim_nao(st.radio('Histórico de Acidente?', ['Sim', 'Não']))
    altura = st.slider('Altura (polegadas)', 43.5, 87.69)
    consumo_estrada = st.slider('Consumo na Estrada (mpg)', 14.5, 42.5)
    cavalo_de_potencia = st.slider('Cavalos de Potência', 70, 450)
    cor_interior = st.selectbox('Cor Interior',['Black', 'Brown (Beige)', 'Black (Charcoal)', 'Jet Black',
       'Charcoal Black', 'Gray', 'jet black', 'White (Ivory)',
       'Black (Ebony)', 'Ebony Black', 'Medium Earth Gray',
       'Medium Ash Gray', 'Diesel Gray/Black', 'Black (charcoal)',
       'Brown (Tan)', 'Titan Black', 'Ash', 'Graphite', 'Light Gray',
       'JET BLACK'])
    ee_cabine = sim_nao(st.radio('Cabine Estendida?', ['Sim', 'Não']))
    ee_novo = sim_nao(st.radio('É Novo?', ['Sim', 'Não']))
    comprimento = st.slider('Comprimento (polegadas)', 155.85, 225.85)
    cor_listagem = st.selectbox('Cor na Listagem', ['GRAY', 'WHITE', 'BLACK', 'UNKNOWN', 'ORANGE', 'BLUE', 'SILVER', 'RED', 'BROWN', 'GREEN', 'GOLD', 'TEAL', 'YELLOW', 'PURPLE', 'PINK'])

with col2:
    nome_marca = st.selectbox('Nome da Marca', ['Volkswagen', 'Nissan', 'RAM', 'Mazda', 'Honda', 'Dodge', 'Ford', 'Hyundai', 'Subaru', 'Land Rover', 'Toyota', 'Cadillac', 'Jeep', 'Chevrolet', 'GMC', 'Mercedes-Benz', 'Kia', 'Buick', 'BMW', 'Audi', 'Lexus', 'Mitsubishi', 'Chrysler', 'Acura', 'INFINITI', 'Porsche', 'Pontiac', 'Lincoln', 'Scion', 'Maserati', 'Hummer', 'Volvo', 'Mercury', 'AM General', 'MINI', 'Alfa Romeo', 'Jaguar', 'FIAT', 'Freightliner', 'Ferrari', 'Genesis', 'Saturn', 'McLaren', 'Bentley', 'Tesla', 'Saab', 'MG', 'Suzuki', 'smart', 'Lamborghini', 'Oldsmobile', 'Rolls-Royce', 'Karma', 'Datsun', 'Aston Martin', 'Plymouth', 'Shelby', 'Isuzu', 'Studebaker'])
    maximo_assentos = st.selectbox('Máximo de Assentos', [5, 7, 2, 8, 6, 4, 3, 9, 15, 12, 10])
    quilometragem = st.slider('Quilometragem (milhas)', 0.0, 103093.5)
    nome_modelo = st.selectbox('Nome do Modelo', ['Accord', 'Rogue', 'Escape', 'Tucson', 'Trax', 'Malibu', '1500', 'Equinox', 'Grand Cherokee', 'F-150', 'Altima', 'Silverado 1500', 'Cherokee', 'Explorer', 'Camry', 'CR-V', 'Fusion', 'Civic', 'RAV4', 'Corolla'])
    qtd_proprietarios = st.selectbox('Quantidade de Proprietários', [1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 12])
    potencia = st.selectbox('Potencia', ['355 hp @ 5,600 RPM', '395 hp @ 5,600 RPM', '190 hp @ 5,600 RPM',
       '310 hp @ 6,700 RPM', '375 hp @ 5,000 RPM', '138 hp @ 4,900 RPM',
       '280 hp @ 6,000 RPM', '395 hp @ 5,750 RPM', '158 hp @ 6,500 RPM',
       '181 hp @ 6,000 RPM', '170 hp @ 5,600 RPM', '147 hp @ 6,200 RPM',
       '170 hp @ 6,000 RPM', '360 hp @ 5,150 RPM', '295 hp @ 6,400 RPM',
       '185 hp @ 6,000 RPM', '283 hp @ 6,400 RPM', '310 hp @ 6,800 RPM',
       '250 hp @ 5,500 RPM', '245 hp @ 5,500 RPM'])
    preco = st.slider('Preco', 484.0, 68328.0)
    recuperado = sim_nao(st.radio('Recuperado?', ['Sim', 'Não']))
    valor_economizado = st.slider('Valor Economizado', 0.0 ,1957.5)
    avaliacao_vendedor = st.selectbox('Avaliacao do Vendedor', [4, 3, 2, 5, 1])
    nome_vendedor = st.selectbox('Nome do Vendedor(Concessionaria)', ['Planet Ford', 'Carvana',
       'AutoNation Chrysler Dodge Jeep Ram Spring',
       'Randy Marion Chevrolet Buick Cadillac', 'Vroom',
       'Huntington Beach Chrysler Dodge Jeep Ram', 'Brunswick Auto Mart',
       'Brandon Ford', 'Honda World', 'Avis Car Sales - Denver',
       'Miami Lakes Automall', 'San Tan Ford',
       'All Star Dodge Chrysler Jeep Ram',
       'Russell Westbrook Chrysler Dodge Jeep Ram of Van Nuys',
       'Varsity Ford', 'McCluskey Chevrolet',
       'Kernersville Chrysler Dodge', 'Dorian Ford',
       'Bomnin Chevrolet Dadeland', "Jeff Belzer's Chevrolet"])
    titulo_roubo = sim_nao(st.radio('Titulo de Roubo?', ['Sim', 'Não']))
    torque = st.selectbox('Torque', ['383 lb-ft @ 4,100 RPM', '175 lb-ft @ 4,400 RPM',
       '275 lb-ft @ 3,000 RPM', '148 lb-ft @ 200 RPM',
       '184 lb-ft @ 2,500 RPM', '410 lb-ft @ 3,950 RPM',
       '178 lb-ft @ 4,000 RPM', '400 lb-ft @ 4,500 RPM',
       '180 lb-ft @ 3,600 RPM', '262 lb-ft @ 4,700 RPM',
       '260 lb-ft @ 4,800 RPM', '132 lb-ft @ 4,500 RPM',
       '203 lb-ft @ 2,000 RPM', '266 lb-ft @ 2,800 RPM',
       '263 lb-ft @ 4,700 RPM', '260 lb-ft @ 4,400 RPM',
       '185 lb-ft @ 4,320 RPM', '138 lb-ft @ 4,200 RPM',
       '390 lb-ft @ 4,250 RPM', '179 lb-ft @ 2,000 RPM'])
    transmissao = st.selectbox('Transmissao', ['A', 'CVT', 'M', 'Dual Clutch'])
    exibicao_transmissao = st.selectbox('Exibicao da Transmissao', ['Automatic', 'Continuously Variable Transmission',
       '6-Speed Automatic', '5-Speed Manual', '5-Speed Automatic',
       '8-Speed Automatic', '6-Speed CVT', '6-Speed Automatic Overdrive',
       '9-Speed Automatic', '7-Speed Automatic', '4-Speed Automatic',
       '1-Speed Automatic', '6-Speed Dual Clutch',
       '9-Speed Automatic Overdrive', 'Manual', '7-Speed CVT',
       '4-Speed Automatic Overdrive', '8-Speed Automatic Overdrive',
       '5-Speed Automatic Overdrive', '6-Speed Manual',
       '7-Speed Automatic Overdrive', '6-Speed Manual Overdrive',
       '5-Speed Manual Overdrive', '1-Speed CVT', '8-Speed Dual Clutch',
       '4-Speed Manual', '7-Speed Dual Clutch', '3-Speed Manual',
       '7-Speed Manual', '2-Speed Automatic', '8-Speed CVT',
       '3-Speed Automatic', '8-Speed Manual', '9-Speed Dual Clutch',
       '10-Speed Automatic', '1-Speed Dual Clutch', '4-Speed CVT'])
    nome_versao = st.selectbox('Nome da Versao', ['SE FWD', 'SEL AWD', 'LS FWD', 'LT FWD', 'XLT SuperCrew 4WD',
       'LE FWD', 'Limited AWD', 'LX AWD', 'Limited FWD', 'SE', 'S FWD',
       'LX FWD', 'SV FWD', 'Limited 4WD', 'LX', 'SE AWD', 'FWD',
       'Touring FWD', 'LE', 'SEL FWD'])
    sistema_rodas = st.selectbox('Sistema de Rodas', ['AWD', 'FWD', 'RWD', '4WD', '4X2'])
    exibicao_sistema_rodas = st.selectbox('Exibicao Sistema de Rodas', ['All-Wheel Drive', 'Front-Wheel Drive', 'Rear-Wheel Drive',
       'Four-Wheel Drive', '4X2'])   
    entre_eixos = st.slider('Entre Eixos',87.39, 137.8)
    largura = st.slider('Largura',57.5, 97.5)
    ano = st.selectbox('Ano de Fabricacao', [2020. , 2015. , 2019. , 2014. , 2021. , 2012.5, 2017. , 2013. ,
       2018. , 2016. ])


input_data = pd.DataFrame({
    'espaco_banco_traseiro': [espaco_banco_traseiro],
    'tipo_carroceria': [tipo_carroceria],
    'cidade': [cidade],
    'consumo_cidade': [consumo_cidade],
    'cilindros_motor': [cilindros_motor],
    'cilindradas_motor': [cilindradas_motor],
    'tipo_motor': [tipo_motor],
    'cor_exterior': [cor_exterior],
    'frota': [frota],
    'chassi_danificado': [chassi_danificado],
    'concessionaria_franqueada': [concessionaria_franqueada],
    'marca_da_franquia': [marca_da_franquia],
    'espaco_banco_dianteiro': [espaco_banco_dianteiro],
    'volume_tanque': [volume_tanque],
    'tipo_combustivel': [tipo_combustivel],
    'historico_acidente': [historico_acidente],
    'altura': [altura],
    'consumo_estrada': [consumo_estrada],
    'cavalo_de_potencia': [cavalo_de_potencia],
    'cor_interior': [cor_interior],
    'ee_cabine': [ee_cabine],
    'ee_novo': [ee_novo],
    'comprimento': [comprimento],
    'cor_listagem': [cor_listagem],
    'nome_marca': [nome_marca],
    'maximo_assentos': [maximo_assentos],
    'quilometragem': [quilometragem],
    'nome_modelo': [nome_modelo],
    'qtd_proprietarios': [qtd_proprietarios],
    'potencia': [potencia],
    'preco': [preco],
    'recuperado': [recuperado],
    'valor_economizado': [valor_economizado],
    'avaliacao_vendedor': [avaliacao_vendedor],
    'nome_vendedor': [nome_vendedor],
    'titulo_roubo': [titulo_roubo],
    'torque': [torque],
    'transmissao': [transmissao],
    'exibicao_transmissao': [exibicao_transmissao],
    'nome_versao': [nome_versao],
    'sistema_rodas': [sistema_rodas],
    'exibicao_sistema_rodas': [exibicao_sistema_rodas],
    'entre_eixos': [entre_eixos],
    'largura': [largura],
    'ano': [ano]
})


input_data = label_encoder(input_data)
input_data = standard(input_data)

if st.button("Realizar Predição"):
    resultado = classification_model.predict(input_data)
    st.write(f"Resultado da Classificação: {resultado[0]}")
