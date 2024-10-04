# Cars Price Predictions

<h4 align="center">
	🚧  Em progresso 🚀 🚧
</h4>

## 💻 Sobre o Projeto

Este projeto academico foi desenvolvido com objetivo de aplicar os conhecimentos de analise exploratoria de dados com metodos estatisticos e algoritimos de machine learning, o dataset predict_car_used que esta hospedado no site kaggle.com/datasets, [Car Price Predictions]( https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset/data) foi selecionado e tratado para obter o melhor resultado dos algoritimos. O objetivo do projeto é buscar uma relação entre os atributos para prever o preço do veiculo mais justo e competitivo dentre o seguimento na venda de usados.

Tabela de Conteúdos
=================
<!--ts-->
  * [Sobre o projeto](#-sobre-o-projeto)
  * [Funcionalidades](#-funcionalidades)
  * [Layout](#-layout)
  * [Como executar o projeto](#-como-executar-o-projeto)
    * [Pré-requisitos](#pré-requisitos)
    * [Rodando a aplicação web (Streamlit)](#user-content--rodando-a-aplicação-web-frontend)
  * [Tecnologias](#-tecnologias)
  * [Autor(es)](#-autor(es))
  * [Licença](#user-content--licença)
<!--te-->





## 🎨 Layout

## ⚙️ Funcionalidades

- [ ] transformat database
- [ ] Cabeçalho
- [ ] Gráfico BoxPlot
- [ ] Gráfico de Dispersão
- [ ] Gráfico Treemap
- [ ] Análise Exploratória
- [ ] Machine Learning
- [ ] Comparate Machine Learning



## 🚀 Como executar o projeto

Este projeto foi desenvolvido e executado no servidor Streamlit:  [Web App(streamlit)](https://streamlit.io)



### Pré-requisitos

Antes de começar, você vai precisar ter instalado em sua máquina as seguintes ferramentas:
[Python](https://www.python.org/downloads/) e um editor de código [VSCode](https://code.visualstudio.com/).
Baixe o dataset que será usado no projeto: [Car Price Predictions]( https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset/data), salve na pasta do projeto gp5/data e renomeie o arquivo para price_cars.



#### 🧭 Rodando a aplicação web (Frontend)

bash
```
# Clone este repositório
$ git clone https://github.com/jailsonnetodev/PISI3.git

# Acesse a pasta do projeto no seu terminal/cmd
$ cd PISI3


# Abra o projeto no VScode
$ code .

# Antes de mais nada , verifique se voce esta com a versao 3.10 do python instalada em sua maquina

# Se solicitado, voce podera atualizar a versao do PIP
$ python -m pip install --upgrade pip

# Recomendamos criar um abiente virtual com mesmo nome da pasta do projeto.
$ python -m venv venv
# no linux
$ virtualenv venv

# Instale as dependências
$ pip install -r requirements.txt

# Converta o dataset em .parquet
$ cd utils
$ python util.py

# Executa a aplicação streamlit
$ cd ..
$ streamlit run Home.py

# A aplicação será aberta na porta:Local URL: http://localhost:8501
  Network URL: http://192.168.0.103:8501

```

---
## Video explicativo

<p align = "center">
<img width="600" height=auto src=" ">
</p>

---
## 🛠 Tecnologias

As seguintes ferramentas foram usadas na construção do projeto:

#### *WebApp*  ([Streamlit](https://streamlit.io)  +  [Python](https://www.python.org))

-   *[Pandas](https://pandas.pydata.org)*
-   *[Numpy](https://numpy.org)*
-   **[Scikit-Learn](https://scikit-learn.org/stable/)**
-   *[MatplotLib](https://matplotlib.org)*
-   *[Plotly](https://plotly.com)*

---

### Autor(es)

* Silas Ribeiro
* Jailson Neto
* Ronald Barbosa
* Yrlan Kléber
* João Gabriel

---

## 📝 Licença

Este projeto esta sobe a licença [MIT](./LICENSE).

Feito por  Silas Ribeiro | Jailson Neto | Ronald Barbosa | Yrlan Kléber | João Gabriel

---
