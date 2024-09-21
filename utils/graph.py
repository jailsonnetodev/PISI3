import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

#grafico de caixa
def boxplot(data, x:str, y:str='', title: str='', p: str=''):
    if y == '':
        fig_box = px.box(data,x)
    else:
        fig_box = px.box(data,x,y)

    field = st.container()
    field.markdown(f'### {title}')
    field.plotly_chart(fig_box)
    field.markdown(p, unsafe_allow_html=True)


#grafico de dispersao
def scatter(data, x:str, title:str='', y:str='',color:str='',p:str=''):
    if y=='':
        fig_scatter = px.scatter(data, x)
    else:
        fig_scatter = px.scatter(data, x, y)

    field = st.container()
    field.markdown(f'### {title}')
    field.plotly_chart(fig_scatter)
    field.markdown(p, unsafe_allow_html=True)



#grafico treemap plotly
def treemap(data, options, default=[]):
  col1, col2 = st.columns((3, 1))
  
  data = data
  options = options
  with col1:
    select = st.multiselect(
      'Selecione dois elementos', 
      options=options, 
      default=default, 
      max_selections=3
    )
  button = col2.empty()
  if button.button(label='Análisar', use_container_width=25):
    button.button("Analisando...", use_container_width=25, disabled=True)
    st.plotly_chart(px.treemap(data, path=select))


#grafico histograma
def hist(data, x:str, y:str='',title:str='',p:str=''):
    if y=='':
        fig_hist = px.histogram(data, x)
    else:
        fig_hist= px.histogram(data, x, y)

    field = st.container()
    field.markdown(f'### {title}')
    field.plotly_chart(fig_hist)
    field.markdown(p, unsafe_allow_html=True)


def strip(data, x:str, y:str='', title:str='', p:str=''):
    if y=='':
        fig_strip = px.strip(data, x)
    else:
        fig_strip = px.strip(data, x, y)
    
    field = st.container()
    field.markdown(f'### {title}')
    field.plotly_chart(fig_strip)
    field.markdown(p, unsafe_allow_html=True)

#grafico de barras plotly

def bar(data, x:str, y:str='', title:str='', mode='group', p:str=''):
    if y=='':
        fig_bar = px.bar(data, x, barmode=mode)
    else:
        fig_bar = px.bar(data, x, y, barmode=mode)
    
    field = st.container()
    field.markdown(f'### {title}')
    field.plotly_chart(fig_bar)
    field.markdown(p, unsafe_allow_html=True)


def line_graph(data, x:str, y:str='', title:str='', p:str=''):
    if y=='':
        fig_bar = px.line(data, x, )
    else:
        fig_bar = px.line(data, x, y, )
    
    field = st.container()
    field.markdown(f'### {title}')
    field.plotly_chart(fig_bar)
    field.markdown(p, unsafe_allow_html=True)



#plotagem de graficos genericos, informe o tipo de grafico e separação de cores
def select_graph(
  data: pd.DataFrame, x: str, options: np.ndarray, type_graph, type_txt: str, p: str=''
  ):

  st.markdown(f'## {type_txt.title()} {x.title()}')
  col1, col2 = st.columns((3, 2))
  options = options.tolist()
  options.remove(x)
  options.insert(-1, 'Total')
  options.insert(0, '')
  with col1:
    st.write(f"Selecione uma opção: ")
    select = st.selectbox(
      f"{x}{type_graph}",
      options=options,
      index=0,
      label_visibility='hidden'
    )
  with col2:
    st.write('Escolha a opção de cores')
    color = st.selectbox(
      f"{x}{type_txt}",
      options=options,
      label_visibility='hidden'
    )
  if select != '':
    if select != 'Total':
      new_data = data.groupby([x, select]).size().reset_index(name='Total')
    else:
      new_data = data.groupby([x]).size().reset_index(name='Total')
    if color == '':
      st.plotly_chart(type_graph(new_data, x=x, y=select,))
    elif color == 'Total':
      st.plotly_chart(type_graph(new_data, x=x, y=select, color=color))
    else:
      if select != 'Total':
        new_data = data.groupby([x, select, color]).size().reset_index(name='Total')
      else:
        new_data = data.groupby([x, color]).size().reset_index(name='Total')
      st.plotly_chart(type_graph(new_data, x=x, y=select, color=color))