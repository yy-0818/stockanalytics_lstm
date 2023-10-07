import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt

plt.style.use('default')


st.set_page_config(
    page_title = 'Real-Time Stock Price Prediction',
    page_icon = '🕵️‍♀️',
    # layout = 'wide'
)

# 加载股票数据
@st.cache_data
def load_stock_data():
    data = pd.read_csv('.\Stock_History_Day_K-Line\苹果.csv')  # 替换为你的股票数据文件
    # data['日期'] = pd.to_datetime(data['日期'])
    # X_test_set = data.iloc[:,4:5].values
    return data


# 加载预测模型
@st.cache_data
def load_prediction_model():
    model = load_model(r'.\Stock_History_Day_K-Line\apple_lstm_model.h5')  # 替换为你的模型文件
    return model


# 创建Streamlit应用程序
st.markdown("<h1 style='text-align: center; color: black;'>深度学习:LSTM股票价格预测应用</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'></h1>", unsafe_allow_html=True)



# sidebar侧边栏：用户输入参数
# st.sidebar.header('设置参数')
# # 输入
# hashtag = st.sidebar.text_input('输入hashtag',value='')
# # button
# button = st.sidebar.button('预测')
# side-bar 
def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.slider('Action1', -31.0, 3.0, 0.0)
    a2 = st.sidebar.slider('Action2', -5.0, 13.0, 0.0)
    a3 = st.sidebar.slider('Action3', -20.0, 6.0, 0.0)
    a4 = st.sidebar.slider('Action4', -26.0, 7.0, 0.0)
    a5 = st.sidebar.slider('Action5', -4.0, 5.0, 0.0)
    a6 = st.sidebar.slider('Action6', -8.0, 4.0, 0.0)
    a7 = st.sidebar.slider('Sales Amount', 1.0, 5000.0, 1000.0)
    a8 = st.sidebar.selectbox("Gender?", ('Male', 'Female'))
    a9 = st.sidebar.selectbox("Agent Status?", ('Happy','Sad','Normal'))
    
    output = [a1,a2,a3,a4,a5,a6,a7,a8,a9]
    return output


# 加载股票数据
stock_data = load_stock_data()
stock_data
X_test_set = load_stock_data().iloc[:,4:5].values
# X_test_set
# 加载预测模型
model = load_prediction_model()


