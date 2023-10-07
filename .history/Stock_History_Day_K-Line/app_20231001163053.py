import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt

plt.style.use('default')


st.set_page_config(
    page_title = 'Real-Time Stock Price Prediction',
    page_icon = '📋',
    layout="wide",
    initial_sidebar_state="expanded",
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
st.markdown("<h1 style='text-align: center; color: black;'>基于 LSTM 模型股票市场分析与预测📈</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Stock Market Analysis 📈 + Prediction using LSTM</h4>", unsafe_allow_html=True)



# sidebar侧边栏：用户输入参数
def user_input_features():
    st.sidebar.header('设置参数📁')
    st.sidebar.write('User input parameters below ⬇️')
    hashtag = st.sidebar.text_input('输入hashtag',value='')
    button = st.sidebar.button('Get Data')

    a8 = st.sidebar.selectbox("Select model?", ('RNN', 'LSTM'))
    a9 = st.sidebar.selectbox("Agent Status?", ('Happy','Sad','Normal'))
    output = [a8,a9]
    return output
outputdf = user_input_features()

# 加载股票数据
stock_data = load_stock_data()
stock_data
X_test_set = load_stock_data().iloc[:,4:5].values
# X_test_set
# 加载预测模型
model = load_prediction_model()


# 预测并绘制plot
prediction = model.predict(X_test_set)
prediction = prediction.reshape(prediction.shape[0])
prediction
#  将预测值转换回股价
prediction = prediction * stock_data['Close'][0]
prediction
