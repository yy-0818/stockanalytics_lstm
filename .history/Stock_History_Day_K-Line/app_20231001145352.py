import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from datetime import datetime

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



# 侧边栏：用户输入参数
st.sidebar.header('设置参数')
# 输入
hashtag = st.sidebar.text_input('输入hashtag',value='')
# button
button = st.sidebar.button('预测')

# 加载股票数据
stock_data = load_stock_data()
stock_data
X_test_set = load_stock_data().iloc[:,4:5].values
# X_test_set
# 加载预测模型
model = load_prediction_model()


