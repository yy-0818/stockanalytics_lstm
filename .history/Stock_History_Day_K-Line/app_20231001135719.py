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
st.title('股票价格预测应用')
# 侧边栏：用户输入参数
st.sidebar.header('设置参数')

# 加载股票数据
stock_data = load_stock_data()
stock_data


# 加载预测模型
model = load_prediction_model()


