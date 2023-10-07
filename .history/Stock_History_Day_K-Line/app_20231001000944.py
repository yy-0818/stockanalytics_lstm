# 导入所需的库
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# 加载股票数据
st.cache_data
def load_stock_data():
    data = pd.read_csv('.\Stock_History_Day_K-Line\苹果.csv')  # 替换为你的股票数据文件
    return data

# 加载预测模型
st.cache_data(allow_output_mutation=True)
def load_prediction_model():
    model = load_model(r'.\Stock_History_Day_K-Line\apple_lstm_model.h5')  # 替换为你的模型文件
    return model


# 主应用程序
def main():
    # 设置页面标题
    st.title('股票价格预测应用')

    # 加载股票数据
    data = load_stock_data()

    # 加载预测模型
    model = load_prediction_model()

    # 用户输入日期
    date_input = st.date_input('请选择日期')



if __name__ == '__main__':
    main()
