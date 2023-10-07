import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os


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
def load_prediction_model(model_name):
    model_path = os.path.join('.\Stock_History_Day_K-Line\Model', f'{model_name}.h5')
    model = load_model(model_path)
    return model


# 创建Streamlit应用程序
st.markdown("<h1 style='text-align: center; color: black;'>基于 LSTM 模型股票市场分析与预测📈</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Stock Market Analysis 📈 + Prediction using LSTM</h4>", unsafe_allow_html=True)



# sidebar侧边栏：用户输入参数
def user_input_features():
    st.sidebar.header('设置参数📁')
    st.sidebar.write('User input parameters below ⬇️')
    hashtag = st.sidebar.text_input('输入hashtag', value='', key='hashtag_input')
    button = st.sidebar.button('Get Data')

    model_path = st.sidebar.selectbox("选择模型",('RNN', 'LSTM'),key='model_selection')


    a9 = st.sidebar.selectbox("Agent Status?", ('Happy','Sad','Normal'))
    output = [model_path,a9]
    return output
    
outputdf = user_input_features()
model_name = user_input_features()[0]


# 加载股票数据
stock_data = load_stock_data()
stock_data
# 加载预测模型
model = load_prediction_model()

# 2. 选择需要的特征（收盘）并进行归一化
X_train_set = stock_data[['收盘']].values.astype(float)
scaler = MinMaxScaler()
X_train_set = scaler.fit_transform(X_train_set)
# X_train_set

# 提取收盘值
X_test_set = load_stock_data().iloc[:,4:5].values

#取出几天前股价来建立成特征和标签数据集
def create_dataset(ds, look_back=1):
    X_data, Y_data = [],[]
    for i in range(len(ds)-look_back):
        X_data.append(ds[i:(i+look_back), 0])
        Y_data.append(ds[i+look_back, 0])
    return np.array(X_data), np.array(Y_data)
look_back = 60

# 分割成特征数据和标签数据
X_train, Y_train = create_dataset(X_train_set, look_back)
# 转换成(样本数, 时步, 特征)张量  
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

# 产生标签数据




# 主函数
def main():
    # 加载选定的模型
    model = load_prediction_model(model_name)

if __name__ == "__main__":
    main()