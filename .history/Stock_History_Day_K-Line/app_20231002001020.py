import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# plt.style.use('default')


st.set_page_config(
    page_title = 'Real-Time Stock Price Prediction',
    page_icon = '📋',
    # layout="wide",
    initial_sidebar_state="expanded",
)

# 加载股票数据
@st.cache_data
def load_stock_data():
    data = pd.read_csv('.\Stock_History_Day_K-Line\苹果.csv')  # 替换为你的股票数据文件
    return data


# 加载预测模型
@st.cache_data
def load_prediction_model():
    model = load_model(r'.\Stock_History_Day_K-Line\Model\apple_rnn_model.h5')  # 替换为你的模型文件
    return model


# 创建Streamlit应用程序
st.markdown("<h1 style='text-align: center; color: black;'>基于 LSTM 模型股票分析与预测📈</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Stock Market Analysis + Prediction using LSTM📈</h4>", unsafe_allow_html=True)



# sidebar侧边栏：用户输入参数
def user_input_features():
    st.sidebar.header('设置参数📁')
    st.sidebar.write('User input parameters below ⬇️')
    hashtag = st.sidebar.text_input('输入hashtag',value='')
    button = st.sidebar.button('Get Data')
    
    # 添加模型选择框
    model_path = st.sidebar.selectbox("选择模型", ('RNN', 'LSTM'))


    output = [hashtag, button, model_path]
    return output
output_data = user_input_features()



# 加载股票数据
stock_data = load_stock_data()
st.write(stock_data)



#取出几天前股价来建立成特征和标签数据集
def create_dataset(ds, look_back=1, scaler=None):
    X_data, Y_data = [],[]
    for i in range(len(ds)-look_back):
        X_data.append(ds[i:(i+look_back), 0])
        Y_data.append(ds[i+look_back, 0])
    return np.array(X_data), np.array(Y_data)
look_back = 60
# 主函数
def main():
    # 提取用户输入
    hashtag, button, model_path = output_data
    stock_data = load_stock_data()
    model = load_prediction_model()
    print(f"Attempting to load model from path: {model_path}")
    if button:
        # 加载选择的预测模型
        if model_path  == 'RNN':
            model = load_model(r'.\Stock_History_Day_K-Line\Model\apple_rnn_model.h5')
            st.write("RNN模型预测")
            print(f"Attempting to load model from path: {model}")
        elif model_path  == 'LSTM':
            model = load_model('.\\Stock_History_Day_K-Line\\Model\\apple_lstm_model.h5')
            st.write("LSTM模型预测")
            print(f"Attempting to load model from path: {model}")
        else:
            st.error("未知模型名称")



    X_train_set = stock_data[['收盘']].values.astype(float)
    scaler = MinMaxScaler()
    X_train_set = scaler.fit_transform(X_train_set)

    X_test_set = load_stock_data().iloc[:,4:5].values

    X_train, Y_train = create_dataset(X_train_set, look_back, scaler)
    _, Y_test = create_dataset(X_test_set, look_back, scaler)

    X_test_s = scaler.transform(X_test_set)
    X_test,_ = create_dataset(X_test_s, look_back, scaler)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_test_pred = model.predict(X_test)

    #  将预测值转换回股价
    X_test_pred_price = scaler.inverse_transform(X_test_pred)

  
    # 创建一个新的Matplotlib图表
    fig, ax = plt.subplots(figsize=(16, 6))

    # 绘制实际股价和预测股票价格
    ax.plot(Y_test, color="red", label="val")
    ax.plot(X_test_pred_price, color="blue", label="prediction")
    ax.set_title("APPLE Stock Price Prediction")
    ax.set_ylabel("APPLE Time Price")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    
    
if __name__ == "__main__":
    main()