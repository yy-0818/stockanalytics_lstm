import logging
import os
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_echarts import st_echarts
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler



sns.set(style='whitegrid', font='SimHei')
plt.switch_backend('agg')  # 
st.set_page_config(
    layout="wide",
    page_title='Real-Time Stock Price Prediction',
    page_icon = '💹',
)


types = ["贵州茅台","苹果","腾讯"]
label_stock_dict_teams = {"Stock Name","Stock Code","Date","Open","Close","High","Low","Volume","Turnover,Amplitude","Change Percent","Change Amount","Turnover Rate"}



@st.cache_data 
def load_models():
    model_paths = [os.getcwd() + '\Stock_History_Day_K-Line\Model\model_{}.h5'.format(n) for n in range(1, 3)]
    models = [load_model(model_path) for model_path in model_paths]
    return models

@st.cache_data 
def load_data():
    data_locs = [os.getcwd() + '\Stock_History_Day_K-Line\Data\stock_{}.csv'.format(n) for n in range(1, 4)]
    data = [pd.read_csv(data_loc) for data_loc in data_locs]
    return data


# 加载数据
stock_data = load_data()
moutai_stock = stock_data[0]
aapl_stock = stock_data[1]
tencent_stock = stock_data[2]
# 加载模型数据
models = load_models()
rnn_model = models[0]
lstm_model = models[1]


# 取出若干天前股价来建立特征和标签数据集
def create_dataset(ds, look_back=1, scaler=None):
    X_data, Y_data = [],[]
    for i in range(len(ds)-look_back):
        X_data.append(ds[i:(i+look_back), 0])
        Y_data.append(ds[i+look_back, 0])
    return np.array(X_data), np.array(Y_data)
look_back = 60


def main():
    stock_data = {
        '贵州茅台': moutai_stock,
        '苹果': aapl_stock,
        '腾讯控股': tencent_stock
    }
    stock_model = {
        'RNN': rnn_model,
        'LSTM': lstm_model,
    }
    with st.sidebar:
        st.title('')
        st.markdown('# 设置参数📁')
        st.write('User input parameters below ⬇️')
        
        # 在侧边栏中创建选择框
        stock_df = st.sidebar.selectbox('选择数据集', list(stock_data.keys()))


        stock_model_n = st.sidebar.selectbox('选择模型', list(stock_model.keys()))

        # st.info('该项目可以帮助你理解LSTM')

    selected_stock_df = stock_data[stock_df]
    selected_stock_model = stock_model[stock_model_n]
    stock_name = selected_stock_df['Stock Name'].iloc[0] 
    # stock_data
    if stock_df in stock_data:
        st.title('')
        see_data = st.expander('查看原始数据 \ View the raw data 👉')
        with see_data:
            st.dataframe(data=selected_stock_df.reset_index(drop=True))
    else:
        st.sidebar.write('未知数据集:', stock_df)


    st.title('{}股票数据关联图'.format(stock_name))
    data_source = selected_stock_df[['Open', 'Close']]
    scatter_chart = {
        "xAxis": {"type": "value", "name": "Open"},
        "yAxis": {"type": "value", "name": "Close"},
        "series": [
            {
                "type": "scatter",
                "data": data_source.values.tolist(),
                "label": {"show": False},
            }
        ],
        "formatter": "{a} <br/>X轴数据: {b} <br/>Y轴数据: {c}",
    }
    # 在Streamlit中显示ECharts图表
    st_echarts(
        options=scatter_chart,
        height="400px",
        key="scatter_chart", 
    )


    # stock_model
    if stock_model_n in stock_model:
        X_train_set = selected_stock_df[['Close']].values.astype(float)
        scaler = MinMaxScaler()
        X_train_set = scaler.fit_transform(X_train_set)
        X_test_set = selected_stock_df.iloc[:,4:5].values

        _, Y_test = create_dataset(X_test_set, look_back, scaler)

        X_test_s = scaler.transform(X_test_set)
        X_test,_ = create_dataset(X_test_s, look_back, scaler)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        X_test_pred = selected_stock_model.predict(X_test)
        #  将预测值转换回股价
        X_test_pred_price = scaler.inverse_transform(X_test_pred)

        # 创建ECharts图表
        echarts_config = {
            "animationDuration": 10000,
            "title": {"text": f"{stock_name} Predicted Stock Price"},
            # "tooltip": {"order": "valueDesc", "trigger": "axis"},
            "tooltip": {"trigger": "axis", "formatter": "{a}: {c}"},
            "xAxis": {"type": "category", "nameLocation": "middle"},
            "yAxis": {"name": "Stock Price"},
            "grid": {"right": 140},
            "series": [
                {
                    "type": "line",
                    "data": Y_test.flatten().tolist(),
                    "name": "Actual",
                    "showSymbol": False,
                    "emphasis": {"focus": "series"},
                },
                {
                    "type": "line",
                    "data": X_test_pred_price.flatten().tolist(),
                    "name": "Prediction",
                    "showSymbol": False,
                    "emphasis": {"focus": "series"},
                },
            ],
        }
        # 在Streamlit中显示ECharts图表
        st_echarts(echarts_config, height="400px")

    else:
        st.sidebar.write('未知模型:', stock_model_n)



    data_point = st.sidebar.slider('选择数据点', min_value=0, max_value=len(Y_test)-1)
    st.sidebar.write(f'日期: {selected_stock_df.iloc[data_point]["Date"]}')
    st.sidebar.write(f'开盘价: {selected_stock_df.iloc[data_point]["Open"]}')
    st.sidebar.write(f'收盘价: {selected_stock_df.iloc[data_point]["Close"]}')
    st.sidebar.write(f'最高价: {selected_stock_df.iloc[data_point]["High"]}')
    st.sidebar.write(f'最低价: {selected_stock_df.iloc[data_point]["Low"]}')


if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)

    main()