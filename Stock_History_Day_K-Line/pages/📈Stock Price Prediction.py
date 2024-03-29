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
from datetime import datetime, timedelta

# plt.switch_backend('agg')  #  切换agg后端渲染
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


#  加载数据
stock_data = load_data()
moutai_stock = stock_data[0]
aapl_stock = stock_data[1]
tencent_stock = stock_data[2]
#  加载模型数据
models = load_models()
rnn_model = models[0]
lstm_model = models[1]


#  取出若干天前股价来建立特征和标签数据集
def create_dataset(ds, look_back=1, scaler=None):
    X_data, Y_data = [],[]
    for i in range(len(ds)-look_back):
        X_data.append(ds[i:(i+look_back), 0])
        Y_data.append(ds[i+look_back, 0])
    return np.array(X_data), np.array(Y_data)
look_back = 60

def predict_future_prices(model, last_data_scaled, look_back, scaler, days_to_predict):
    # 获取当前日期
    last_date = datetime.now().date()
    # 初始化预测列表和日期列表
    future_prices_scaled = []
    future_dates = []
    current_batch = last_data_scaled.reshape((1, look_back, 1))

    # 逐天预测未来的股价
    for i in range(days_to_predict):
        # 使用当前批次数据进行预测
        future_price_scaled = model.predict(current_batch)[0]
        # 将预测结果添加到列表
        future_prices_scaled.append(future_price_scaled)
        # 更新批次数据，将新预测结果添加到批次数据的末尾，并移除最早的数据
        current_batch = np.append(current_batch[:, 1:, :], [[future_price_scaled]], axis=1)
        # 计算未来的日期并添加到日期列表
        future_dates.append((last_date + timedelta(days=i+1)).strftime('%Y-%m-%d'))

    # 将预测结果的缩放值转换回原始股价范围
    future_prices = scaler.inverse_transform(np.array(future_prices_scaled).reshape(-1, 1))
    return future_dates, future_prices.flatten().tolist()


def upload_stock_data():
    uploaded_file = st.sidebar.file_uploader("上传 CSV 文件进行预测分析", type="csv")
    if uploaded_file is not None:
        try:
            #  读取上传的 CSV 文件
            uploaded_data = pd.read_csv(uploaded_file)
            #  列名映射字典
            column_mapping = {
                '股票名称': 'Stock Name',
                '股票代码': 'Stock Code',
                '日期': 'Date',
                '开盘': 'Open',
                '收盘': 'Close',
                '最高': 'High',
                '最低': 'Low',
                '成交量': 'Volume',
                '成交额': 'Turnover',
                '振幅': 'Amplitude',
                '涨跌幅': 'Change Percent',
                '涨跌额': 'Change Amount',
                '换手率': 'Turnover Rate'
            }
            #  验证 CSV 文件是否包含所有必要的列
            if not set(column_mapping.keys()).issubset(uploaded_data.columns):
                st.sidebar.error("上传的 CSV 不包含必需的列.")
                return None
            #  重命名列
            uploaded_data.rename(columns=column_mapping, inplace=True)

            #  类型转换（例如日期列）
            uploaded_data['Date'] = pd.to_datetime(uploaded_data['Date'])

            #  数据清洗
            # ...
            return uploaded_data

        except Exception as e:
            st.sidebar.error(f"处理文件时发生错误: {e}")
            return None

    return None

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
        st.markdown('# 设置参数📁')
        st.write('User input parameters below ⬇️')

        uploaded_data = upload_stock_data()
        if uploaded_data is not None:
            stock_data['uploaded'] = uploaded_data
            stock_df = 'uploaded'  #  用于标识上传的数据集
        else:
            #  在侧边栏中创建选择框
            stock_df = st.sidebar.selectbox('选择数据集', list(stock_data.keys()))
            selected_stock_df = stock_data[stock_df]
        
        stock_model_n = st.sidebar.selectbox('选择模型', list(stock_model.keys()))
    
    selected_stock_df = stock_data[stock_df]
    selected_stock_model = stock_model[stock_model_n]
    stock_name = selected_stock_df['Stock Name'].iloc[0] 
    st.title('{}股票数据关联图'.format(stock_name))
    #  stock_data
    if stock_df in stock_data:
        st.title('')
        see_data = st.expander('查看原始数据 \ View the raw data 👉')
        with see_data:
            st.dataframe(data=selected_stock_df.reset_index(drop=True))


    data_source = selected_stock_df[['Open', 'Close']]
    scatter_chart = {
        "title": {"text": f"{stock_name}股票：开盘价和收盘价之间的关系"},
        "xAxis": {"type": "value", "name": "开盘价"},
        "yAxis": {"type": "value", "name": "收盘价"},
        "series": [
            {
                "type": "scatter",
                "data": data_source.values.tolist(),
                "label": {"show": False},
            }
        ],
        "tooltip": {"trigger": "item", "formatter": "{c}"},
    }
    #  在Streamlit中显示ECharts图表
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
        #  确保日期列是 datetime 类型
        selected_stock_df['Date'] = pd.to_datetime(selected_stock_df['Date'])
        #  创建ECharts图表
        echarts_config = {
            "animationDuration": 10000,
            "title": {"text": f"{stock_name}：股价预测"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["实际", "预测"]}, 
            "toolbox": {
                "feature": {
                    "dataZoom": {
                        "yAxisIndex": "none"
                    },
                    "restore": {},
                    "saveAsImage": {}
                }
            },
            "dataZoom": [
                {
                    "type": "inside",
                    "start": 50,
                    "end": 100
                },
                {
                    "type": "slider",
                    "start": 50,
                    "end": 100,
                    "handleSize": "80%",
                    "handleStyle": {
                        "color": "#fff",
                        "shadowBlur": 3,
                        "shadowColor": "rgba(0, 0, 0, 0.6)",
                        "shadowOffsetX": 2,
                        "shadowOffsetY": 2
                    }
                }
            ],
            "xAxis": {
                "type": "category",
                "data": selected_stock_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d')).tolist(),
                "nameLocation": "middle"
            },
            "yAxis": {"name": "股价"},
            "grid": {"right": 140},
            "series": [
                {
                    "type": "line",
                    "data": Y_test.flatten().tolist(),
                    "name": "实际",
                    "showSymbol": True,
                    "itemStyle": {"color": "#ff4d4f"},
                    "emphasis": {"focus": "series"},
                },
                {
                    "type": "line",
                    "data": X_test_pred_price.flatten().tolist(),
                    "name": "预测",
                    "showSymbol": True,
                    "lineStyle": {"type": "dashed"},
                    "itemStyle": {"color": "#1890ff"},
                    "emphasis": {"focus": "series"},
                },
            ],
        }
        st_echarts(echarts_config, height="400px")
        # 选择最后look_back天的数据作为预测的输入
        last_data_scaled = X_train_set[-look_back:]
        # 设定预测的未来天数
        days_to_predict = 60
        # 获取未来股价预测
        future_dates, future_prices = predict_future_prices(selected_stock_model, last_data_scaled, look_back, scaler, days_to_predict)
        # 创建未来股价预测图表配置
        future_echarts_config = {
            "title": {"text": f"{stock_name}：未来股价预测"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["预测"]},
            "xAxis": {
                "type": "category",
                "data": future_dates,
            },
            "yAxis": {"name": "股价"},
            "series": [
                {
                    "type": "line",
                    "data": future_prices,
                    "name": "预测",
                    "showSymbol": True,
                    "lineStyle": {"type": "dashed"},
                    "itemStyle": {"color": "#1890ff"},
                },
            ],
        }
        st_echarts(future_echarts_config, height="400px")

    else:
        st.sidebar.write('未知模型:', stock_model_n)

    data_point = st.sidebar.slider('选择数据点', min_value=0, max_value=len(Y_test)-1)
    # st.sidebar.write(f'日期: {selected_stock_df.iloc[data_point]["Date"]}')
    # st.sidebar.write(f'开盘价: {selected_stock_df.iloc[data_point]["Open"]}')
    # st.sidebar.write(f'收盘价: {selected_stock_df.iloc[data_point]["Close"]}')
    # st.sidebar.write(f'最高价: {selected_stock_df.iloc[data_point]["High"]}')
    # st.sidebar.write(f'最低价: {selected_stock_df.iloc[data_point]["Low"]}')
    selected_date = selected_stock_df.iloc[data_point + look_back]["Date"]
    actual_price = selected_stock_df.iloc[data_point + look_back]["Close"]
    predicted_price = X_test_pred_price[data_point][0]

    st.sidebar.write(f'日期: {selected_date}')
    st.sidebar.write(f'实际收盘价: {actual_price}')
    st.sidebar.write(f'预测收盘价: {predicted_price}')
    st.sidebar.info('该项目可以帮助你理解LSTM')
    st.divider()
    st.sidebar.caption('<p style="text-align:center">made with ❤️ by Yuan</p>', unsafe_allow_html=True)


if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)
    main()