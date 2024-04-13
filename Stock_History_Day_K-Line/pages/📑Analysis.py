import logging
import os
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_echarts import st_echarts

st.set_page_config(
    layout="wide",
    page_title='Real-Time Stock Price Prediction',
    page_icon = '💹',
)

types = ["贵州茅台","苹果","腾讯"]
label_stock_dict_teams = {"Stock Name","Stock Code","Date","Open","Close","High","Low","Volume","Turnover,Amplitude","Change Percent","Change Amount","Turnover Rate"}

@st.cache_data 
def load_data():
    data_locs = [os.getcwd() + '\Stock_History_Day_K-Line\Data\stock_{}.csv'.format(n) for n in range(1, 5)]
    data = [pd.read_csv(data_loc) for data_loc in data_locs]
    return data


#  加载数据
stock_data = load_data()
moutai_stock = stock_data[0]
aapl_stock = stock_data[1]
tencent_stock = stock_data[2]
tcl_stock = stock_data[3]


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
            return uploaded_data

        except Exception as e:
            st.sidebar.error(f"处理文件时发生错误: {e}")
            return None
    return None


def main():
    stock_data = {
        '贵州茅台': moutai_stock,
        '苹果': aapl_stock,
        '腾讯控股': tencent_stock,
        'TCL科技': tcl_stock,
    }

    with st.sidebar:
        st.markdown('# 设置参数📁')
        st.write('User input parameters below ⬇️')

        uploaded_data = upload_stock_data()
        if uploaded_data is not None:
            st.success("文件上传成功.")
            stock_data['uploaded'] = uploaded_data
            stock_df = 'uploaded'  #  用于标识上传的数据集
        else:
            #  在侧边栏中创建选择框
            stock_df = st.sidebar.selectbox('选择数据集', list(stock_data.keys()))
            selected_stock_df = stock_data[stock_df]
        
    selected_stock_df = stock_data[stock_df]
    stock_name = selected_stock_df['Stock Name'].iloc[0] 
    st.title('{}股票数据关联图'.format(stock_name))
    if stock_df in stock_data:
        st.title('')
        see_data = st.expander('查看原始数据 \ View the raw data 👉')
        with see_data:
            st.dataframe(data=selected_stock_df.reset_index(drop=True))

    if 'Date' in selected_stock_df.columns:
        selected_stock_df['Date'] = pd.to_datetime(selected_stock_df['Date'])
    else:
        if not isinstance(selected_stock_df.index, pd.DatetimeIndex):
            selected_stock_df.index = pd.to_datetime(selected_stock_df.index)

    data_source = selected_stock_df.reset_index()[['Date', 'Close']]
    dates = data_source['Date'].dt.strftime('%Y-%m-%d').tolist()
    closing_prices = data_source['Close'].tolist()
    options = {
        "title": {
            "text": f'{stock_name} - 历史收盘价',
        },
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
        "tooltip": {
            "trigger": 'axis'
        },
        "xAxis": {
            "type": 'category',
            "data": dates,
        },
        "yAxis": {
            "type": 'value'
        },
        "series": [{
            "data": closing_prices,
            "type": 'line',
            "smooth": True
        }]
    }
    st_echarts(options=options, height="500px")

    data_source2 = selected_stock_df.reset_index()[['Date', 'Volume']]
    dates = data_source2['Date'].dt.strftime('%Y-%m-%d').tolist()
    volumes = data_source2['Volume'].tolist()
    options = {
        "title": {
            "text": f"{stock_name} - 历史成交量",
        },
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
        "tooltip": {
            "trigger": 'axis'
        },
        "xAxis": {
            "type": 'category',
            "data": dates
        },
        "yAxis": {
            "type": 'value',
            "name": 'Volume'
        },
        "series": [{
            "data": volumes,
            "type": 'line',
            "smooth": True
        }]
    }
    st_echarts(options=options, height="500px")

    data_source3 = selected_stock_df.reset_index()[['Date', 'Close']]
    dates = data_source3['Date'].dt.strftime('%Y-%m-%d').tolist()
    ma_day = [10, 20, 50]
    for ma in ma_day:
        column_name = f"MA for {ma} days"
        selected_stock_df[column_name] = selected_stock_df['Close'].rolling(window=ma, min_periods=1).mean()
    if all(f"MA for {ma} days" in selected_stock_df.columns for ma in ma_day):
        data_source4 = selected_stock_df.reset_index()[['MA for 10 days', 'MA for 20 days','MA for 50 days']]
        close_prices = data_source3['Close'].tolist()
        ma_10 = data_source4['MA for 10 days'].tolist()
        ma_20 = data_source4['MA for 20 days'].tolist()
        ma_50 = data_source4['MA for 50 days'].tolist()
        options = {
        "title": {
            "text": f"{stock_name} - 调整收盘价和移动平均线"
        },
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
        "tooltip": {
            "trigger": 'axis'
        },
        "legend": {
            "data": ['Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']
        },
        "xAxis": {
            "type": 'category',
            "boundaryGap": False,
            "data": dates
        },
        "yAxis": {
            "type": 'value',
            "name": 'Price'
        },
        "series": [
            {
                "name": 'Close',
                "type": 'line',
                "data": close_prices,
                "smooth": True
            },
            {
                "name": 'MA for 10 days',
                "type": 'line',
                "data": ma_10,
                "smooth": True
            },
            {
                "name": 'MA for 20 days',
                "type": 'line',
                "data": ma_20,
                "smooth": True
            },
            {
                "name": 'MA for 50 days',
                "type": 'line',
                "data": ma_50,
                "smooth": True
            }
        ]
    }
    st_echarts(options=options,height='500px')
   
if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)
    main()
