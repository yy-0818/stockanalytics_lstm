import pandas as pd
import numpy as np
import streamlit as st
import logging
import os
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm

st.set_page_config(
    layout="wide",
    page_title='Real-Time Stock Price Prediction',
    page_icon = '💹',
)
 
st.title("使用 Pygwalker 自主分析数据集")

# 初始化Streamlit与Pygwalker的通信
init_streamlit_comm()


@st.cache_data 
def load_data():
    data_locs = [os.getcwd() + '\Stock_History_Day_K-Line\Data\stock_{}.csv'.format(n) for n in range(1, 5)]
    data = [pd.read_csv(data_loc) for data_loc in data_locs]
    return data

# 使用st.cache_resource装饰器来缓存渲染器创建函数
@st.cache_resource
def get_pyg_renderer(df) -> "StreamlitRenderer":
    return StreamlitRenderer(df, spec="./gw_config.json", debug=False)


stock_data = load_data()
moutai_stock = stock_data[0]
aapl_stock = stock_data[1]
tencent_stock = stock_data[2]
tcl_stock = stock_data[3]

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
        uploaded_file = st.file_uploader("上传你的数据文件", type="csv")

        if uploaded_file is not None:
            # 如果用户上传了文件，则使用该文件创建渲染器
            df = pd.read_csv(uploaded_file)
            st.success("文件上传成功.")
            stock_data['uploaded'] = df
            stock_df = 'uploaded'  # 标识上传的数据集
        else:
            stock_df = st.selectbox('选择数据集', list(stock_data.keys()))
            df = stock_data[stock_df]

    # 显示选定的股票数据
    # st.title(f'数据集：{stock_df}')
    see_data = st.expander('查看原始数据 \ View the raw data 👉')
    with see_data:
        st.dataframe(df.reset_index(drop=True))

    # 创建渲染器并渲染数据
    renderer = get_pyg_renderer(df)
    renderer.render_explore()
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)
    main()