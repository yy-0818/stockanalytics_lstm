import logging
import os
import pandas as pd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling


st.set_page_config(
    layout="wide",
    page_title='Real-Time Stock Price Prediction',
    page_icon='💹',
)

# 使用 Streamlit 的缓存机制来加速数据加载
@st.cache_data 
def load_data():
    data_locs = [os.path.join(os.getcwd(), 'Stock_History_Day_K-Line', 'Data', f'stock_{n}.csv') for n in range(1, 4)]
    data = [pd.read_csv(data_loc) for data_loc in data_locs]
    return data


# 加载数据
stock_data = load_data()
moutai_stock = stock_data[0]
aapl_stock = stock_data[1]
tencent_stock = stock_data[2]

def add_logo():
     st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://s2.loli.net/2024/03/28/s3i6mgKr5vd9ADR.png);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Stock Market Analysis";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def upload_and_profile_data():
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # 读取上传的 CSV 文件
        df = pd.read_csv(uploaded_file)
        profile = pandas_profiling.ProfileReport(df)
        st_profile_report(profile)
    else:
        st.sidebar.write("上传 CSV 文件以生成配置文件报告.")

def main():
    stock_data = {
        '贵州茅台': moutai_stock,
        '苹果': aapl_stock,
        '腾讯控股': tencent_stock
    }
    with st.sidebar:
        st.title('实时股票价格预测')
        st.markdown('## 设置参数 📁')
        st.write('User input parameters below ⬇️')

        # 在侧边栏中创建选择框
        stock_df = st.sidebar.selectbox('选择数据集', list(stock_data.keys()) + ["上传CSV文件"])
        
        # st.info('该项目可以帮助你理解LSTM')

    # 如果用户选择了上传 CSV 文件，则调用上传功能
    if stock_df == "上传CSV文件":
        upload_and_profile_data()
    else:
        selected_stock_df = stock_data[stock_df]

        # stock_data
        if stock_df in stock_data:
            see_data = st.expander(f'查看原始数据 \ View the raw data for {stock_df} 👉')
            with see_data:
                st.dataframe(selected_stock_df.reset_index(drop=True))
            
            profile = pandas_profiling.ProfileReport(selected_stock_df)
            st_profile_report(profile)
        else:
            st.sidebar.write('未知数据集:', stock_df)

if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)
    add_logo()
    main()