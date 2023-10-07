import logging
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling


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


def interactive_galaxies():
    stock_data = {
        '贵州茅台': moutai_stock,
        '苹果': aapl_stock,
        '腾讯控股': tencent_stock
    }
    with st.sidebar:
        st.title('')
        # st.image('https://static.ivanlife.cn/img/wallhaven-9mo7kw_1920x1080.png')
        st.markdown('# 设置参数📁')
        st.write('User input parameters below ⬇️')
        
        # 在侧边栏中创建选择框
        stock_df = st.sidebar.selectbox('选择数据集', list(stock_data.keys()))


        st.info('该项目可以帮助你理解LSTM')

    selected_stock_df = stock_data[stock_df]

    # stock_data
    if stock_df in stock_data:
        see_data = st.expander('查看原始数据 \ View the raw data 👉')
        with see_data:
            st.dataframe(data=selected_stock_df.reset_index(drop=True))
    else:
        st.sidebar.write('未知数据集:', stock_df)


    profile = pandas_profiling.ProfileReport(selected_stock_df)
    st_profile_report(profile)

if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)

    interactive_galaxies()