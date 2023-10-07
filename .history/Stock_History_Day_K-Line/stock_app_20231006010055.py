import json
import logging

import os
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib as mpl
import tensorflow as tf
# import tensorflow_probability as tfp
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling

from PIL import Image

# mpl.use('TkAgg')
sns.set(style='whitegrid', font='SimHei')
plt.switch_backend('agg')  # 
st.set_page_config(
    layout="wide",
    page_title='Real-Time Stock Price Prediction',
    page_icon = '📋',
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

def main():
    # st.title('基于 LSTM 模型股票分析与预测📈')
    st.markdown("<h1 style='text-align: center; color: black;'>基于 LSTM 模型股票分析与预测📈</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>Stock Market Analysis + Prediction using LSTM📈</h4>", unsafe_allow_html=True)

    st.subheader('by YuanKai ([@ivan](https://blog.ivanlife.cn/))')

    st.markdown(
    """
    
    <br><br/>
    近年来，股票预测还处于一个很热门的阶段，因为股票市场的波动十分巨大，随时可能因为一些新的政策或者其他原因，进行大幅度的波动，导致自然人股民很难对股票进行投资盈利。

    因此本文想利用现有的模型与算法，对股票价格进行预测，从而使自然人股民可以自己对股票进行预测。

    理论上，股票价格是可以预测的，但是影响股票价格的因素有很多，而且目前为止，它们对股票的影响还不能清晰定义。这是因为股票预测是高度非线性的，这就要预测模型要能够处理非线性问题，并且，股票具有时间序列的特性，因此适合用循环神经网络，对股票进行预测。
    
    虽然循环神经网络（RNN），允许信息的持久化，然而，一般的RNN模型对具备长记忆性的时间序列数据刻画能力较弱，在时间序列过长的时候，因为存在梯度消散和梯度爆炸现象RNN训练变得非常困难。Hochreiter 和 Schmidhuber 提出的长短期记忆（ Long Short-Term Memory，LSTM）模型在RNN结构的基础上进行了改造，从而解决了RNN模型无法刻画时间序列长记忆性的问题。

    """
    , unsafe_allow_html=True)
    should_tell_me_more = st.button('Tell me more')
    if should_tell_me_more:
        tell_me_more()
        st.markdown('---')
    else:
        st.markdown('---')
        interactive_galaxies()




if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)

    # df = load_data()

    main()