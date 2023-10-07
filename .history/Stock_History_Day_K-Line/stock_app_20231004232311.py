import json
import logging

import os
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
# import tensorflow_probability as tfp
from keras.models import load_model
from streamlit_pandas_profiling import st_profile_report
from PIL import Image

sns.set(style='whitegrid', font='SimHei')
st.set_page_config(
    layout="wide",
    page_title='Real-Time Stock Price Prediction',
    page_icon = '📋',
)

@st.cache_data 
def load_models():
    model_paths = [os.getcwd() + '\Stock_History_Day_K-Line\Model\model_{}.h5'.format(n) for n in range(1, 3)]
    models = [load_model(model_path) for model_path in model_paths]
    return models

@st.cache_data 
def load_data():
    df_locs = [os.getcwd() + '\Stock_History_Day_K-Line\Data\stock_{}.csv'.format(n) for n in range(1, 4)]
    dfs = [pd.read_csv(df_loc) for df_loc in df_locs]
    return dfs

# 加载数据
stock_data = load_data()
moutai_stock = stock_data[0]
aapl_stock = stock_data[1]
tencent_stock = stock_data[2]
# 加载模型数据
models = load_models()
rnn_model = models[0]
lstm_model = models[1]

def main(df):
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
        interactive_galaxies(df)



def tell_me_more():
    st.title('Building the Model')

    st.button('Back to galaxies')  # will change state and hence trigger rerun and hence reset should_tell_me_more

    st.markdown("""
    We require a model which can:
    - Learn efficiently from volunteer responses of varying (i.e. heteroskedastic) uncertainty
    - Predict posteriors for those responses on new galaxies, for every question

    In [previous work](https://arxiv.org/abs/1905.07424), we modelled volunteer responses as being binomially distributed and trained our model to make maximum likelihood estimates using the loss function:
    """)

    st.latex(
    """
    \mathcal{L} = k \log f^w(x) + (N-k) \log(1-f^w(x))
    """
    )
    st.markdown(
    r"""
    where, for some target question, k is the number of responses (successes) of some target answer, N is the total number of responses (trials) to all answers, and $f^w(x) = \hat{\rho}$ is the predicted probability of a volunteer giving that answer.
    """
    )
   
    st.markdown(
    r"""
    This binomial assumption, while broadly successful, broke down for galaxies with vote fractions k/N close to 0 or 1, where the Binomial likelihood is extremely sensitive to $f^w(x)$, and for galaxies where the question asked was not appropriate (e.g. predict if a featureless galaxy has a bar). 
    
    Instead, in our latest work, the model predicts a distribution 
    """)

    st.latex(r"""
    f^w(x) = p(\rho|f^w(x))
    """)
    
    st.markdown(r"""
    and $\rho$ is then drawn from that distribution.
    
    For binary questions, one could use the Beta distribution (being flexible and defined on the unit interval), and predict the Beta distribution parameters $f^w(x) = (\hat{\alpha}, \hat{\beta})$ by minimising

    """)

    st.latex(
    r"""
        \mathcal{L} = \int Bin(k|\rho, N) Beta(\rho|\alpha, \beta) d\alpha d\beta    
    """
    )
    st.markdown(r"""

    where the Binomial and Beta distributions are conjugate and hence this integral can be evaluated analytically.

    In practice, we would like to predict the responses to questions with more than two answers, and hence we replace each distribution with its multivariate counterpart; Beta($\rho|\alpha, \beta$) with Dirichlet($\vec{\rho}|\vec{\alpha})$, and Binomial($k|\rho, N$) with Multinomial($\vec{k}|\vec{\rho}, N$).
    """)
    
    st.latex(r"""
     \mathcal{L}_q = \int Multi(\vec{k}|\vec{\rho}, N) Dirichlet(\vec{\rho}| \vec{\alpha}) d\vec{\alpha}
    """)
    
    st.markdown(r"""
    where $\vec{k}, \vec{\rho}$ and $\vec{\alpha}$ are now all vectors with one element per answer. 

    Using this loss function, our model can predict posteriors with excellent calibration.

    For the final GZ DECaLS predictions, I actually use an ensemble of models, and apply active learning - picking the galaxies where the models confidently disagree - to choose the most informative galaxies to label with Galaxy Zoo. Check out the paper for more.
    
    """)

    st.button('Back to galaxies', key='back_again')  # will change state and hence trigger rerun and hence reset should_tell_me_more



def interactive_galaxies(df):
    # 创建一个字典，将数据集名称映射到数据框
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
        st.image('https://static.ivanlife.cn/img/wallhaven-9mo7kw_1920x1080.png')
        st.markdown('# 设置参数📁')
        st.write('User input parameters below ⬇️')
        
        # 在侧边栏中创建选择框
        stock_df = st.sidebar.selectbox('选择数据集', list(stock_data.keys()))
        # st.sidebar.write('当前数据集为：', stock_df)
        stock_model = st.sidebar.selectbox('选择模型', list(stock_model.keys()))
        st.info('该项目可以帮助你理解LSTM')

    selected_stock_df = stock_data[stock_df]
    # 如果用户选择了特定的数据集，则显示对应的原始数据
    if stock_df in stock_data:
        see_data = st.expander('查看原始数据 \ View the raw data 👉')
        with see_data:
            st.dataframe(data=selected_stock_df.reset_index(drop=True))
    else:
        st.sidebar.write('未知数据集:', stock_df)

 
    stock_name = selected_stock_df['股票名称'].iloc[0]
    # 创建Streamlit应用程序
    st.title('{}股票数据关联图'.format(stock_name))
    plt.figure(figsize=(8, 6))


    jointplot = sns.jointplot(x='开盘', y='收盘', data=selected_stock_df, kind='scatter') 
    jointplot.set_axis_labels(f'{stock_name}', f'{stock_name}', fontsize=12)

    st.pyplot()
    st.markdown("")
    st_profile_report(selected_stock_df)


if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)

    df = load_data()

    main(df)