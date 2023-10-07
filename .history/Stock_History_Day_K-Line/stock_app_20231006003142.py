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

# 取出若干天前股价来建立特征和标签数据集
def create_dataset(ds, look_back=1, scaler=None):
    X_data, Y_data = [],[]
    for i in range(len(ds)-look_back):
        X_data.append(ds[i:(i+look_back), 0])
        Y_data.append(ds[i+look_back, 0])
    return np.array(X_data), np.array(Y_data)
look_back = 60

def interactive_galaxies(df):
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
        # st.image('https://static.ivanlife.cn/img/wallhaven-9mo7kw_1920x1080.png')
        st.markdown('# 设置参数📁')
        st.write('User input parameters below ⬇️')
        
        # 在侧边栏中创建选择框
        stock_df = st.sidebar.selectbox('选择数据集', list(stock_data.keys()))


        stock_model_n = st.sidebar.selectbox('选择模型', list(stock_model.keys()))

        st.info('该项目可以帮助你理解LSTM')

    selected_stock_df = stock_data[stock_df]
    selected_stock_model = stock_model[stock_model_n]
    # stock_data
    if stock_df in stock_data:
        see_data = st.expander('查看原始数据 \ View the raw data 👉')
        with see_data:
            st.dataframe(data=selected_stock_df.reset_index(drop=True))
    else:
        st.sidebar.write('未知数据集:', stock_df)

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

        # 创建一个新的Matplotlib图表
        fig, ax = plt.subplots(figsize=(16, 6))

        # 绘制实际股价和预测股票价格
        ax.plot(Y_test, color="darkorange", label="val")
        ax.plot(X_test_pred_price, color="deepskyblue", label="prediction")
        ax.set_title("APPLE Stock Price Prediction")
        ax.set_ylabel("APPLE Time Price")
        ax.legend()
        ax.grid(True)
        plt.switch_backend('tkagg')
        st.pyplot(fig)
    else:
        st.sidebar.write('未知模型:', stock_model_n)

    stock_name = selected_stock_df['Stock Name'].iloc[0]
    # 创建Streamlit应用程序
    st.title('{}股票数据关联图'.format(stock_name))
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(selected_stock_df['Open'], selected_stock_df['Close'])
    ax.set_xlabel(f'{stock_name}')
    ax.set_ylabel(f'{stock_name}')
    plt.switch_backend('tkagg')
    st.pyplot(fig)
    st.markdown("")

    # profile = pandas_profiling.ProfileReport(selected_stock_df)
    # st_profile_report(profile)


if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)

    df = load_data()

    main(df)