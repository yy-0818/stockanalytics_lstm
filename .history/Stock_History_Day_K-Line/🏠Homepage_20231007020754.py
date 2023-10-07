import json
import logging

import os
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from PIL import Image


sns.set(style='whitegrid', font='SimHei')
plt.switch_backend('agg')  # 
st.set_page_config(
    layout="wide",
    page_title='Real-Time Stock Price Prediction',
    page_icon = '💹',
)


types = ["贵州茅台","苹果","腾讯"]
label_stock_dict_teams = {"Stock Name","Stock Code","Date","Open","Close","High","Low","Volume","Turnover,Amplitude","Change Percent","Change Amount","Turnover Rate"}



def main():

    with st.sidebar:
        st.title('')
        st.info('该项目可以帮助你理解LSTM')
    # st.title('基于 LSTM 模型股票分析与预测📈')
    st.markdown("<h1 style='text-align: center; color: black;'>基于 LSTM 模型股票分析与预测📊</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>Stock Market Analysis + Prediction using LSTM📊</h4>", unsafe_allow_html=True)

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
        # upload_data()



def tell_me_more():
    st.markdown("<h2>构建模型</h2>",unsafe_allow_html=True)
    st.markdown("<h4>Building the Model</h4>", unsafe_allow_html=True)

    st.button('Back Homepage')  # will change state and hence trigger rerun and hence reset should_tell_me_more

    st.markdown("""
        1. RNN 神经网络模型

        循环神经网络是一种对序列数据进行建模的神经网络，它像一个循环动态系统，在该结构中当前的 输出会流入下一步的输入中，为下一次输出做出贡献。其主要形式是该结构有个循环结构会保留前一次 循环的输出结果并作为下一次循环输人的一部分输入。
        
      """)


    st.image(r'Stock_History_Day_K-Line\images\rnn.jpg', width=600, caption='循环神经网络结构图')
    st.markdown("""
        RNN 模型是 LSTM 模型的底层结构，函数形式为：
    """)
    st.latex(
        """
        h_t = \sigma(W_{hx} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)
        y_t = \sigma(W_{hy} \cdot h_t + b_y)

        """
    )

    st.markdown("""
        2. LSTM 神经网络模型
        LSTM 算法全称为 Long short-term memory，最早由 Sepp Hochreiter 和 Jürgen Schmidhuber 于 1997 年 提出，是一种特定形式的 RNN (Recurrent neural network，循环神经网络)

        在普通 RNN 基础上，在隐藏层各神经单元中增加及一单元，从而使 RNN 具备了长期的记忆功能。
        
        LSTM 神经网络由相互联系的递归子网络，即记忆模块组成，记忆模块主要包括三个门：遗忘门，输入门，输出门，和一个记忆单元。

    """)
    col1, col2 = st.columns(2)
    col1.image(r'Stock_History_Day_K-Line\images\lstm.jpg', width=500, caption='LSTM 神经网络结构图')
    col2.image(r'Stock_History_Day_K-Line\images\lstm_inside.jpg', width=500, caption='LSTM 神经网络内部结构')
    st.latex(
    """
    f_t = \sigma(W_hx \cdot [h_{t-1}, h_t] + b_f)
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
    C_t = \tanh(W_hx \cdot b_C)
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

    st.button('Back Homepage', key='back_again')  # will change state and hence trigger rerun and hence reset should_tell_me_more


def upload_data():
    st.markdown(r"""
    The data is available at [https://github.com/mjuric/gz-decals](https://github.com/mjuric/gz-decals).
    """)

    st.markdown(r"""
    The data is available at [https://github.com/mjuric/gz-decals](https://github.com/mjuric/gz-decals).
    """)

if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)

    # df = load_data()

    main()