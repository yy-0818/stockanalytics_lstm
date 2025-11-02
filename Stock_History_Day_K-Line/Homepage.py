import logging
import os
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title='Real-Time Stock Price Prediction',
    page_icon = '💹',
)


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

def main():
    with st.sidebar:
        st.info('该项目可以帮助你理解LSTM([github](https://github.com/yy-0818/stockanalytics_lstm))')
        st.caption('<p style="text-align:center">made with ❤️ by Yuan</p>', unsafe_allow_html=True)

    # st.title('基于 LSTM 模型股票分析与预测📈')
    st.markdown("<h1 style='text-align: center; color: black;'>基于 LSTM 模型股票分析与预测📊</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>Stock Market Analysis + Prediction using LSTM📊</h4>", unsafe_allow_html=True)

    st.subheader('by YuanKai ([@ivan](https://github.com/yy-0818/))')

    st.markdown(
    """
    
    <br><br/>
    &emsp;近年来，股票预测还处于一个很热门的阶段，因为股票市场的波动十分巨大，随时可能因为一些新的政策或者其他原因，进行大幅度的波动，导致自然人股民很难对股票进行投资盈利。

    &emsp;因此本文想利用现有的模型与算法，对股票价格进行预测，从而使自然人股民可以自己对股票进行预测。

    &emsp;理论上，股票价格是可以预测的，但是影响股票价格的因素有很多，而且目前为止，它们对股票的影响还不能清晰定义。这是因为股票预测是高度非线性的，这就要预测模型要能够处理非线性问题，并且，股票具有时间序列的特性，因此适合用循环神经网络，对股票进行预测。
    
    &emsp;虽然循环神经网络（RNN），允许信息的持久化，然而一般的RNN模型对具备长记忆性的时间序列数据刻画能力较弱，在时间序列过长的时候，因为存在梯度消散和梯度爆炸现象RNN训练变得非常困难。Hochreiter 和 Schmidhuber 提出的长短期记忆（ Long Short-Term Memory，LSTM）模型在RNN结构的基础上进行了改造，从而解决了RNN模型无法刻画时间序列长记忆性的问题。

    """
    , unsafe_allow_html=True)

    should_tell_me_more = st.button('Tell me more')
    if should_tell_me_more:
        tell_me_more()
        st.markdown('---')
    else:
        st.markdown('---')
        upload_data()


def tell_me_more():
    st.markdown("<h2>构建模型</h2>",unsafe_allow_html=True)
    st.markdown("<h4>Building the Model</h4>", unsafe_allow_html=True)

    st.button('Back Homepage')

    st.markdown("""
        **1. RNN 神经网络模型**

        &emsp;循环神经网络是一种对序列数据进行建模的神经网络，它像一个循环动态系统，在该结构中当前的输出会流入下一步的输入中，为下一次输出做出贡献。其主要形式是该结构有个循环结构会保留前一次 循环的输出结果并作为下一次循环输人的一部分输入。
        
      """)

    col1, col2,col3 = st.columns(3)
    # col2.image(r'Stock_History_Day_K-Line\images\rnn.png', width=650, caption='循环神经网络结构图')
    col2.image(os.path.join('Stock_History_Day_K-Line', 'images', 'rnn.png'), width=650, caption='循环神经网络结构图')
    st.markdown("""
        &emsp;RNN 模型是 LSTM 模型的底层结构，函数形式为：
    """)
    st.latex(
        r"""
        \boldsymbol{h_t = \sigma(W_{hx} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)}
        """
    )
    st.latex(
        r"""
        \boldsymbol{y_t = \sigma(W_{hy} \cdot h_t + b_y)}
        """
    )

    st.markdown(r"""
        **2. LSTM 神经网络模型**

        &emsp;LSTM 算法全称为 Long short-term memory，最早由 Sepp Hochreiter 和 Jürgen Schmidhuber 于 1997 年 提出，是一种特定形式的 RNN (Recurrent neural network，循环神经网络)。
        从数学上来讲，LSTM是一个高度复合的非线性参数函数，它将一列向量$\boldsymbol{(x_{1},···x_{n})}$通过隐含层$\boldsymbol{(h_{1},···h_{n})}$ 映射到另一组向量$\boldsymbol{(y_{1},···y_{n})}$。
                
        &emsp;LSTM 神经网络由相互联系的递归子网络，即记忆模块组成，记忆模块主要包括三个门：遗忘门，输入门，输出门，和一个记忆单元。
    """)
    col14, col5,col6 = st.columns(3)
    # col5.image(r'Stock_History_Day_K-Line\images\lstm.png', width=650, caption='LSTM 神经网络结构图')
    col5.image(os.path.join('Stock_History_Day_K-Line', 'images', 'lstm.png'), width=650, caption='LSTM 神经网络结构图')
    # st.latex(r"""
    # \begin{align*}
    # f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
    # i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
    # \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\
    # C_t &= f_t \ast C_{t-1} + i_t \ast \tilde{C}_t \\
    # o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
    # h_t &= o_t \ast \tanh(C_t)
    # \end{align*}
    # """)
    st.markdown(r"""
    &emsp;遗忘门（Forget Gate）,在 LSTM 中的第一步是决定我们会从细胞状态中丢弃什么信息。这个决定通过一个称为遗忘门层完成。该门会读取 $\boldsymbol{h_{t-1}}$和 $\boldsymbol{x_t}$ ，输出一个在 0 到 1 之间的数值给每个细胞状态 $\boldsymbol{C_{t-1}}$ 中的数字。1表示“完全保留”，0 表示“完全舍弃”:
    """)
    st.latex(
    r"""
    \boldsymbol{f_t = \sigma(W_hx \cdot [h_{t-1}, h_t] + b_f)}
    """
    )
    st.markdown(r"""
    &emsp;其中 $\boldsymbol{h_{t-1}}$ 表示的是上一个 cell 的输出， $\boldsymbol{C_{t-1}}$ 表示的是当前细胞的输入, $\boldsymbol{σ}$ 表示 sigmod 函数。
    """)
    st.markdown(
    r"""
    &emsp;输入门（Input Gate），下一步是决定让多少新的信息加入到 cell 状态中来。实现这个需要包括两个步骤：首先一个叫做“input gate layer”的 sigmoid 层决定哪些信息需要更新；一个 $\boldsymbol{tanh}$ 层生成一个向量，也就是备
    选的用来更新的内容。在下一步，我们把这两部分联合起来，对 cell 的状态进行一个更新：
    """
    )
    st.latex(r"""
    \boldsymbol{i_t = \sigma(W_hx \cdot [h_{t-1}, h_t] + b_i)}
    """)
    st.latex(r"""
    \boldsymbol{\tilde{C}_t = \tanh(W_{C} \cdot [h_{t-1}, x_t] + b_C)}
    """)
    st.markdown(r"""
    &emsp;这里的 $(\tilde{C}_t)$ 是候选记忆单元（Candidate Memory Cell）,代表可能加入当前细胞状态的新信息,最后我们组合以上信息来更新 cell 状态：
    """)
    # st.latex(r"""
    # C_t = \tanh(W_hx \cdot b_C)
    # """)
    st.latex(r"""
    \boldsymbol{C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t}
    """)
    st.markdown(r"""
    &emsp;输出门（Output Gate）:我们将确定从 cell 状态中输出什么值,此步骤先通过 sigmoid 层确定 cell 状态的哪一部分值会被输出。然后我们将 cell 状态通过 $tanh$ 处理得到一个在值域 **\( -1 \)** 至 **\( 1 \)** 之间的值，并乘以 sigmoid 层的输出，以此决定最终的输出值：
    """)
    st.latex(r"""
    \boldsymbol{o_t = \sigma(W_hx \cdot [h_{t-1}, h_t] + b_o)}   
    """
    )
    st.markdown("""
    &emsp;更新记忆单元（Update Memory Cell）:
    """)
    st.latex(r"""
    \boldsymbol{h_t = o_t \cdot \tanh(C_t)}
    """)
    st.markdown(r"""
    &emsp;其中 $\boldsymbol{h_t}$ 为该时间步的最终输出，$\boldsymbol{o_t}$ 是输出门激活值。
    """)

    st.button('Back Homepage', key='back_again')  # will change state and hence trigger rerun and hence reset should_tell_me_more


def upload_data():
    pass


if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)
    add_logo()
    main()
