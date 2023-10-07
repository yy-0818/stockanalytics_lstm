import json
import logging

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
# import tensorflow_probability as tfp
from PIL import Image



def main(df):
    # st.title('基于 LSTM 模型股票分析与预测📈')
    st.markdown("<h1 style='text-align: center; color: black;'>基于 LSTM 模型股票分析与预测📈</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>Stock Market Analysis + Prediction using LSTM📈</h4>", unsafe_allow_html=True)

    st.subheader('by YuanKai ([]())')

    st.markdown(
    """
    
    <br><br/>
    Galaxy Zoo DECaLS includes deep learning classifications for all galaxies. 

    Our model learns from volunteers and predicts posteriors for every Galaxy Zoo question.

    Explore the predictions using the filters on the left. Do you agree with the model?

    To read more about how the model works, click below.

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
    pass

st.set_page_config(
    layout="wide",
    page_title='GZ DECaLS',
    page_icon='gz_icon.jpeg'
)

@st.cache_data 
def load_data():
    # df_locs = ['decals_{}.csv'.format(n) for n in range(4)]
    # dfs = [pd.read_csv(df_loc) for df_loc in df_locs]
    # return pd.concat(dfs)
    pass


if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)

    df = load_data()

    main(df)