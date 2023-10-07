import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model

# 加载股票数据
@st.cache_data
def load_stock_data():
    data = pd.read_csv('.\Stock_History_Day_K-Line\苹果.csv')  # 替换为你的股票数据文件
    data['日期'] = pd.to_datetime(data['日期'])
    return data

# 加载预测模型
@st.cache_data
def load_prediction_model():
    model = load_model(r'.\Stock_History_Day_K-Line\apple_lstm_model.h5')  # 替换为你的模型文件
    return model

# 创建Streamlit应用程序
st.title('股票价格预测应用')
# 侧边栏：用户输入参数
st.sidebar.header('设置参数')

# 获取用户输入参数
start_date = st.sidebar.date_input("选择开始日期", min_value=data['日期'].min(), max_value=data['日期'].max())
end_date = st.sidebar.date_input("选择结束日期", min_value=data['日期'].min(), max_value=data['日期'].max())
prediction_days = st.sidebar.number_input("预测天数", min_value=1, max_value=365, value=30)

# 处理用户输入参数并准备数据
start_index = data[data['日期'] == start_date].index[0]
end_index = data[data['日期'] == end_date].index[0]
input_data = data['收盘'][start_index:end_index+1].values

# 使用模型进行预测
def make_predictions(input_data, prediction_days):
    predictions = []
    for i in range(prediction_days):
        # 获取最后一天的输入数据
        last_input = input_data[-model.input_shape[1]:].reshape(1, -1, 1)
        # 使用模型预测未来一天的股价
        next_day_prediction = model.predict(last_input)[0][0]
        # 将预测结果添加到列表
        predictions.append(next_day_prediction)
        # 更新输入数据，将预测结果添加到末尾
        input_data = np.append(input_data, next_day_prediction)
    return predictions

if st.sidebar.button('开始预测'):
    predictions = make_predictions(input_data, prediction_days)
    st.subheader('预测结果')
    # 显示预测结果的折线图
    st.line_chart(predictions)


