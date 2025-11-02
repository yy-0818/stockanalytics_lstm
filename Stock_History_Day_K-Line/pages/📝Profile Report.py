import logging
import os
import pandas as pd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
# import pandas_profiling


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

def safe_profile_report(df, title="数据分析报告"):
    """安全生成数据报告，避免图像处理错误"""
    try:
        # 使用最小化配置避免图像处理错误
        profile = ProfileReport(
            df,
            title=title,
            minimal=True,  # 使用最小化模式
            explorative=True,
            correlations=None,  # 禁用相关分析
            missing_diagrams=None,  # 禁用缺失值图表
            interactions=None,  # 禁用交互图表
        )
        return profile
    except Exception as e:
        st.error(f"生成详细报告时出错: {str(e)}")
        # 备用方案：使用更简化的配置
        try:
            profile = ProfileReport(df, minimal=True, title=title)
            return profile
        except Exception as e2:
            st.error(f"简化报告也失败: {str(e2)}")
            return None

def create_basic_report(df, title="数据分析报告"):
    """创建基本数据报告（备用方案）"""
    st.header(f"📊 {title}")
    
    # 基本信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("数据行数", df.shape[0])
    with col2:
        st.metric("数据列数", df.shape[1])
    with col3:
        st.metric("缺失值总数", df.isnull().sum().sum())
    with col4:
        st.metric("重复行数", df.duplicated().sum())
    
    # 数据预览
    st.subheader("数据预览")
    st.dataframe(df.head(10))
    
    # 数据类型
    st.subheader("数据类型")
    dtype_df = pd.DataFrame(df.dtypes, columns=['数据类型'])
    st.dataframe(dtype_df)
    
    # 描述性统计
    st.subheader("描述性统计")
    st.dataframe(df.describe())
    
    # 缺失值分析
    st.subheader("缺失值分析")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_df = pd.DataFrame({
            '列名': missing_data.index,
            '缺失值数量': missing_data.values,
            '缺失值比例': (missing_data.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['缺失值数量'] > 0]
        st.dataframe(missing_df)
    else:
        st.success("✅ 没有缺失值")

def upload_and_profile_data():
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        file_name = os.path.splitext(uploaded_file.name)[0]
        st.title(f"{file_name}—自动分析报告")
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("文件上传成功.")
        
        # 使用安全的报告生成
        with st.spinner("正在生成数据分析报告..."):
            profile = safe_profile_report(df, f"{file_name}数据分析报告")
            
            if profile is not None:
                try:
                    st_profile_report(profile)
                except Exception as e:
                    st.error(f"显示报告时出错: {str(e)}")
                    st.info("使用基本数据报告替代...")
                    create_basic_report(df, f"{file_name}基本数据报告")
            else:
                st.warning("无法生成详细报告，显示基本数据报告")
                create_basic_report(df, f"{file_name}基本数据报告")
    else:
        st.sidebar.write("上传 CSV 文件以生成配置文件报告.")

def main():
    stock_data_dict = {
        '贵州茅台': moutai_stock,
        '苹果': aapl_stock,
        '腾讯控股': tencent_stock
    }
    
    with st.sidebar:
        st.markdown('## 设置参数 📁')
        st.write('User input parameters below ⬇️')

        # 在侧边栏中创建选择框
        stock_df = st.sidebar.selectbox('选择数据集', list(stock_data_dict.keys()) + ["上传CSV文件"])
        
        # st.info('该项目可以帮助你理解LSTM')

    # 如果用户选择了上传 CSV 文件，则调用上传功能
    if stock_df == "上传CSV文件":
        upload_and_profile_data()
        st.divider()
        st.sidebar.caption('<p style="text-align:center">made with ❤️ by Yuan</p>', unsafe_allow_html=True)
    else:
        selected_stock_df = stock_data_dict[stock_df]
        stock_name = selected_stock_df['Stock Name'].iloc[0] if 'Stock Name' in selected_stock_df.columns else stock_df
        st.title(f"{stock_name}——自动分析报告")
        st.divider()
        st.sidebar.caption('<p style="text-align:center">made with ❤️ by Yuan</p>', unsafe_allow_html=True)
        
        # 显示原始数据
        if stock_df in stock_data_dict:
            see_data = st.expander(f'查看原始数据 \ View the raw data for {stock_df} 👉')
            with see_data:
                st.dataframe(selected_stock_df.reset_index(drop=True))
            
            # 使用安全的报告生成
            with st.spinner("正在生成数据分析报告..."):
                profile = safe_profile_report(selected_stock_df, f"{stock_name}数据分析报告")
                
                if profile is not None:
                    try:
                        st_profile_report(profile)
                    except Exception as e:
                        st.error(f"显示报告时出错: {str(e)}")
                        st.info("使用基本数据报告替代...")
                        create_basic_report(selected_stock_df, f"{stock_name}基本数据报告")
                else:
                    st.warning("无法生成详细报告，显示基本数据报告")
                    create_basic_report(selected_stock_df, f"{stock_name}基本数据报告")
        else:
            st.sidebar.write('未知数据集:', stock_df)

if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)
    add_logo()
    main()
