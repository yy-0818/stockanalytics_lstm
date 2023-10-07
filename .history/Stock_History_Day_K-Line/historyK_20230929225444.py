from typing import Dict
import efinance as ef
import pandas as pd
import time
from datetime import datetime
import os

stock_codes = ['贵州茅台', '腾讯', '苹果']
# df = ef.stock.get_quote_history(stock_codes)

# 创建一个空的DataFrame，用于存储合并后的数据
combined_df = pd.DataFrame()

stocks_df: Dict[str, pd.DataFrame] = ef.stock.get_quote_history(stock_codes)
# print(stocks_df)
for stock_code, df in stocks_df.items():
    # 构造文件名
        file_name = f'{stock_code}.csv'
        # 构造文件路径
        file_path = os.path.join(os.getcwd()+'\Stock_History_Day_K-Line', file_name)
        # 将数据保存到文件
        df.to_csv(file_path, encoding='utf-8-sig', index=None)
        print(f'已将{stock_code} 的历史股票数据存储到文件: {file_path}中')
        print('-'*10)
        # 将股票数据添加到合并后的DataFrame中
        combined_df = pd.concat([combined_df, df])

# 保存合并后的数据到一个新的CSV文件
combined_file_path = os.path.join(os.getcwd()+'\Stock_History_Day_K-Line', 'combined_stock_data.csv')
combined_df.to_csv(combined_file_path, encoding='utf-8-sig', index=None)

print(f'已合并所有股票数据到文件: {combined_file_path}')