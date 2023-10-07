from typing import Dict
import efinance as ef
import pandas as pd
import time
from datetime import datetime
import os
# 股票代码或者名称列表
stock_codes = ['贵州茅台', '腾讯', 'AAPL']

# 数据间隔时间为 1 分钟
freq = 1
status = {stock_code: 0 for stock_code in stock_codes}
while len(stock_codes) != 0:
    # 获取最新一个交易日的分钟级别股票行情数据
    stocks_df: Dict[str, pd.DataFrame] = ef.stock.get_quote_history(
        stock_codes, klt=freq)
    for stock_code, df in stocks_df.items():
        # 现在的时间
        now = str(datetime.today()).split('.')[0]
        # 获取当前目录
        current_dir = os.getcwd()
        # 构造文件名
        file_name = f'{stock_code}.csv'
        # 构造文件路径
        file_path = os.path.join(current_dir+'\Stock_Daily_K-Line', file_name)
        # 将数据保存到文件
        df.to_csv(file_path, encoding='utf-8-sig', index=None)
        print(f'已在 {now}, 将股票: {stock_code} 的行情数据存储到文件: {file_path} 中！')
        
        if len(df) == status[stock_code]:
            # 移除已经收盘的股票代码
            stock_codes.remove(stock_code)
            print(f'股票 {stock_code} 已收盘！')
        status[stock_code] = len(df)
    if len(stock_codes) != 0:
        print('暂停 60 秒')
        time.sleep(60)
    print('-'*10)

print('全部股票已收盘')