from typing import Dict
import efinance as ef
import pandas as pd
import time
from datetime import datetime
import os

stock_codes = ['贵州茅台', '腾讯', '苹果']
# df = ef.stock.get_quote_history(stock_codes)

stocks_df: Dict[str, pd.DataFrame] = ef.stock.get_quote_history(stock_codes, klt=freq)
stocks_df