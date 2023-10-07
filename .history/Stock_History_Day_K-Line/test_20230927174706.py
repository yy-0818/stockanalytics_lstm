import efinance as ef

stock_codes = ['600519', '腾讯', 'AAPL']
df = ef.stock.get_quote_history(stock_codes)
print(df)