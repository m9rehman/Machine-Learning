import pandas as pd 
import quandl


df = quandl.get("WIKI/GOOGL")

#Cleaning up the stock df
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HighLow_Volatility'] = (df['Adj. High']- df['Adj. Low'])/df['Adj. High'] * 100 
df['OpenClose_Change'] = (df['Adj. Close']- df['Adj. Open'])/df['Adj. Open'] * 100 

#Redefining the df
df = df[['Adj. Close','OpenClose_Change','HighLow_Volatility','Adj. Volume']]
print(df.head())
print(1)