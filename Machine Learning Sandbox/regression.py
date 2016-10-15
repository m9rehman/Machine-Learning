import pandas as pd 
import quandl
import math


df = quandl.get("WIKI/GOOGL")

#Cleaning up the stock df
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HighLow_Volatility'] = (df['Adj. High']- df['Adj. Low'])/df['Adj. High'] * 100 
df['OpenClose_Change'] = (df['Adj. Close']- df['Adj. Open'])/df['Adj. Open'] * 100 

#Redefining the df
df = df[['Adj. Close','OpenClose_Change','HighLow_Volatility','Adj. Volume']] #Our Features 

#We want to try and forecast the Adj. Close
forecast_column = 'Adj. Close'
df.fillna(-9999, inplace=True) #fills the NaN with the value we pass

forecast_out = int(math.ceil(0.01*len(df))) #Percentage of our data to forecast out

df['label'] = df[forecast_column].shift(-forecast_out) 
#Features are what may cause our label to change 

df.dropna()
print(df.tail())