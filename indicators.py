from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data
import numpy as np

def get_bollinger_bands(prices: pd.DataFrame):
	if prices.shape[0]<2:
		raise Exception('ERROR: Prices cannot be empty or have only 1 value')
	rolling_means = prices.rolling(20).mean()
	rolling_stds = prices.rolling(20).std()
	upper_band = rolling_means + 2 * rolling_stds
	lower_band = rolling_means -  2 * rolling_stds
	return [upper_band,lower_band,rolling_means]
    
def get_bb_percentage(prices: pd.DataFrame, upper: pd.DataFrame, lower: pd. DataFrame):
	bb_percentage = (prices-lower)/(upper-lower)
	return bb_percentage

def plot_bb_percentage(prices: pd.DataFrame, bb_percentage: pd.DataFrame):
	ax1 = plt.subplot2grid((10,1),(0,0),rowspan=5,colspan=1)
	ax2 = plt.subplot2grid((8,1),(5,0),rowspan=3,colspan=1)
	#SYMBOL stock price
	plt.sca(ax1)
	ax1.plot(prices.index,prices,label='BB%')
	ax1.set_ylabel('Prices')
	ax1.grid()
	ax1.set_xlim(prices.index[0],prices.index[-1])
	ax1.text(0.5, 0.5, 'Abhishek Birhade', transform=ax1.transAxes,fontsize=40, color='gray', alpha=0.5,ha='center', va='center', rotation=30)
	ax1.set_title('JPM prices and % Bollinger Bands')
	plt.xticks(rotation='30',ha='right')
	#bb% plot
	plt.sca(ax2)
	ax2.plot(bb_percentage.index,bb_percentage,label='BB%',color='orange')
	ax2.text(0.5, 0.5, 'Abhishek Birhade', transform=ax2.transAxes,fontsize=40, color='gray', alpha=0.5,ha='center', va='center', rotation=30)
	ax2.legend(loc='upper right')
	ax2.grid()
	ax2.set_xlim(prices.index[0],prices.index[-1])
	ax2.set_ylabel('BB%')
	plt.xticks(rotation='30',ha='right')
	plt.savefig('./bb_percentage.png')
    
def plot_bollinger_bands(prices: pd.DataFrame, upper_band: pd.DataFrame, lower_band: pd.DataFrame, sma: pd.DataFrame, ticker='JPM'):
	if prices.shape[0]<2:
		raise Exception('ERROR: Prices cannot be empty or have only 1 value')
	if upper_band.shape != lower_band.shape:
		raise Exception('ERROR upper and lower bands must have the same shape')
	ax = prices.plot(title='Bollinger Bands Analysis',label=ticker)
	ax.plot(sma.index,sma,label='20 day moving Average')
	ax.plot(upper_band.index,upper_band,label='upper band')
	ax.plot(lower_band.index,lower_band,label='lower band')
	ax.legend(loc='lower right')
	ax.text(0.5, 0.5, 'Abhishek Birhade', transform=ax.transAxes,fontsize=40, color='gray', alpha=0.5,ha='center', va='center', rotation=30)
	ax.grid()
	plt.xticks(rotation=30,ha='right')
	plt.xlabel('Dates')
	plt.ylabel('Prices')
	plt.savefig('./bollinger_bands.png')
	plt.cla()
 
def get_ema(prices: pd.DataFrame, fast=9, slow=13, long=50):
	if prices.shape[0]<fast:
		raise Exception('ERROR: Prices cannot be empty or have only 1 value')
	ema_1=prices.ewm(span=fast,adjust=False).mean()
	ema_2=prices.ewm(span=slow,adjust=False).mean()
	ema_3=prices.ewm(span=long,adjust=False).mean()
	return [ema_1,ema_2,ema_3]

def plot_ema(prices: pd.DataFrame, ema: pd.DataFrame):
	ax=prices.plot(title='EMA Analysis',label='JPM')
	if len(ema==1):
		ax.plot(prices.index,ema,label='9 Day EMA')
	else:
		ax.plot(prices.index,ema[0].iloc,label='9 day EMA')
		ax.plot(prices.index,ema[1].iloc,label='13 day EMA')
		ax.plot(prices.index,ema[2].iloc,label='50 day EMA')
	ax.legend(loc='upper right')
	ax.text(0.5, 0.5, 'Abhishek Birhade', transform=ax.transAxes,fontsize=40, color='gray', alpha=0.5,ha='center', va='center', rotation=30)
	ax.grid()
	plt.xticks(rotation=30,ha='right')
	plt.xlabel('Dates')
	plt.ylabel('Prices')
	plt.savefig('./EMA_Analysis.png')
	plt.cla()

def get_rsi(prices: pd.DataFrame, period=14):
	diff = prices.diff()
	up = diff.clip(lower=0)
	down = -1* diff.clip(upper=0)
	ma_up = up.rolling(period).mean()
	ma_down = down.rolling(period).mean()
	rsi = ma_up/ma_down
	rsi = 100 - (100/(1+rsi))
	return rsi
 
def plot_rsi(rsi: pd.DataFrame, prices: pd.DataFrame):
	rsi['RSI'] = rsi
	rsi.drop(['JPM'],inplace=True,axis=1)
	ax=rsi.plot(title='RSI',label='RSI')
	ax.plot(prices.index,prices,label='JPM')
	ax.plot(rsi.index, [70]*rsi.shape[0],label='Over bought')
	ax.plot(rsi.index, [30]*rsi.shape[0],label='Sold bought')
	ax.legend(loc='upper left')
	ax.text(0.5, 0.5, 'Abhishek Birhade', transform=ax.transAxes,fontsize=40, color='gray', alpha=0.5,ha='center', va='center', rotation=30)
	ax.grid()
	plt.xticks(rotation=30,ha='right')
	plt.xlabel('Dates')
	plt.ylabel('RSI Value')
	plt.savefig('./rsi.png')
	plt.cla()

def get_oscillator(prices: pd.DataFrame, period=14):
	high = prices['High'].rolling(period).max()
	low = prices['Low'].rolling(period).min()
	numer = prices['Close'] - low
	denom = high-low
	fast = 100*(numer/denom)
	slow = fast.rolling(3).mean()
	return fast.iloc[14:],slow.iloc[14:]

def plot_oscillator(prices: pd.DataFrame, fast: pd.DataFrame, slow: pd.DataFrame):
	ax1 = plt.subplot2grid((10,1),(0,0),rowspan=5,colspan=1)
	ax2 = plt.subplot2grid((7,1),(5,0),rowspan=3,colspan=1)
	plt.sca(ax1)
	plt.title('JPM Prices And Fast Oscillator(%K) Analysis')
	ax1.plot(prices.index,prices,label='JPM')
	ax1.legend(loc='lower right')
	ax1.grid()
	ax1.text(0.5, 0.5, 'Abhishek Birhade', transform=ax1.transAxes,fontsize=40, color='gray', alpha=0.5,ha='center', va='center', rotation=30)
	plt.xticks(rotation=30,ha='right')
	plt.ylabel('Prices')

	plt.sca(ax2)
	ax2.plot(fast.index,fast,label='%K',color='green')
	ax2.plot(slow.index,slow,label='%D', color='magenta')
	ax2.plot(fast.index,[80]*len(fast.index),color='black')
	ax2.plot(fast.index,[20]*len(fast.index),color='black')
	ax2.text(0.5, 0.5, 'Abhishek Birhade', transform=ax2.transAxes,fontsize=40, color='gray', alpha=0.5,ha='center', va='center', rotation=30)
	ax2.grid()
	ax2.legend(loc='lower right')
	plt.ylim(0,100)
	plt.xticks(rotation=30,ha='right')
	plt.xlabel('Dates')
	plt.ylabel('OSCI Value')
	plt.savefig('./oscillator.png')
	plt.cla()
 
def get_macd_and_signal(prices: pd.DataFrame, fast=9, slow=13):
	ema_fast,ema_slow,_ = get_ema(prices)
	ema_fast = ema_fast
	ema_slow = ema_slow
	macd = ema_fast-ema_slow
	signal = macd.ewm(span=9,adjust=False).mean()
	signal['SIGNAL'] = signal
	signal.drop(['JPM'],axis=1,inplace=True)
	macd['MACD'] = macd
	macd.drop(['JPM'],axis=1,inplace=True)
	hist = pd.DataFrame(data=macd.to_numpy() - signal.to_numpy(),columns=['HIST'], index=signal.index)
	dfs = [macd,signal,hist]
	macd_metadata = pd.concat(dfs,join='inner',axis=1)
	return macd_metadata
 
def plot_macd(macd_metadata: pd.DataFrame,prices: pd.DataFrame):
	ax1 = plt.subplot2grid((10,1),(0,0),rowspan=5,colspan=1)
	ax2 = plt.subplot2grid((7,1),(5,0),rowspan=3,colspan=1)
	plt.sca(ax1)
	plt.ylabel('Prices')
	ax1.set_title('MACD Analysis of JPM')
	ax1.text(0.5, 0.5, 'Abhishek Birhade', transform=ax1.transAxes,fontsize=40, color='gray', alpha=0.5,ha='center', va='center', rotation=30)
	ax1.plot(prices.index,prices,label='JPM')
	ax1.set_title
	plt.xticks(rotation=30,ha='right')
	ax1.grid()
	ax1.legend(loc='lower right')
	plt.sca(ax2)
	ax2.plot(prices.index,macd_metadata['MACD'], color = 'orange', linewidth = 1.5, label = 'MACD')
	ax2.plot(prices.index,macd_metadata['SIGNAL'], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')
	ax2.text(0.5, 0.5, 'Abhishek Birhade', transform=ax2.transAxes,fontsize=40, color='gray', alpha=0.5,ha='center', va='center', rotation=30)
	for index in macd_metadata.index:
		if macd_metadata.loc[index,'HIST']<0:
			ax2.bar(index,macd_metadata.loc[index,'HIST'],color='red')
		else:
			ax2.bar(index,macd_metadata.loc[index,'HIST'],color='green')
	plt.xticks(rotation=30,ha='right')
	plt.xlabel('Dates')
	plt.ylabel('MACD value')
	plt.grid()
	plt.legend(loc='lower right')
	plt.savefig('./macd.png')

#Expecation here is to create a new equation for each of these based on the buy sell signals of the indicators. 
#And come up with a value of the portfolio

'''
Which indicators to pick-
RSI, bollinger bands, EMA
'''

if __name__=='__main__':
	start_date,end_date='2007-10-19','2009-12-31'
	dates = pd.date_range(start_date,end_date)
	prices = get_data(['JPM'],dates)
	jpm=prices.drop(['SPY'],axis=1)
	u,l=get_bollinger_bands(jpm.loc['2008-1-2':])
	plot_bollinger_bands(jpm.iloc[50:],u,l)

	emas = get_ema(jpm)
	plot_ema(jpm,emas)

	rsi=get_rsi(jpm.loc['2008-1-2':])
	plot_rsi(rsi,jpm.loc['2008-1-2':])

	prices_osci_h = get_data(symbols=['JPM'],dates=dates,colname='High').drop(['SPY'],axis=1)
	prices_osci_l = get_data(symbols=['JPM'],dates=dates,colname='Low').drop(['SPY'],axis=1)
	prices_osci = get_data(symbols=['JPM'],dates=dates,colname='Adj Close').drop(['SPY'],axis=1)
	prices_osci['Adj Close'] = prices_osci
	prices_osci.drop(['JPM'],axis=1,inplace=True)
	prices_osci['High'] = prices_osci_h
	prices_osci['Low'] = prices_osci_l
	fast,slow = get_oscillator(prices_osci.loc['2007-12-11':])
	plot_oscillator(fast,slow)

	macd_data = get_macd_and_signal(jpm,fast=13,slow=26)
	macd_data = macd_data.loc['2008-01-02':]
	print('macd_data \n',macd_data)
	plot_macd(macd_data,jpm.loc['2008-01-02':])