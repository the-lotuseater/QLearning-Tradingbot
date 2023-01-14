import datetime as dt  		  	   		  	  		  		  		    	 		 		   		 		  
import numpy as np		  	   		  	  		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  	  		  		  		    	 		 		   		 		  
import _util as ut  		  	   		  	  		  		  		    	 		 		   		 		  
import QLearner as ql  		  
import indicators	   		  
	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    :param impact: The market self.impact of each transaction, defaults to 0.0  		  	   		  	  		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param self.commission: The self.commission amount charged, defaults to 0.0  		  	   		  	  		  		  		    	 		 		   		 		  
    :type self.commission: float  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    # constructor  		  	   		  	  		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False,impact=0.0, commission=0.0):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		  	  		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   	
        self.discretized_states = 9
        self.labels = [_ for _ in range(9)]	  	  		  		  		    	 		 		   		 		  
        self.commission = commission  		  	   		  	  		  		  		    	 		 		   		 		  
        self.qlearner = ql.QLearner( 	  	   		  	  		  		  		    	 		 		   		 		  
                            num_states=999,  		  	   		  	  		  		  		    	 		 		   		 		  
                            num_actions=3,  		  	   		  	  		  		  		    	 		 		   		 		  
                            alpha=0.55,  		  	   		  	  		  		  		    	 		 		   		 		  
                            gamma=0.75,  		  	   		  	  		  		  		    	 		 		   		 		  
                            rar=0.7,  		  	   		  	  		  		  		    	 		 		   		 		  
                            radr=0.7,  		  	   		  	  		  		  		    	 		 		   		 		  
                            dyna=0,  		  	   		  	  		  		  		    	 		 		   		 		  
                            verbose=False,  		  	   		  	  		  		  		    	 		 		   		 		  
                        )  	  	
  		  	   		  	  		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		  	  		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		  	  		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        sv=10000,
        max_share=1000	  	   		  	  		  		  		    	 		 		   		 		  
    ):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		  	  		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        # add your code to do learning here  		  	   		  	  		  		  		    	 		 		   		 		  
        # example usage of the old backward compatible util function  		  	   		  	  		  		  		    	 		 		   		 		  
        syms = [symbol]  		  	   		  	  		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		  	   		  	  		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		  	  		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  		  	   		  	  		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		  	  		  		  		    	 		 		   		 		  
            print(prices)  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        # example use with new colname  		  	   		  	  		  		  		    	 		 		   		 		  
        volume_all = ut.get_data(  		  	   		  	  		  		  		    	 		 		   		 		  
            syms, dates, colname="Volume"  		  	   		  	  		  		  		    	 		 		   		 		  
        )  # automatically adds SPY  		  	   		  	  		  		  		    	 		 		   		 		  
        volume = volume_all[syms]  # only portfolio symbols  		  	   		  	  		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		  	  		  		  		    	 		 		   		 		  
            print(volume)  		  	   		  	  		  		  		    	 		 		   		 		  
        start = str(sd)
        adjusted_start = sd-dt.timedelta(100)
        dates = pd.date_range(adjusted_start,ed)
        prices = ut.get_data([symbol],dates)
        prices.drop(['SPY'],axis=1,inplace=True)
        [upper_band,lower_band,_] = indicators.get_bollinger_bands(prices)
        bbp,rsi,ema = indicators.get_bb_percentage(prices=prices,upper=upper_band,lower=lower_band), indicators.get_ema(prices)[0], indicators.get_rsi(prices)
        bbp = bbp.loc[start:]
        rsi = rsi.loc[start:]
        ema = ema.loc[start:]
        discrete_bbp = pd.qcut(bbp.iloc[0:,0],q=self.discretized_states,labels=self.labels)
        discrete_rsi = pd.qcut(rsi.iloc[0:,0],q=self.discretized_states,labels=self.labels)
        discrete_ema = pd.qcut(ema.iloc[0:,0],q=self.discretized_states,labels=self.labels)
        SELL_IDX,HOLD_IDX,BUY_IDX = 0,1,2
        prev_daily_returns = pd.DataFrame()
        prices = prices.loc[start:]
        converged = False
        indices = discrete_bbp.index
        s = 0
        #always assume that a state will be of the form bbp rsi ema and pregenerate states 
        states = pd.DataFrame([discrete_bbp,discrete_rsi,discrete_ema]).astype(str).agg(''.join).astype(int)
        prices['cash'] = 1
        final_holdings = None
        max_iter = 50
        final_trades = None
        itr = 1
        while not converged and itr<=max_iter:
            curr_daily_returns = pd.DataFrame(data=np.zeros(prices.shape[0],),index=prices.index)
            trades = pd.DataFrame(data=np.zeros(prices.shape),index=prices.index,columns=prices.columns)
            holdings = pd.DataFrame(data=np.zeros(prices.shape),index=prices.index,columns=prices.columns)
            for day_num,day in enumerate(indices):
                s_prime = states.loc[day]
                #print('new day new state!',day,s_prime)
                a,r,s = 0,0,0
                if day_num==0:
                    curr_daily_returns.iloc[0]=0
                    a = self.qlearner.querysetstate(s,learning=True)
                    #print('Action on day 1',a)
                    if a==SELL_IDX:
                        factor = -1
                        position_delta = factor*max_share
                        trades.loc[day,symbol] += position_delta
                        imp_adj_share_price = (1-self.impact)*prices.loc[day,symbol]
                        trades.loc[day,'cash'] += -1*position_delta*imp_adj_share_price - self.commission
                    elif a==BUY_IDX:
                        factor = 1
                        position_delta = factor*max_share
                        trades.loc[day,symbol] += position_delta
                        imp_adj_share_price = (1+self.impact)*prices.loc[day,symbol]
                        trades.loc[day,'cash'] += -1*position_delta*imp_adj_share_price - self.commission
                    holdings.iloc[day_num,:] = trades.iloc[day_num,:]
                    holdings.iloc[day_num,-1] += sv
                    #print('Day 1 iteration of holdings',holdings.iloc[day_num,:])
                    #print('Trades on day1 \n',trades.loc[day])
                    #print('Holdings on day1 \n',holdings.loc[day])
                else:
                    r=0
                    if day_num>1:
                        r = ((curr_daily_returns.iloc[day_num-1]-curr_daily_returns.iloc[day_num-2])/curr_daily_returns.iloc[day_num-2])*100
                    a = self.qlearner.query(s_prime,r)
                    if a==SELL_IDX:
                        if holdings.iloc[day_num-1,0]>=0:
                            factor = -1
                            position_delta = factor*max_share
                            trades.loc[day,symbol] += position_delta
                            imp_adj_share_price = (1-self.impact)*prices.loc[day,symbol]
                            trades.loc[day,'cash'] += -1*position_delta*imp_adj_share_price - self.commission
                    elif a==HOLD_IDX:
                        holdings.iloc[day_num,:] = holdings.iloc[day_num-1,:]
                    elif a==BUY_IDX:
                        if holdings.iloc[day_num-1,0]<=0:
                            factor = 1
                            position_delta = factor*max_share
                            trades.loc[day,symbol] += position_delta
                            imp_adj_share_price = (1+self.impact)*prices.loc[day,symbol]
                            trades.loc[day,'cash'] += -1*position_delta*imp_adj_share_price - self.commission
                    holdings.iloc[day_num,:] = holdings.iloc[day_num-1,:] + trades.iloc[day_num,:]
                    #if sum(holdings.iloc[day_num-1,:]*prices.iloc[day_num-1,:])==0:
                        # print('Zero in denom on day', day_num)
                        # print('Holdings',holdings.iloc[day_num-1,:])
                        # print('Prices',prices.iloc[day_num-1,:])

                    #calculate port_vals[i]
                    curr_daily_returns.iloc[day_num] = (sum(holdings.iloc[day_num,:]*prices.iloc[day_num,:])/sum(holdings.iloc[day_num-1,:]*prices.iloc[day_num-1,:]))-1
                    #print('Trades on day1 \n',trades.loc)[day])
                    #print('Holdings on day1 \n',holdings.loc[day])
            if itr>=1:
                if curr_daily_returns.equals(prev_daily_returns):
                    converged = True
                    final_holdings = holdings
                    final_trades = trades
                elif itr+1>=max_iter:
                    final_holdings = holdings
                    final_trades = trades
                    #print('Converged!')
            
            itr+=1
            prev_daily_returns = curr_daily_returns
        if self.verbose:
            print(final_holdings.describe())
            print('Converged after', itr-1)
        #print('por_vals\n',port_vals)
        return final_holdings, final_trades

  		  	   		  	  		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		  	  		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		  	  		  		  		    	 		 		   		 		  
        sd=dt.datetime(2010, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        ed=dt.datetime(2011, 12, 31),  		  	   		  	  		  		  		    	 		 		   		 		  
        sv=100000,  		  	   		  	  		  		  		    	 		 		   		 		  
        max_share=1000
    ):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		  	  		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  	  		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  	  		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  	  		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        # here we build a fake set of trades  		  	   		  	  		  		  		    	 		 		   		 		  
        # your code should return the same sort of data  		
        dates = pd.date_range(sd, ed)  		  	   	
        start = str(sd)
        adjusted_start = sd-dt.timedelta(100)
        new_dates = pd.date_range(adjusted_start,ed)
        prices = ut.get_data([symbol],new_dates)
        prices.drop(['SPY'],axis=1,inplace=True)
        [upper_band,lower_band,_] = indicators.get_bollinger_bands(prices)
        bbp,rsi,ema = indicators.get_bb_percentage(prices=prices,upper=upper_band,lower=lower_band), indicators.get_ema(prices)[0], indicators.get_rsi(prices)
        bbp = bbp.loc[start:]
        rsi = rsi.loc[start:]
        ema = ema.loc[start:]
        discrete_bbp = pd.qcut(bbp.iloc[0:,0],q=self.discretized_states,labels=self.labels)
        discrete_rsi = pd.qcut(rsi.iloc[0:,0],q=self.discretized_states,labels=self.labels)
        discrete_ema = pd.qcut(ema.iloc[0:,0],q=self.discretized_states,labels=self.labels)
        SELL_IDX,HOLD_IDX,BUY_IDX = 0,1,2
        prices = prices.loc[start:]
        indices = discrete_bbp.index
        states = pd.DataFrame([discrete_bbp,discrete_rsi,discrete_ema]).astype(str).agg(''.join).astype(int)
        prices['cash'] = 1
        trades = pd.DataFrame(data=np.zeros(prices.shape),index=prices.index,columns=prices.columns)
        holdings = pd.DataFrame(data=np.zeros(prices.shape),index=prices.index,columns=prices.columns)
        for day_num,day in enumerate(indices):
                s_prime = states.loc[day]
                #print('new day new state!',day,s_prime)
                a = self.qlearner.querysetstate(s_prime)
                if day_num==0:
                    #print('Action on day 1',a)
                    if a==SELL_IDX:
                        factor = -1
                        position_delta = factor*max_share
                        trades.loc[day,symbol] += position_delta
                        imp_adj_share_price = (1-self.impact)*prices.loc[day,symbol]
                        trades.loc[day,'cash'] += -1*position_delta*imp_adj_share_price - self.commission
                    elif a==BUY_IDX:
                        factor = 1
                        position_delta = factor*max_share
                        trades.loc[day,symbol] += position_delta
                        imp_adj_share_price = (1+self.impact)*prices.loc[day,symbol]
                        trades.loc[day,'cash'] += -1*position_delta*imp_adj_share_price - self.commission
                    holdings.iloc[day_num,:] = trades.iloc[day_num,:]
                    holdings.iloc[day_num,-1] += sv
                    #print('Day 1 iteration of holdings',holdings.iloc[day_num,:])
                    #print('Trades on day1 \n',trades.loc[day])
                    #print('Holdings on day1 \n',holdings.loc[day])
                else:
                    if a==SELL_IDX:
                        if holdings.iloc[day_num-1,0]>=0:
                            factor = -1
                            position_delta = factor*max_share
                            trades.loc[day,symbol] += position_delta
                            imp_adj_share_price = (1-self.impact)*prices.loc[day,symbol]
                            trades.loc[day,'cash'] += -1*position_delta*imp_adj_share_price - self.commission
                    elif a==HOLD_IDX:
                        holdings.iloc[day_num,:] = holdings.iloc[day_num-1,:]
                    elif a==BUY_IDX:
                        if holdings.iloc[day_num-1,0]<=0:
                            factor = 1
                            position_delta = factor*max_share
                            trades.loc[day,symbol] += position_delta
                            imp_adj_share_price = (1+self.impact)*prices.loc[day,symbol]
                            trades.loc[day,'cash'] += -1*position_delta*imp_adj_share_price - self.commission
                    holdings.iloc[day_num,:] = holdings.iloc[day_num-1,:] + trades.iloc[day_num,:]
                    #if sum(holdings.iloc[day_num-1,:]*prices.iloc[day_num-1,:])==0:
                        # print('Zero in denom on day', day_num)
                        # print('Holdings',holdings.iloc[day_num-1,:])
                        # print('Prices',prices.iloc[day_num-1,:])

                    #calculate port_vals[i]
                    #print('Trades on day1 \n',trades.loc)[day])
                    #print('Holdings on day1 \n',holdings.loc[day]) 
        if self.verbose:  		  	   		  	  		  		  		    	 		 		   		 		  
            print(trades)  		  	   		  	  		  		  		    	 		 		   		 		  
        return trades.iloc[:,:1]  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  	  		  		  		    	 		 		   		 		  
    mylearner = StrategyLearner()
    start_val=100000
    ticker='JPM'
    mylearner.add_evidence(
        symbol=ticker,  		  	   		  	  		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 31),  		  	   		  	  		  		  		    	 		 		   		 		  
        sv=start_val,
        max_share=1000)
    trades = mylearner.testPolicy(symbol=ticker)
    trades_1 = mylearner.testPolicy(symbol=ticker)
    print('Checking if result both times is the same...')
    print(trades.equals(trades_1))
    print('Trades df \n',trades)
    print(trades.describe())
    holdings = pd.DataFrame(data=np.zeros(trades.shape),columns=trades.columns,index=trades.index)
     
    holdings.iloc[0,:] = trades.iloc[0,:]
    holdings.iloc[0,-1] += start_val

    print('Holdings',holdings.shape)
    for row in range(1,holdings.shape[0]):
        holdings.iloc[row,:] = holdings.iloc[row-1,:] + trades.iloc[row,:]
    print('Holdings\n',holdings)
    start,end = '2010-01-01','2011-12-31'
    dates = pd.date_range(start,end)
    prices = ut.get_data([ticker],dates)
    prices.drop(['SPY'],axis=1,inplace=True)
    prices['cash']=1
    prices_data = prices.to_numpy()
    holding_data = holdings.to_numpy()
    vals = prices_data*holding_data
    port_vals_data = np.sum(vals,axis=1)
    port_vals = pd.DataFrame(data=port_vals_data,columns=[ticker],index=holdings.index)
    print('Portfolio values\n',port_vals)
    