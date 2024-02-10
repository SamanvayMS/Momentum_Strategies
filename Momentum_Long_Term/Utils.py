import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import scipy.optimize as sc
import seaborn as sns
from IPython.display import display
yf.pdr_override()
    
class Strategy():
    def __init__(self,portfolio_value=1000000,index = 'SP500',data_start_date = '2000-01-01',data_end_date = dt.datetime.today().strftime('%Y-%m-%d'),resampling = 'M',long_only = True,risk_free_rate = 0.02, trading_fees = 0):
        
        self.initial_portfolio_value = portfolio_value
        self.risk_free_rate = risk_free_rate
        self.portfolio_value = portfolio_value
        self.resampling = resampling
        self.long_only = long_only
        self.portfolio_df = None
        self.index = index
        self.tickers = None
        self.start_date = data_start_date
        self.end_date = data_end_date
        self.data_df = None
        self.mtl_df = None
        self.filter_dict = None
        self.returns_list = None
        self.trade_dates = None
        self.top_stocks = None
        self.percentile = 1
        self.trading_fees = trading_fees/100
        self.filter_list_path = None
        self.replenishment = False
        self.extra_replenishment = {}
        self.trading_cost = 0
        self.download_data()
        
    def set_percentile(self, percentile):
        self.percentile = percentile
        
    def set_tradingfees(self, tradingfees):
        self.trading_fees = tradingfees/100
    
    def download_data(self):
        self.get_tickers()
        self.gather_data()
        
    def initialise(self,filter_list_path):
        self.filter_list_path = filter_list_path
        self.read_filters(self.filter_list_path)
        self.generate_returns_list()
        self.get_trade_dates()
        
    def get_tickers(self,get_list = False):
        """Get list of tickers from Wikipedia

        Args:
            market (str, optional): market tickers. Defaults to 'SP500'.

        Raises:
            ValueError: wrong market type string

        Returns:
            list: list of tickers
        """
        if self.index == 'SP500':
            self.get_SP500_tickers()
        elif self.index == 'NASDAQ100':
            self.get_NASDAQ100_tickers()
        elif self.index == 'NSE':
            self.get_nifty_tickers()
        else:
            raise ValueError('Invalid market name')
        if get_list:
            return self.tickers
        else:
            return None
        
    def get_SP500_tickers(self):
        """Get list of tickers from Wikipedia

        Returns:
            list: list of tickers
        """
        self.tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol.to_list()
    
    def get_NASDAQ100_tickers(self):
        """Get list of tickers from Wikipedia
        
        Returns:
            list: list of tickers
        """
        i = 1
        for _ in range(5):
            try:
                self.tickers = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')[i].Ticker.to_list()
            except:
                i+=1

    def get_nifty_tickers(self):
        '''
        Returns a list of all the tickers in NIFTY 50 and NIFTY Next 50
        '''
        i = 1
        for _ in range(5):
            try:
                nifty50 = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50")[i].Symbol.to_list()
            except:
                i+=1
        i = 1
        for _ in range(5):
            try:
                niftynext50 = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_Next_50")[i].Symbol.to_list()
            except:
                i+=1
        nifty50tickers = [x + '.NS' for x in nifty50]
        niftynext50tickers = [x + '.NS' for x in niftynext50]
        self.tickers = nifty50tickers + niftynext50tickers
    
    def gather_data(self):
        self.data_df = yf.download(self.tickers,start=self.start_date,end=self.end_date)['Adj Close']
        self.data_df.dropna(axis=1,inplace=True)
        self.data_df.index = pd.to_datetime(self.data_df.index)
        self.mtl_df = (self.data_df.pct_change()+1)[1:].resample(self.resampling).prod()
        self.data_df = np.round(self.data_df,2)
        
    def read_filters(self,filter_list_path):
        filter_dict = {}
        with open(filter_list_path, mode='r') as file:
            filter_list = file.read().splitlines()
            for line in filter_list:
                lookback,number = line.split(',')
                filter_dict[int(lookback)] = int(number)
        self.filter_dict = filter_dict
    
    def get_rolling_ret(self,n):
        """get the rolling returns"""
        return self.mtl_df.rolling(n).apply(lambda x: np.prod(x)).dropna(axis = 0)
    
    def generate_returns_list(self):
        """generate returns list for all lookbacks"""
        ret_list = []
        for lookback in self.filter_dict.keys():
            ret_list.append(self.get_rolling_ret(lookback))
        self.returns_list = ret_list
    
    def get_trade_dates(self):
        self.trade_dates = self.returns_list[0].index
    
    def get_stocks(self,date):
        if date not in self.trade_dates:
            return None
        self.top_stocks = self.returns_list[0].loc[date].index
        filter_list = list(self.filter_dict.values())
        for i,returns_df in enumerate(self.returns_list):
                threshold = returns_df.loc[date,self.top_stocks].quantile(self.percentile)
                self.top_stocks = returns_df.loc[date,self.top_stocks][returns_df.loc[date, self.top_stocks] < threshold]
                self.top_stocks = self.top_stocks.sort_values(ascending=False)[:filter_list[i]].index
        
    def gather_index_data(self):
        start = self.trade_dates[0]
        end = self.trade_dates[-1]
        
        trading_days = self.data_df.loc[start:end].index
        if start not in trading_days:
            start = self.data_df.index[self.data_df.index.get_loc(trading_days[0])-1]
        end = self.data_df.index[-1]

        if self.index=='SP500':
            sp500_df=yf.download('^GSPC',start=start,end=end)['Adj Close'].pct_change()
            return sp500_df
        elif self.index=='NASDAQ':
            nasdaq_df=yf.download('^IXIC',start=start,end=end)['Adj Close'].pct_change()
            return nasdaq_df
        elif self.index=='NSE':
            nse_df=yf.download('^NSEI',start=start,end=end)['Adj Close'].pct_change()
            return nse_df
        else:
            raise ValueError('Invalid market name')
    
    def portfolio_returns(self,date,portfolio_criteria = 'equal'):
        self.get_stocks(date)
        po = Portfolio_Optimisation(self.data_df,self.mtl_df,date,self.top_stocks,portfolio_criteria,self.risk_free_rate)
        weights = po.get_weights()
        if np.round(np.sum(weights),3) != 1:
            print('Weights do not sum to 1')
        if weights.any() < 0:
            print('Negative weights')
        return self.calculate_portfolio_value(weights,date)
    
    def reset_portfolio(self):
        self.portfolio_value = self.initial_portfolio_value
        self.portfolio_df = pd.DataFrame(columns = ['investment_value','portfolio_value','cash','trading_cost'])
        self.portfolio_df.loc[self.trade_dates[0]] = [0,self.portfolio_value,self.portfolio_value,0]
        self.extra_replenishment = {}
        self.trading_cost = 0
        
    def gen_returns(self,portfolio_criteria = 'equal'):
        self.reset_portfolio()
        for date in self.trade_dates[:-1]:
            df = self.portfolio_returns(date,portfolio_criteria)
            self.portfolio_df = pd.concat([self.portfolio_df,df])
        self.portfolio_df.dropna(axis=0,inplace=True)
        self.portfolio_df.drop_duplicates(inplace=True)
        self.portfolio_df.sort_index(inplace=True)
        return self.portfolio_df
    
    def calculate_portfolio_value(self,weights,date):
        # start and end date to procure the dataframe
        start_date = date
        end_date = self.trade_dates[self.trade_dates.get_loc(date)+1]
        
        # enable,disable replenishment
        if self.replenishment and (self.portfolio_value < self.initial_portfolio_value):
            self.extra_replenishment[date] = self.initial_portfolio_value - self.portfolio_value
            self.portfolio_value = self.initial_portfolio_value
        
        # get the daily dataframe
        df = self.data_df.loc[start_date:end_date,self.top_stocks]
        if df.index[0] != start_date:
            start_date = self.data_df.index[self.data_df.index.get_loc(df.index[0])-1]
            df = self.data_df.loc[start_date:end_date,self.top_stocks]

        # allocation of cash to each stock
        portfolio_allocation = np.array(weights * self.portfolio_value).reshape(-1)
        # stock prices on the start date
        prices = np.array(df.iloc[0])
        # add trading fees to the stock prices
        prices = prices + (prices*self.trading_fees)
        # stock quantity to buy
        qty = np.floor(portfolio_allocation/prices)
        # actual allocation of cash to each stock
        actual_allocation = qty*prices
        # Trading fees
        self.trading_cost += np.round(np.sum(qty*prices*self.trading_fees),2)
        # cash left after buying stocks
        cash = round(self.portfolio_value - np.sum(actual_allocation),2)
        # updating portfolio value for the next day
        portfolio_df = pd.DataFrame(index = df.index[1:],columns = ['investment_value','portfolio_value','cash'])
        
        for date in df.index[1:]:
            prices = np.array(df.loc[date])
            portfolio_df.loc[date,'investment_value'] = np.sum(prices*qty)
            portfolio_df.loc[date,'cash'] = cash
            portfolio_df.loc[date,'portfolio_value'] = portfolio_df.loc[date,'investment_value'] + portfolio_df.loc[date,'cash']
            portfolio_df.loc[date,'trading_cost'] = self.trading_cost
        self.portfolio_value = portfolio_df.portfolio_value[-1]
        return portfolio_df

class Portfolio_Optimisation():
    
    def __init__(self,data_df,mtl_df,date,stocks,portfolio_criteria,risk_free_rate = 0.02):
        self.data_df = data_df
        self.dailypct = self.data_df.pct_change().dropna()
        self.mtl_df = mtl_df
        self.date = date
        self.stocks = stocks
        self.portfolio_criteria = portfolio_criteria
        self.risk_free_rate = risk_free_rate

    def ret_matrix(self):
        returns = self.dailypct[:self.date][-251:]
        negative_returns=returns[returns<0]
        negative_returns=negative_returns.replace(np.nan,0)
        negative_returns = negative_returns[self.stocks].T
        mean_returns = returns.mean()
        mean_returns = mean_returns[self.stocks]
        returns = returns[self.stocks].T
        return np.cov(returns),mean_returns,np.cov(negative_returns)

    def portfolio_performance(self,mean_returns,cov_matrix,weights):
        returns = np.sum(mean_returns*weights)
        std = np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))*np.sqrt(252)
        return returns, std
    
    def negativeSR(self,weights,meanReturns,cov_matrix):
        preturns, pstd = self.portfolio_performance(meanReturns,cov_matrix,weights)
        return -(preturns-self.risk_free_rate)/pstd

    def maxSR(self,meanReturns,cov_matrix,constraintset= (0,1)):
        "minimise the negative sharpe ratio"
        args = (meanReturns,cov_matrix)
        constraints = ({'type':'eq','fun':lambda x: np.sum(x)-1})
        bounds = tuple(constraintset for asset in range(10))
        result = sc.minimize(self.negativeSR,np.ones(10)*0.1,args=args,method = 'SLSQP',bounds=bounds,constraints=constraints)
        return result

    def maximising_sharpe_ratio(self):
        current_portfolio = self.mtl_df.loc[self.date:,self.stocks][1:2]
        cov_matrix , meanReturns,sortino_matrix = self.ret_matrix()
        result = self.maxSR(meanReturns,cov_matrix)
        final_weights=np.round(result['x'],4)
        return final_weights

    def maximising_sortino_ratio(self):
        current_portfolio = self.mtl_df.loc[self.date:,self.stocks][1:2]
        cov_matrix , meanReturns,sortino_matrix = self.ret_matrix()
        result = self.maxSR(meanReturns,sortino_matrix)
        final_weights=np.round(result['x'],4)
        return final_weights
    
    def mean_variance(self,weights,meanReturns,cov_matrix):
        mean_var_returns, mean_var_std = self.portfolio_performance(meanReturns,cov_matrix,weights)
        return mean_var_std

    def MV_portfolio(self,meanReturns,cov_matrix,constraintset= (0,1)):
        "minimise the variance"
        args = (meanReturns,cov_matrix)
        constraints = ({'type':'eq','fun':lambda x: np.sum(x)-1})
        bounds = tuple(constraintset for asset in range(10))
        result = sc.minimize(self.mean_variance,np.ones(10)*0.1,args=args,method = 'SLSQP',bounds=bounds,constraints=constraints)
        return result

    def minimise_variance(self):
        current_portfolio = self.mtl_df.loc[self.date:,self.stocks][1:2]
        cov_matrix , meanReturns,sortino_matrix = self.ret_matrix()
        result_mv = self.MV_portfolio(meanReturns,cov_matrix)
        final_weights_mv=np.round(result_mv['x'],4)
        return final_weights_mv

    def minimise_downside_var(self):
        current_portfolio = self.mtl_df.loc[self.date:,self.stocks][1:2]
        cov_matrix , meanReturns,sortino_matrix = self.ret_matrix()
        result_mv = self.MV_portfolio(meanReturns,sortino_matrix)
        final_weights_mv=np.round(result_mv['x'],4)
        return final_weights_mv
    
    def equal_weightage(self):
        current_portfolio = self.mtl_df.loc[self.date:,self.stocks][1:2]
        final_weights = np.ones(len(self.stocks))/len(self.stocks)
        return final_weights
    
    def get_weights(self):
        if self.portfolio_criteria == 'max_sharpe':
            return self.maximising_sharpe_ratio()
        elif self.portfolio_criteria == 'max_sortino':
            return self.maximising_sortino_ratio()
        elif self.portfolio_criteria == 'min_var':
            return self.minimise_variance()
        elif self.portfolio_criteria == 'min_d_var':
            return self.minimise_downside_var()
        elif self.portfolio_criteria == 'equal':
            return self.equal_weightage()
        else:
            raise ValueError('Invalid portfolio criteria')
        
class Analytics():
    def __init__(self,Strategy):
        self.strategy = Strategy
        self.risk_free_rate = self.strategy.risk_free_rate
        self.equal_returns = self.strategy.gen_returns('equal')
        self.max_sharpe_ratio = self.strategy.gen_returns('max_sharpe')
        self.min_variance = self.strategy.gen_returns('min_var')
        self.index_df = None
        self.var_level = 5
        self.initialise_index()
        self.dfs = [self.equal_returns,self.max_sharpe_ratio,self.min_variance]
        self.create_all_stats()
        
    
    def initialise_index(self):
        self.index_df = self.strategy.gather_index_data()
        self.index_df.replace(np.nan,0,inplace=True)
        self.index_df = pd.DataFrame(self.index_df)
        self.index_df.columns = ['returns']
        self.index_df['cumulative_returns'] = (self.index_df.returns+1).cumprod()
        self.index_df['portfolio_value'] = self.index_df.cumulative_returns*self.strategy.initial_portfolio_value
        self.index_df['account_drawdown'] = (self.index_df.portfolio_value.cummax()-self.index_df.portfolio_value)
        self.index_df['drawdown_pct'] = self.index_df.account_drawdown/self.index_df.portfolio_value.cummax()
        
    def set_var_level(self,level):
        self.var_level = level 
        
    def add_stats_df(self,df):
        df['returns'] = df.portfolio_value.pct_change()
        df['cumulative_returns'] = (df.returns+1).cumprod()
        df['account_drawdown'] = (df.portfolio_value.cummax()-df.portfolio_value)
        df['drawdown_pct'] = df.account_drawdown/df.portfolio_value.cummax()
        df['utilisation_ratio'] = df.investment_value/df.portfolio_value
        
    def create_all_stats(self):
        for df in self.dfs:
            self.add_stats_df(df)

    def plot_Account_value(self,df):
        plt.figure(figsize=(20,10))
        plt.plot(df.portfolio_value,label='portfolio_value')
        plt.plot(df.account_drawdown,label='account_drawdown')
        plt.legend()
        plt.tight_layout()
        plt.title('Account_value')
        plt.show()
        
    def plot_strategy_comparison(self):
        plt.figure(figsize=(20,10))
        plt.plot(self.equal_returns.portfolio_value,label='Equal Weightage')
        plt.plot(self.max_sharpe_ratio.portfolio_value,label='Max Sharpe Ratio')
        plt.plot(self.min_variance.portfolio_value,label='Min Variance')
        plt.plot(self.index_df.portfolio_value,label='Index')
        plt.ylabel('Portfolio Value')
        upper_ylim = max(self.equal_returns.portfolio_value.max(),self.max_sharpe_ratio.portfolio_value.max(),self.min_variance.portfolio_value.max(),self.index_df.portfolio_value.max())*1.1
        plt.ylim(0, upper_ylim)
        plt.xlabel('Date')
        plt.ticklabel_format(style='plain', axis='y',scilimits=(6,6))
        plt.legend()
        plt.tight_layout()
        plt.title('Strategy Comparison')
        plt.show()
        
    def histogram(self,title,returns,ax,bins=25):
        ax.hist(returns,bins)
        ax.set_title(title)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Returns')
        
    def plot_histogram(self):
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))
        
        self.histogram('Equal Weightage',self.equal_returns.returns,axs[0])
        self.histogram('Max Sharpe Ratio',self.max_sharpe_ratio.returns,axs[1])
        self.histogram('Min Variance',self.min_variance.returns,axs[2])
        self.histogram('Index',self.index_df.returns,axs[3])
        
        fig.tight_layout()
        fig.show()
        
    def calculate_financial_metrics(self,df,plot_monte_carlo = False, title = None):
        """
        Calculate various financial metrics for a series of returns.
        
        :param returns: A series of returns (e.g., daily returns).
        :return: A dictionary containing the calculated metrics.
        """
        # Convert returns to a numpy array for efficiency
        returns = df['returns']
        returns.dropna(inplace=True)

        # Calculate annualized return
        annualized_return = (np.prod(returns + 1)**(252 / len(returns)) - 1)
        
        volatility = np.std(returns) * np.sqrt(252)
        # Max Drawdown
        max_drawdown = df['drawdown_pct'].max()

        # Sharpe Ratio
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns)
        sortino_ratio = (annualized_return - self.risk_free_rate) / (downside_std * np.sqrt(252))

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown)
        
        # VaR and CVaR
        VaR , CVaR = self.monte_carlo_simulation(returns, plot = plot_monte_carlo, title = title)

        return {
            'Annualized Return': annualized_return,
            'Max Drawdown': max_drawdown,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'VaR': VaR,
            'CVaR': CVaR
        }
        
    def get_financial_metrics(self,print = False,plot_monte_carlo = False):
        metrics = {}
        metrics['Equal Weightage'] = self.calculate_financial_metrics(self.equal_returns,plot_monte_carlo, 'Equal Weightage')
        metrics['Max Sharpe Ratio'] = self.calculate_financial_metrics(self.max_sharpe_ratio,plot_monte_carlo, 'Max Sharpe Ratio')
        metrics['Min Variance'] = self.calculate_financial_metrics(self.min_variance,plot_monte_carlo, 'Min Variance')
        metrics['Index'] = self.calculate_financial_metrics(self.index_df,plot_monte_carlo, 'Index')
        metrics_df = pd.DataFrame(metrics)
        if print:
            self.Display_metrics_df(metrics_df)
        return metrics_df
    
    def monte_carlo_simulation(self,returns, title, num_simulations=10000,plot = False):
        """
        Perform Monte Carlo simulation on a series of returns to calculate VaR and CVaR.
        
        :param returns: A series of returns.
        :param num_simulations: Number of Monte Carlo simulations to run.
        :param time_horizon: Time horizon for the simulation (in days).
        :param var_level: The confidence level for VaR and CVaR calculation.
        :return: A dictionary containing VaR and CVaR.
        """
        # Convert returns to a numpy array for efficiency
        returns = np.array(returns)
        time_horizon = 252
        # Simulate paths
        simulated_paths = np.zeros((num_simulations, time_horizon))
        for i in range(num_simulations):
            random_sample = np.random.choice(returns, size=time_horizon, replace=True)
            simulated_paths[i] = np.cumprod(random_sample + 1)

        # Calculate VaR and CVaR
        sorted_returns = np.sort(simulated_paths[:,-1])
        var = np.percentile(sorted_returns, self.var_level)
        cvar = sorted_returns[sorted_returns <= var].mean()

        if plot:
            plt.figure(figsize=(20,10))
            plt.plot(simulated_paths.T, alpha=0.5, lw=0.5)
            plt.title(f'Simulated Returns for {num_simulations} Simulations for a period of {time_horizon} days, weightage = {title}')
            plt.xlabel('Time')
            plt.ylabel('Returns (1 + x)')
            plt.hlines(1, 0, time_horizon, linestyles='dashed', color='black')
            plt.hlines(var, 0, time_horizon, linestyles='dashed', color='red', label=f'VaR ({self.var_level}%): {var:.2%}')
            plt.hlines(cvar, 0, time_horizon, linestyles='dashed', color='green', label=f'CVaR ({self.var_level}%): {cvar:.2%}')
            plt.legend()    
            plt.show()
            
        return var, cvar

    def plot_individual_graphs(self,df,title):
        fig, ax = plt.subplots(4, 1, figsize=(20, 20))
        ax[0].plot(df.cumulative_returns - 1,label='Cumulative Returns')
        ax[0].plot(self.index_df.cumulative_returns - 1,label='Index')
        ax[0].set_title(title)
        ax[0].set_ylabel('returns')
        ax[0].legend()
        
        ax[1].plot(-df.drawdown_pct,label='Drawdown')
        ax[1].set_ylabel('drawdown')
        ax[1].legend()
        
        ax[2].plot(df.utilisation_ratio[1:],label='Utilisation Ratio')
        ax[2].set_ylabel('Utilisation Ratio')
        ax[2].set_xlabel('Date')
        ax[2].legend()
        
        ax[3].plot(df.trading_cost,label='Trading Cost')
        ax[3].set_ylabel('Trading Cost')
        ax[3].set_xlabel('Date')
        ax[3].legend()
        
        fig.tight_layout()
        fig.show()
        
    def plot_all_individual_graphs(self):
        self.plot_individual_graphs(self.equal_returns,'Equal Weightage')
        self.plot_individual_graphs(self.max_sharpe_ratio,'Max Sharpe Ratio')
        self.plot_individual_graphs(self.min_variance,'Min Variance')
        
    def yearly_metrics(self,plot = False,print = True):
        yearly_performance = {}
        yearly_performance['equal_returns'] = self.equal_returns.portfolio_value.resample('Y').last().pct_change()[1:]
        yearly_performance['max_sharpe_ratio'] = self.max_sharpe_ratio.portfolio_value.resample('Y').last().pct_change()[1:]
        yearly_performance['min_variance'] = self.min_variance.portfolio_value.resample('Y').last().pct_change()[1:]
        yearly_performance['index'] = self.index_df.portfolio_value.resample('Y').last().pct_change()[1:]
        yearly_performance = pd.DataFrame(yearly_performance)
        yearly_performance.index = yearly_performance.index.year
        
        max_drawdown = {}
        max_drawdown['equal_returns'] = self.equal_returns.drawdown_pct.resample('Y').max()[1:]
        max_drawdown['max_sharpe_ratio'] = self.max_sharpe_ratio.drawdown_pct.resample('Y').max()[1:]
        max_drawdown['min_variance'] = self.min_variance.drawdown_pct.resample('Y').max()[1:]
        max_drawdown['index'] = self.index_df.drawdown_pct.resample('Y').max()[1:]
        max_drawdown = pd.DataFrame(max_drawdown)
        max_drawdown.index = max_drawdown.index.year
        
        if plot: 
            self.plot_yearly_metrics(yearly_performance, max_drawdown)
            
        if print:
            print('Yearly Returns')
            self.Display_yearly_df(yearly_performance)
            print('Yearly Max Drawdown')
            self.Display_yearly_df(max_drawdown, reversed = True)
            
        return yearly_performance,max_drawdown
    
    def plot_yearly_metrics(self,yearly_performance, max_drawdown):
        # Generate Plots
        plt.figure(figsize=(20, 10))

        # Plotting each column
        for column in yearly_performance.columns:
            sns.lineplot(data=yearly_performance[column], label=column + 'return',markers= True)
        for column in max_drawdown.columns:
            sns.lineplot(data=-max_drawdown[column], label=column + ' drawdown',markers=True, linestyle="dashed")
            
        plt.hlines(0, yearly_performance.index[0], yearly_performance.index[-1], linestyles='dashed', color='black',alpha = 0.5)

        plt.title('Yearly Performance Metrics')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        
    def Display_metrics_df(self,df):
        df.loc[["Max Drawdown", "Annualized Return", "Volatility", "VaR", "CVaR"], :] = df.loc[["Max Drawdown", "Annualized Return", "Volatility", "VaR", "CVaR"], :].apply(lambda x: x * 100)
        df.loc[['Max Drawdown'],:] = -df.loc[['Max Drawdown'],:]

        # Apply conditional formatting
        def highlight_max_min(s):
            is_max = s == s.max()
            is_min = s == s.min()
            return ['background-color: rgba(255, 0, 0, 0.2)' if v else 'background-color: rgba(0, 255, 0, 0.2)' if m else '' for v, m in zip(is_min, is_max)]
        def highlight_min_max(s):
            is_max = s == s.max()
            is_min = s == s.min()
            return ['background-color: rgba(0, 255, 0, 0.2)' if m else 'background-color: rgba(255, 0, 0, 0.2)' if v else '' for m, v in zip(is_min, is_max)]
            
        def apply_styles(df):
            reversed_styling = ['Max Drawdown','Volatility', 'VaR', 'CVaR']
            formatted_numbers = ['Annualized Return', 'Max Drawdown','Volatility', 'VaR', 'CVaR']
            df = df.style.apply(highlight_max_min, axis=1)\
                .apply(highlight_min_max, axis=1, subset=pd.IndexSlice[reversed_styling, :])\
                .format("{:.2%}", subset=pd.IndexSlice[formatted_numbers, :])
            return df

        # Applying the styles
        styled_df = apply_styles(df)
        display(styled_df)
        
    def Display_yearly_df(self,df,reversed = False):
        def highlight_max_min(s):
            is_max = s == s.max()
            is_min = s == s.min()
            return ['background-color: rgba(255, 0, 0, 0.2)' if v else 'background-color: rgba(0, 255, 0, 0.2)' if m else '' for v, m in zip(is_min, is_max)]
        def highlight_min_max(s):
            is_max = s == s.max()
            is_min = s == s.min()
            return ['background-color: rgba(0, 255, 0, 0.2)' if m else 'background-color: rgba(255, 0, 0, 0.2)' if v else '' for m, v in zip(is_min, is_max)]

        # Apply the styling
        if reversed:
            styled_df = df.style.apply(highlight_min_max, axis=1)\
                .format("{:.2%}")
        else:
            styled_df = df.style.apply(highlight_max_min, axis=1) \
                .format("{:.2%}")

        display(styled_df)