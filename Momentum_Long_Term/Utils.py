import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import scipy.optimize as sc
yf.pdr_override()
    
class Strategy():
    def __init__(self,portfolio_value=1000000,index = 'SP500',data_start_date = '2000-01-01',data_end_date = dt.datetime.today().strftime('%Y-%m-%d'),filter_list_path = 'filter_list.txt'):
        
        self.initial_portfolio_value = portfolio_value
        self.portfolio_value = portfolio_value
        self.portfolio_df = None
        self.index = index
        self.tickers = None
        self.start_date = data_start_date
        self.end_date = data_end_date
        self.data_df = None
        self.mtl_df = None
        self.index_df = None
        self.filter_dict = None
        self.returns_list = None
        self.trade_dates = None
        self.top_stocks = None
        self.percentile = 1
        self.filter_list_path = filter_list_path
        self.replenishment = False
        self.extra_replenishment = {}
        self.initialise()
        
    def set_percentile(self, percentile):
        self.percentile = percentile
    
    def initialise(self):
        self.get_tickers()
        self.gather_data()
        self.read_filters(self.filter_list_path)
        self.generate_returns_list()
        self.get_trade_dates()
        self.gather_index_data()
        self.portfolio_df = pd.DataFrame(index = self.trade_dates,columns = ['investment_value','portfolio_value','cash'])
        
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
        self.mtl_df = (self.data_df.pct_change()+1)[1:].resample("M").prod()
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
        if self.index=='SP500':
            sp500_df=yf.download('^GSPC',start=start,end=end)['Adj Close'].pct_change()
            sp500_df=sp500_df[start:]
            self.index_df = sp500_df
        elif self.index=='NASDAQ':
            nasdaq_df=yf.download('^IXIC',start=start,end=end)['Adj Close'].pct_change()
            nasdaq_df=nasdaq_df[start:]
            self.index_df = nasdaq_df
        elif self.index=='NSE':
            nse_df=yf.download('^NSEI',start=start,end=end)['Adj Close'].pct_change()
            nse_df=nse_df[start:]
            self.index_df = nse_df
        else:
            raise ValueError('Invalid market name')
    
    def portfolio_returns(self,date,portfolio_criteria = 'equal'):
        self.get_stocks(date)
        po = Portfolio_Optimisation(self.data_df,self.mtl_df,date,self.top_stocks,portfolio_criteria)
        returns , weights, stocks = po.get_stats()
        return self.calculate_portfolio_value(weights,date)
    
    def gen_returns(self,portfolio_criteria = 'equal'):
        self.portfolio_value = self.initial_portfolio_value
        self.portfolio_df.loc[self.trade_dates[0]] = [0,self.portfolio_value,self.portfolio_value]
        for date in self.trade_dates[:-1]:
            df = self.portfolio_returns(date,portfolio_criteria)
            self.portfolio_df = pd.concat([self.portfolio_df,df])
        self.portfolio_df.sort_index(inplace=True)
        return self.portfolio_df
    
    def calculate_portfolio_value(self,weights,date, with_replenishment = False):
        # start and end date to procure the dataframe
        start_date = date
        end_date = self.trade_dates[self.trade_dates.get_loc(date)+1]
        print(start_date,end_date)
        
        # enable,disable replenishment
        if with_replenishment and (self.portfolio_value < self.initial_portfolio_value):
            self.extra_replenishment[date] = self.initial_portfolio_value - self.portfolio_value
            self.portfolio_value = self.initial_portfolio_value
        
        # get the daily dataframe
        df = self.data_df.loc[start_date:end_date,self.top_stocks]
        if df.index[0] != start_date:
            start_date = self.data_df.index[self.data_df.index.get_loc(df.index[0])-1]
            df = self.data_df.loc[start_date:end_date,self.top_stocks]
        
        print(df.index)
        # allocation of cash to each stock
        portfolio_allocation = np.array(weights * self.portfolio_value).reshape(-1)
        # stock prices on the start date
        prices = np.array(df.iloc[0])
        # stock quantity to buy
        qty = np.floor(portfolio_allocation/prices)
        # actual allocation of cash to each stock
        actual_allocation = qty*prices
        # cash left after buying stocks
        cash = round(self.portfolio_value - np.sum(actual_allocation),2)
        # updating portfolio value for the next day
        portfolio_df = pd.DataFrame(index = df.index[1:],columns = ['investment_value','portfolio_value','cash'])
        
        for date in df.index[1:]:
            prices = np.array(df.loc[date])
            portfolio_df.loc[date,'investment_value'] = np.sum(prices*qty)
            portfolio_df.loc[date,'cash'] = cash
            portfolio_df.loc[date,'portfolio_value'] = portfolio_df.loc[date,'investment_value'] + portfolio_df.loc[date,'cash']
        self.portfolio_value = portfolio_df.portfolio_value[-1]
        return portfolio_df

class Portfolio_Optimisation():
    
    def __init__(self,data_df,mtl_df,date,stocks,portfolio_criteria):
        self.data_df = data_df
        self.dailypct = self.data_df.pct_change().dropna()
        self.mtl_df = mtl_df
        self.date = date
        self.stocks = stocks
        self.portfolio_criteria = portfolio_criteria

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
    
    def negativeSR(self,weights,meanReturns,cov_matrix,riskfreerate=0):
        preturns, pstd = self.portfolio_performance(meanReturns,cov_matrix,weights)
        return -(preturns-riskfreerate)/pstd

    def maxSR(self,meanReturns,cov_matrix,riskfreerate = 0,constraintset= (0,1)):
        "minimise the negative sharpe ratio"
        args = (meanReturns,cov_matrix,riskfreerate)
        constraints = ({'type':'eq','fun':lambda x: np.sum(x)-1})
        bounds = tuple(constraintset for asset in range(10))
        result = sc.minimize(self.negativeSR,np.ones(10)*0.1,args=args,method = 'SLSQP',bounds=bounds,constraints=constraints)
        return result

    def maximising_sharpe_ratio(self):
        current_portfolio = self.mtl_df.loc[self.date:,self.stocks][1:2]
        cov_matrix , meanReturns,sortino_matrix = self.ret_matrix()
        result = self.maxSR(meanReturns,cov_matrix)
        final_weights=np.round(result['x'],4)
        finalreturn = np.sum(current_portfolio*final_weights,axis=1)
        return finalreturn[0],final_weights,current_portfolio

    def maximising_sortino_ratio(self):
        current_portfolio = self.mtl_df.loc[self.date:,self.stocks][1:2]
        cov_matrix , meanReturns,sortino_matrix = self.ret_matrix()
        result = self.maxSR(meanReturns,sortino_matrix)
        final_weights=np.round(result['x'],4)
        finalreturn = np.sum(current_portfolio*final_weights,axis=1)
        return finalreturn[0],final_weights,current_portfolio
    
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
        finalreturn_mv = np.sum(current_portfolio*final_weights_mv,axis=1)
        return finalreturn_mv[0],final_weights_mv,current_portfolio

    def minimise_downside_var(self):
        current_portfolio = self.mtl_df.loc[self.date:,self.stocks][1:2]
        cov_matrix , meanReturns,sortino_matrix = self.ret_matrix()
        result_mv = self.MV_portfolio(meanReturns,sortino_matrix)
        final_weights_mv=np.round(result_mv['x'],4)
        finalreturn_mv = np.sum(current_portfolio*final_weights_mv,axis=1)
        return finalreturn_mv[0],final_weights_mv,current_portfolio
    
    def equal_weightage(self):
        current_portfolio = self.mtl_df.loc[self.date:,self.stocks][1:2]
        final_weights = np.ones(len(self.stocks))/len(self.stocks)
        finalreturn = np.sum(current_portfolio*final_weights,axis=1)
        return finalreturn[0],final_weights,current_portfolio
    
    def get_stats(self):
        if self.portfolio_criteria == 'maximising_sharpe_ratio':
            return self.maximising_sharpe_ratio()
        elif self.portfolio_criteria == 'maximising_sortino_ratio':
            return self.maximising_sortino_ratio()
        elif self.portfolio_criteria == 'minimise_variance':
            return self.minimise_variance()
        elif self.portfolio_criteria == 'minimise_downside_var':
            return self.minimise_downside_var()
        elif self.portfolio_criteria == 'equal':
            return self.equal_weightage()
        else:
            raise ValueError('Invalid portfolio criteria')
        
class Analytics():
    def __init__(self,Strategy):
        self.strategy = Strategy
        self.equal_returns = self.strategy.gen_returns('equal')
        self.max_sharpe_ratio = self.strategy.gen_returns('maximising_sharpe_ratio')
        self.max_sortino_ratio = self.strategy.gen_returns('maximising_sortino_ratio')
        self.min_variance = self.strategy.gen_returns('minimise_variance')
        self.min_downside_var = self.strategy.gen_returns('minimise_downside_var')
        self.dfs = [self.equal_returns,self.max_sharpe_ratio,self.max_sortino_ratio,self.min_variance,self.min_downside_var]
        self.create_all_stats()
    
    def add_stats_df(self,df):
        df['returns'] = df.portfolio_value.pct_change()
        df['cumulative_returns'] = (df.returns+1).cumprod()
        df['account_drawdown'] = (df.portfolio_value.cummax()-df.portfolio_value)
        df['drawdown_pct'] = df.account_drawdown/df.portfolio_value.cummax()
        df['uitilisation_ratio'] = df.investment_value/df.portfolio_value

    def plot_Account_value(self,df):
        plt.figure(figsize=(20,10))
        plt.plot(df.portfolio_value,label='Portfolio Value')
        plt.plot(df.account_drawdown,label='Account Drawdown')
        plt.legend()
        plt.tight_layout()
        plt.title('Account_value')
        plt.show()
        
    def plot_strategy_comparison(self):
        plt.figure(figsize=(20,10))
        plt.plot(self.equal_returns.portfolio_value,label='Equal Weightage')
        plt.plot(self.max_sharpe_ratio.portfolio_value,label='Max Sharpe Ratio')
        plt.plot(self.max_sortino_ratio.portfolio_value,label='Max Sortino Ratio')
        plt.plot(self.min_variance.portfolio_value,label='Min Variance')
        plt.plot(self.min_downside_var.portfolio_value,label='Min Downside Variance')
        plt.legend()
        plt.tight_layout()
        plt.title('Strategy Comparison')
        plt.show()
        
    def create_all_stats(self):
        for df in self.dfs:
            self.add_stats_df(df)

    def plot_histogram(self):
        fig, axs = plt.subplots(5, 1, figsize=(10, 20))
        
        axs[0].hist(self.equal_returns.returns,bins=25)
        axs[0].set_title('Equal Weightage')
        axs[0].set_ylabel('Frequency')
        axs[0].set_xlabel('Returns')
        
        axs[1].hist(self.max_sharpe_ratio.returns,bins=25)
        axs[1].set_title('Max Sharpe Ratio')
        axs[1].set_ylabel('Frequency')
        axs[1].set_xlabel('Returns')
        
        axs[2].hist(self.max_sortino_ratio.returns,bins=25)
        axs[2].set_title('Max Sortino Ratio')
        axs[2].set_ylabel('Frequency')
        axs[2].set_xlabel('Returns')
        
        axs[3].hist(self.min_variance.returns,bins=25)
        axs[3].set_title('Min Variance')
        axs[3].set_ylabel('Frequency')
        axs[3].set_xlabel('Returns')
        
        axs[4].hist(self.min_downside_var.returns,bins=25)
        axs[4].set_title('Min Downside Variance')
        axs[4].set_ylabel('Frequency')
        axs[4].set_xlabel('Returns')
        
        fig.tight_layout()
        fig.show()
        
    def calculate_financial_metrics(self,df):
        """
        Calculate various financial metrics for a series of returns.
        
        :param returns: A series of returns (e.g., daily returns).
        :return: A dictionary containing the calculated metrics.
        """
        # Convert returns to a numpy array for efficiency
        returns = df['returns']

        # Calculate annualized return
        annualized_return = np.prod(returns)**(252 / len(returns)) - 1

        # Max Drawdown
        max_drawdown = df['drawdown_pct'].max()

        # Sharpe Ratio
        sharpe_ratio = (np.mean(returns)) / np.std(returns) * np.sqrt(252)

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns)
        sortino_ratio = (np.mean(returns)) / downside_std * np.sqrt(252)

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown)
        
        # VaR and CVaR
        VaR , CVaR = self.monte_carlo_simulation(returns)

        return {
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Annualized Return': annualized_return,
            'VaR': VaR,
            'CVaR': CVaR
        }
        
    def get_financial_metrics(self,print = False):
        metrics = {}
        metrics['Equal Weightage'] = self.calculate_financial_metrics(self.equal_returns.returns)
        metrics['Max Sharpe Ratio'] = self.calculate_financial_metrics(self.max_sharpe_ratio.returns)
        metrics['Max Sortino Ratio'] = self.calculate_financial_metrics(self.max_sortino_ratio.returns)
        metrics['Min Variance'] = self.calculate_financial_metrics(self.min_variance.returns)
        metrics['Min Downside Variance'] = self.calculate_financial_metrics(self.min_downside_var.returns)
        metrics_df = pd.DataFrame(metrics)
        if print:
            plt.table(cellText=metrics_df.values, colLabels=metrics_df.columns, rowLabels=metrics_df.index, loc='center')
        return metrics_df
    
    def monte_carlo_simulation(self,returns, num_simulations=1000, var_level=0.05):
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
        time_horizon = len(returns)
        # Simulate paths
        simulated_paths = np.zeros((num_simulations, time_horizon))
        for i in range(num_simulations):
            random_sample = np.random.choice(returns, size=time_horizon, replace=True)
            simulated_paths[i] = np.cumprod(random_sample + 1)

        plt.plot(simulated_paths.T)
        plt.title('Simulated Returns')
        plt.show()
        # Calculate VaR and CVaR
        sorted_returns = np.sort(simulated_paths[:,-1])
        var = np.percentile(sorted_returns, var_level * 100)
        cvar = sorted_returns[sorted_returns <= var].mean()

        return {
            'VaR': var,
            'CVaR': cvar
        }
        
        