# https://github.com/Chiu-Huang/Simple_stock_analysis/blob/master/Basic_Interactive_Stock_Analysis_Template.ipynb
# The source codes have been modified to include Plotly plots

# Data and array manipulation
import pandas as pd
import numpy as np
from IPython.display import display, HTML
from warnings import warn

# Datetime manipulation
import datetime as dt

# Plotting and Visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import chart_studio.plotly as py
import cufflinks as cf

# Fetching stock data from web
import pandas_datareader.data as web
import yfinance as yf

# Interactive charts
from ipywidgets import interact, fixed, IntSlider
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Portfolio optimization
import pypfopt
from pypfopt import risk_models
from pypfopt import plotting
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import discrete_allocation

# Options
cached_data = {} 
init_notebook_mode(connected = True)
cf.go_offline()
pd.options.display.float_format = '{:,.4f}'.format

# Default settings
default_yaxis = dict(showgrid = False,
                     zeroline = False,
                     showline = False,
                     showticklabels = True)
default_RgSlct = dict(buttons = list([dict(count = 1, label = "1 Month", step = "month", stepmode = "backward"),
                                      dict(count = 6, label = "6 Months", step = "month", stepmode = "backward"),
                                      dict(count = 1, label = "1 Year", step = "year", stepmode = "backward"),
                                      dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
                                      dict(label = "All Data", step = "all")]))

cached_data = {}

# Functions for back testing
def stock_hist(symbol, start = None, end = None, cached_data = cached_data):
    '''Convenience function to get cached data '''
    if not symbol in cached_data:
        cached_data[symbol] = yf.download(symbol)
        print(F'Loaded {symbol} num values = {len(cached_data[symbol])}')
    return cached_data[symbol]

def backtest(symbols, start_day = 0, time_horizon = 365, **active):
    
    # Fetching stock data
    filtered = [symbol for symbol in symbols if active.get(symbol, True)]
    prices = pd.concat((stock_hist(symbol)['Adj Close'] for symbol in filtered), axis = 1, keys = filtered).dropna(axis = 0)
    
    # Setting dates
    start_dates = prices.index[0] + dt.timedelta(days = start_day)
    end_dates = start_dates + dt.timedelta(days = time_horizon)
    
    # calculations
    prices = prices.loc[start_dates:end_dates] 
    unit_pos = prices / prices.iloc[0,:]
    unit_pos['Portfolio'] = unit_pos.sum(axis = 1) / unit_pos.shape[1]

    # Extra information
    print(f'backtest from {start_dates} to {end_dates}')
    
    # Plotting
    fig = go.Figure()
    for _ in unit_pos.columns[:-1]:
        fig.add_trace(go.Scatter(x = unit_pos.index, 
                                 y = unit_pos[_], 
                                 mode = 'lines', 
                                 name = _, 
                                 line = dict(color = '#ff0000', width = 1)))

    fig.add_trace(go.Scatter(x = unit_pos.index, 
                             y = unit_pos.Portfolio, 
                             mode = 'lines', 
                             name = 'Equally Weighted Portfolio', 
                             line = dict(color = '#0f0f0f', width = 2)))
    # Updating layout to reduce clutter
    fig.update_layout(
        title = f'Backtest for the training window',
        xaxis_title = 'Training window',
        yaxis_title = 'Cumulative returns',
        yaxis = dict(showgrid = False,
                     zeroline = False,
                     showline = False,
                     showticklabels = True),
        autosize = False,
        width = 1000,
        height = 600,
        plot_bgcolor = 'white')

    fig.add_hline(y = 1, 
                  line_dash = "dash",
                  opacity = 0.5,
                  annotation_text="Break Even", 
                  annotation_position="bottom right")

    # Creating slider
    fig.update_xaxes(
        rangeselector = dict(buttons = list([dict(count = 1, label = "1 Month", step = "month", stepmode = "backward"),
                                             dict(count = 6, label = "6 Months", step = "month", stepmode = "backward"),
                                             dict(count = 1, label = "1 Year", step = "year", stepmode = "backward"),
                                             dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
                                             dict(label = "All Data", step = "all")])))
    fig.show()    
    
# Function for fetching data conveniently
def fnc_fetchdata(t, s = None, e = None, c = 'all'):
    '''Convenience function to fetch data '''
    colnames = ['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']
    
    # Fetching data from Yahoo! Finance
    df = yf.download(t, start = s, end = e).stack().reset_index().sort_values('level_1')
    df.columns = ['Date', 'Ticker', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    # Extracting a certain column
    if c != 'all' and c not in colnames:
        return(r"'column' parameter should be either 'all' or a specified column in 'High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume'")
    elif c == 'all':
        return(df.sort_values(by = ['Ticker', 'Date']).reset_index().drop(labels = 'index', axis = 1))
    else:
        c_show = ['Date', c, 'Ticker']
        print(F'Extracting column {c}')
        df_pivot = df.loc[:, c_show].pivot(index = 'Date', columns = 'Ticker', values = c)
        return(df_pivot)
    
# summary class
class ef_summary():
    
    def __init__(self, ef, price = None, fund = None, type = 'minimum volatility'):
        self.type = type
        self.ef = ef
        # check optimizing type
        if self.type == 'minimum volatility':
            self.op = ef.min_volatility()
        elif self.type == 'max sharpe':
            self.op = ef.max_sharpe()
        else:
            self.op = 'invalid'
        # calculate weights
        if self.op == 'invalid':
            warn('invalid type, should be "minimum volatility" or "max sharpe". Portfolio weights will not be calculated!')
        else:
            self.weight = pd.Series(self.ef.clean_weights())
        # check price
        if price is None:
            warn('No "price" data is given, discrete allocation will not be computed')
            self.price = price
        else:
            self.price = price
        # check fund
        if fund is None:
            warn('No "fund" data is given, discrete allocation will not be computed')
            self.fund = fund
        else:
            self.fund = fund
            
        if self.price is None or self.fund is None:
            warn('either "price" or "fund" data has not been given, discrete allocation cannot be computed')
        else:
            self.da = discrete_allocation.DiscreteAllocation(self.op, self.price, total_portfolio_value = self.fund)
            self.allocation, self.leftover = self.da.lp_portfolio()
    
    def display_weight(self):
        self.weight.name = 'Weights'
        self.html_weight = HTML(self.weight.to_frame().to_html())
        display(self.html_weight)
    
    def display_price(self):
        if self.price is None:
            warn('no "price" data is given')
        else:
            self.html_price = HTML(self.price.to_frame().to_html())
            display(self.html_price)
    
    def optimization_result(self):
        print(f'Portfolio weights for {self.type} optimization:\n')
        self.display_weight()
        print(f'\nPerformance details:\n')
        print(f'Portfolio performance: {self.ef.portfolio_performance(verbose = True)}')
    
    def visualize(self):
        fig = px.bar(self.weight,
                     labels = {'index': '', 'value': 'Proportion'},
                     title = f'Portfolio weights for {self.type} portfolio',
                     text_auto = '.3f',
                     width = 600,
                     height = 600,
                     orientation = 'h')
        fig.update_layout(showlegend = False, autosize = True, plot_bgcolor = 'white')
        fig.show()
    
    def discrete_allocation(self):
        if self.price is None or self.fund is None:
            warn('either "price" or "fund" data has not been given, discrete allocation cannot be computed')
        else:       
            print(f'The latest prices are:\n ')
            self.display_price()
            print(f'\n')
            print(f'Discrete allocation performed with ${self.leftover:.2f} leftover as follows\n\n{self.allocation}\n')
            print(f'The detailed proportions are as follows: \n')
            for _ in self.allocation.keys():
                print(f'Stock: {_}, value = {self.allocation[_] * self.price[_] / self.fund:.4f}')
    
    def summary(self):
        print(f'OPTIMIZATION RESULTS FOR {self.type.upper()} CONDITIONS:\n')
        print(f'-----------------------------------------------------------------------')
        print(f'1. Optimization results:')
        print(f'-----------------------------------------------------------------------\n')
        self.optimization_result()
        print(f'\n')
        print(f'-----------------------------------------------------------------------')
        print(f'2. Weights visualization:')
        print(f'-----------------------------------------------------------------------\n')
        self.visualize()
        print(f'\n')
        print(f'-----------------------------------------------------------------------')
        print(f'3. Discrete allocation:')
        print(f'-----------------------------------------------------------------------\n')
        self.discrete_allocation()


# Portfolio Analysis
def fnc_value(portfolio, master_df, start, end = None):
    all_prices = master_df[['Date', 'Ticker', 'Adj Close']].set_index('Date')
    all_prices['qty'] = all_prices.Ticker.map(portfolio)
    all_prices['PORTFOLIO'] = all_prices['qty'] * all_prices['Adj Close']
    if end is None:
        portfolio = all_prices[all_prices.index >= start].groupby('Date')['PORTFOLIO'].sum()
    else:
        portfolio = all_prices.loc[(all_prices.index >= start) & (all_prices.index <= end)].groupby('Date')['PORTFOLIO'].sum()
    return portfolio

def forwardtest(symbols, fund, start_train, end_train, start_test, end_test = None, **active):
    
    # Fetching stock data
    filtered = [symbol for symbol in symbols if active.get(symbol, True)]
    prices = pd.concat((stock_hist(symbol)['Adj Close'] for symbol in filtered), axis = 1, keys = filtered).dropna(axis = 0)
       
    # calculations
    prices_train = prices.loc[start_train:end_train] 
    
    # Stock statistics
    mu = expected_returns.mean_historical_return(prices_train.loc[start_train:end_train, :])
    S = risk_models.CovarianceShrinkage(prices_train.loc[start_train:end_train, :]).ledoit_wolf()

    # Current stock prices
    current_prices = discrete_allocation.get_latest_prices(prices_train.loc[start_train:end_train, :])
    
    # Calculating ef
    ef_minvol = ef_summary(EfficientFrontier(mu, S, weight_bounds = (-1, 1)), price = current_prices, fund = fund, type = 'minimum volatility')
    ef_minvol_ns = ef_summary(EfficientFrontier(mu, S), price = current_prices, fund = fund, type = 'minimum volatility')
    ef_maxsharpe = ef_summary(EfficientFrontier(mu, S, weight_bounds = (-1, 1)), price = current_prices, fund = fund, type = 'max sharpe')
    ef_maxsharpe_ns = ef_summary(EfficientFrontier(mu, S), price = current_prices, fund = fund, type = 'max sharpe')
       
    # Taking portfolio information from the summary objects
    portfolio_minvol = ef_minvol.allocation
    portfolio_minvol_ns = ef_minvol_ns.allocation
    portfolio_maxsharpe = ef_maxsharpe.allocation
    portfolio_maxsharpe_ns = ef_maxsharpe_ns.allocation
    
    # master data
    master_df = fnc_fetchdata(t = filtered)
    
    # Calculating portfolio prices for forward test
    s_minvol_price = fnc_value(portfolio = portfolio_minvol, master_df = master_df, start = start_test, end = end_test)
    s_minvolns_price = fnc_value(portfolio = portfolio_minvol_ns, master_df = master_df, start = start_test, end = end_test)
    s_maxsharpe_price = fnc_value(portfolio = portfolio_maxsharpe, master_df = master_df, start = start_test, end = end_test)
    s_maxsharpens_price = fnc_value(portfolio = portfolio_maxsharpe_ns, master_df = master_df, start = start_test, end = end_test)
    
    # Testing prices
    if end_test is None:
        prices_test = prices[prices.index >= start_test]
    else:
        prices_test = prices[prices.index >= start_test and prices.index <= end_test]
        
    prices_test['Min Volatility'] = s_minvol_price
    prices_test['Min Volatility NS'] = s_minvolns_price
    prices_test['Max Sharpe'] = s_maxsharpe_price
    prices_test['Max Sharpe NS'] = s_maxsharpens_price
    
    # cumulative returns for the testing period
    unit_pos = prices_test / prices_test.iloc[0,:]
    
    # Extra information
    if end_test is None:
        print(f'Forwardtest from {start_test} to today')
    else:
        print(f'Forwardtest from {start_test} to {end_test}')
    
    print(f'Portfolios constructed from data collected from {start_train} to {end_train}')
    
    # Printing portfolio discrete allocation
    print(f'1. The minimum volatility portfolio is: {portfolio_minvol}')
    print(f'2. The minimum volatility no short portfolio is: {portfolio_minvol_ns}')
    print(f'3. The maximum Sharpe ratio portfolio is: {portfolio_maxsharpe}')
    print(f'4. The maximum Sharpe ratio no short portfolio is: {portfolio_maxsharpe_ns}')
    
    # Plotting
    fig = go.Figure()
    for _ in unit_pos.columns[:-4]:
        fig.add_trace(go.Scatter(x = unit_pos.index, 
                                 y = unit_pos[_], 
                                 mode = 'lines', 
                                 name = _, 
                                 line = dict(color = '#b4cbcb', width = 1)))

    fig.add_trace(go.Scatter(x = unit_pos.index, 
                             y = unit_pos['Min Volatility'], 
                             mode = 'lines', 
                             name = 'Min Volatility Portfolio', 
                             line = dict(color = '#ff0000', width = 2)))
    
    fig.add_trace(go.Scatter(x = unit_pos.index, 
                             y = unit_pos['Min Volatility NS'], 
                             mode = 'lines', 
                             name = 'Min Volatility no short Portfolio', 
                             line = dict(color = '#0066ff', width = 2)))
    
    fig.add_trace(go.Scatter(x = unit_pos.index, 
                             y = unit_pos['Max Sharpe'], 
                             mode = 'lines', 
                             name = 'Max Sharpe Portfolio', 
                             line = dict(color = '#9966ff', width = 2)))
    
    fig.add_trace(go.Scatter(x = unit_pos.index, 
                             y = unit_pos['Max Sharpe NS'], 
                             mode = 'lines', 
                             name = 'Max Sharpe no short Portfolio', 
                             line = dict(color = '#cc6699', width = 2)))
    
    # Updating layout to reduce clutter
    fig.update_layout(
        title = f'Forwardtest for the testing window',
        xaxis_title = 'Testing window',
        yaxis_title = 'Cumulative returns',
        yaxis = dict(showgrid = False,
                     zeroline = False,
                     showline = False,
                     showticklabels = True),
        autosize = False,
        width = 1000,
        height = 600,
        plot_bgcolor = 'white')

    fig.add_hline(y = 1, 
                  line_dash = "dash",
                  opacity = 0.5,
                  annotation_text = "Break Even", 
                  annotation_position = "bottom right")

    # Creating slider
    fig.update_xaxes(
        rangeselector = dict(buttons = list([dict(count = 1, label = "1 Month", step = "month", stepmode = "backward"),
                                             dict(count = 6, label = "6 Months", step = "month", stepmode = "backward"),
                                             dict(count = 1, label = "1 Year", step = "year", stepmode = "backward"),
                                             dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
                                             dict(label = "All Data", step = "all")])))
    fig.show()    