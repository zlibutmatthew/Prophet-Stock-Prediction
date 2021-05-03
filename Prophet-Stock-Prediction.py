# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import matplotlib
from prophet import Prophet
from pandas import to_datetime
from pandas import DataFrame
from prophet import Prophet
from matplotlib import pyplot
from prophet.plot import plot_plotly, plot_components_plotly


class Stock():
    def __init__(self, stock_abbr, start_date=None, end_date=None):
        self.abbr = stock_abbr.upper()
        
        self.df = web.DataReader(self.abbr, data_source='yahoo', start=start_date, end=end_date)
        self.df = self.df.reset_index()
        
        #find the minimum and maximum date
        self.min_date = min(self.df['Date'])
        self.max_date = max(self.df['Date'])
        
        #setting start and end date if None
        if start_date == None:
            start_date = self.min_date
        if end_date == None:
            end_date = self.max_date
        
        #find the minimum and max values, and when they occured
        self.min_price = np.min(self.df['Adj Close'])
        self.max_price = np.max(self.df['Adj Close'])
        
        self.min_price_date = self.df[self.df['Adj Close'] == self.min_price]['Date']
        self.max_price_date = self.df[self.df['Adj Close'] == self.max_price]['Date']
        
        #find starting and most recent price
        self.starting_price = float(self.df.loc[self.df.index[0], 'Adj Close'])
        self.most_recent_price = float(self.df.loc[self.df.index[-1], 'Adj Close'])
        
        print('Stocker initialized', self.abbr, 'from', str(self.min_date)[:10], 'to', str(self.max_date)[:10])
        
    def make_df(self):
        df = self.df
        lst1, lst2 = df['Date'], df['Adj Close']
        df = pd.DataFrame(list(zip(lst1, lst2)), columns = ['ds', 'y'])
        df['ds']= to_datetime(df['ds'])
        self.df = df
        
    def plot_stock(self):
        x = self.df['ds']
        y = self.df['y']
        plt.figure(figsize=(16, 8))
        plt.plot(
            x,
            y,
            color='r')
        plt.grid()
        plt.xlabel("Date")
        plt.ylabel("US $")
        plt.title("%s Stock History" % self.abbr)
        plt.legend(prop={"size": 10})

        plt.show()
        
    def make_model(self, days):
        self.model = Prophet()
        self.model.fit(self.df)
        future = self.model.make_future_dataframe(periods=days)
        forecast = self.model.predict(future)
        self.forecast = forecast
        
    def interactive_plot(self):
        return plot_plotly(self.model, self.forecast)
        
    def plot_model(self):
        fig1 = self.model.plot(self.forecast)
        
    def evaluate_comps(self):
        fig2 = self.model.plot_components(self.forecast)


appl = Stock('aapl')

appl.make_df()

appl.plot_stock()

appl.make_model(days=100)

appl.plot_model()

appl.evaluate_comps()

appl.interactive_plot()
