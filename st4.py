# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:01:50 2024

@author: atom
"""

from twelvedata import TDClient
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st
position = None
entry_price = 0
portfolio_value = []
shares = 0
trades = []

stop_loss = st.slider("Set stop loss percentage:", 0.01, 0.20, 0.07, 0.01)
initial_capital = st.number_input("Set initial capital:", min_value=100, value=470, step=10)
ss = st.number_input("How many strategies need to agree:", min_value=1, value=1, step=1)

cash = initial_capital


class DataFeeder:
    def __init__(self, ticker, api_key, timeframe):
        self.ticker = ticker
        self.api_key = api_key
        self.timeframe = timeframe
        self.td = TDClient(apikey=self.api_key)
        try:
            self.df = self.fetch_ticker()
            print(f"Data fetched for {self.ticker}")
        except Exception as e:
            print("nope", e)

    def fetch_ticker(self):
        ts = self.td.time_series(
            symbol=self.ticker,
            interval=self.timeframe,
            outputsize=5000,
            timezone="America/New_York",
        )
        df = ts.as_pandas()
        df = df.reset_index()
        df.rename(columns={'datetime': 'date'}, inplace=True)
        return df
    
class DataFeederML:
    def __init__(self, ticker, api_key, timeframe, output):
        self.ticker = ticker
        self.api_key = api_key
        self.timeframe = timeframe
        self.td = TDClient(apikey=self.api_key)
        self.out = output
        try:
            self.df = self.fetch_ticker()
            print(f"Data fetched for {self.ticker}")
        except Exception as e:
            print("nope", e)

    def fetch_ticker(self):
        ts = self.td.time_series(
            symbol=self.ticker,
            interval=self.timeframe,
            outputsize=self.out,
            timezone="America/New_York",
        )
        df = ts.as_pandas()
        df = df.reset_index()
        df.rename(columns={'datetime': 'date'}, inplace=True)
        return df

class DataFeeder2:
    def __init__(self, ticker, api_key, timeframe='2min'):
        if timeframe != '2min':
            raise ValueError("This class only supports a 2-minute timeframe.")
        
        self.ticker = ticker
        self.api_key = api_key
        self.timeframe = '1min'  # Use 1-minute intervals to fetch data
        self.td = TDClient(apikey=self.api_key)
        
        try:
            self.df = self.fetch_and_aggregate_data()
            print(f"Data fetched and aggregated for {self.ticker}")
        except Exception as e:
            print("Data fetching failed:", e)

    def fetch_and_aggregate_data(self):
        # Fetch 1-minute interval data
        ts = self.td.time_series(
            symbol=self.ticker,
            interval=self.timeframe,
            outputsize=5000,
            timezone="America/New_York",
        )
        df = ts.as_pandas().reset_index()
        df.rename(columns={'datetime': 'date'}, inplace=True)
        
        # Aggregate the data into 2-minute intervals
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df_agg = df.resample('2T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()

        return df_agg

class Indicators:
    def __init__(self, df):
        self.df = df

    def calculate_sma(self, periods):
        for period in periods:
            self.df[f'SMA_{period}'] = self.df['close'].astype(float).rolling(window=period).mean()
            self.df[f'SMA_{period}'].fillna(method='bfill', inplace=True)

    def calculate_stochastic(self, periodK=12, smoothK=3, periodD=1, periodD1=4):
        # Calculate lowest low and highest high over periodK
        self.df['lowest_low'] = self.df['low'].rolling(window=periodK).min()
        self.df['highest_high'] = self.df['high'].rolling(window=periodK).max()
        
        # Calculate %K
        self.df['%K'] = 100 * (self.df['close'] - self.df['lowest_low']) / (self.df['highest_high'] - self.df['lowest_low'])
        
        # Smooth %K with an SMA over smoothK
        self.df['%K_smooth'] = self.df['%K'].rolling(window=smoothK).mean()
        
        # Calculate %D as the SMA of %K_smooth over periodD
        self.df['%D'] = self.df['%K_smooth'].rolling(window=periodD).mean()
        
        # Calculate %D1 as the SMA of %K_smooth over periodD1
        self.df['%D1'] = self.df['%K_smooth'].rolling(window=periodD1).mean()
        
        # Calculate Double Slow K Denominator (highest - lowest over periodK)
        self.df['DoubleSlowKDen'] = self.df['%K_smooth'].rolling(window=periodK).apply(lambda x: x.max() - x.min(), raw=True)
        
        # Calculate Double Slow D Denominator (highest - lowest over periodK)
        self.df['DoubleSlowDDen'] = self.df['%D'].rolling(window=periodK).apply(lambda x: x.max() - x.min(), raw=True)
        
        # Calculate Double Slow K
        self.df['DoubleSlowK'] = 100 * (self.df['%K_smooth'] - self.df['%K_smooth'].rolling(window=periodK).min()) / self.df['DoubleSlowKDen']
        
        # Calculate Double Slow D
        self.df['%FastD'] = self.df['%D'] 
        
        self.df['DoubleSlowD'] = 100 * (self.df['%D'] - self.df['%D'].rolling(window=periodK).min()) / self.df['DoubleSlowDDen']
        self.df.fillna(method='bfill', inplace=True) 
        plt.figure(figsize=(12,10))
        plt.plot( self.df.index , self.df['%FastD'] )
        plt.figure(figsize=(12,11))
        plt.plot( self.df.index , self.df['DoubleSlowK'] )

class MABStrategy(Indicators):
    def __init__(self, df):
        super().__init__(df)
    
    def mab_strategy(self, period=5):
  
        # Create a new DataFrame to store the strategy signals
        strategy_signals = pd.DataFrame(index=self.df.index)
        strategy_signals['MAB_Signal'] = 0  # 1 for long, -1 for short, 0 for no signal

        # Iterate over the DataFrame rows
        for i in range(period, len(self.df)):
            # Check if the price bar touches or moves within $0.10 of any one of the 21, 50, or 200 moving averages
            if self.check_price_bar_touches_moving_averages(i):
                # Check if the price bar moves in the opposite direction
                if self.check_price_bar_moves_opposite_direction(i):
                    # Check if the Fast D is in the >80 zone or coming from that zone and decreasing
                    if self.check_fast_d_condition(i):
                        # Check if the 21 moving average is below the 50 moving average
                        if self.df['SMA_21'][i] < self.df['SMA_50'][i]:
                            # Check for at least 1 additional strategy pattern found
                            # This is a placeholder for the additional pattern check
                            if self.additional_pattern_check(i):
                                strategy_signals['MAB_Signal'][i] = 1  # Long signal
                        elif self.df['SMA_21'][i] > self.df['SMA_200'][i]:
                            # Check for at least 1 additional strategy pattern found
                            # This is a placeholder for the additional pattern check
                            if self.additional_pattern_check(i):
                                strategy_signals['MAB_Signal'][i] = -1  # Short signal
    
        return strategy_signals
        
    
    def check_price_bar_touches_moving_averages(self, i):
        """
        Checks if the price bar touches or moves within $0.10 of any one of the 21, 50, or 200 moving averages.
        """
        return (abs(self.df['close'][i] - self.df['SMA_21'][i]) <= 0.10 or
                abs(self.df['close'][i] - self.df['SMA_50'][i]) <= 0.10 or
                abs(self.df['close'][i] - self.df['SMA_200'][i]) <= 0.10)
    
    def check_price_bar_moves_opposite_direction(self, i):
        """
        Checks if the price bar moves in the opposite direction.
        """
        return (self.df['close'][i] > self.df['SMA_21'][i] and
                self.df['close'][i-1] < self.df['SMA_21'][i-1] )
    
    def check_fast_d_condition(self, i):
        """
        Checks if the Fast D is in the >80 zone or coming from that zone and decreasing.
        """
        return (self.df['%D'][i] > 80 or
                (self.df['%D'][i] > 80 and 
                 self.df['%D'][i] < self.df['%D'][i-1]))
    
    def additional_pattern_check(self, i):
        """
        Placeholder function for checking additional strategy patterns.

        Parameters:
        - df: DataFrame containing the price data and indicators.
        - i: The current row index.

        Returns:
        - True if at least 1 additional pattern is found, False otherwise.
        """
        # This function should contain logic to check for additional patterns
        return True  # Placeholder: replace with actual condition

   
    def run_strategy(self):
        # Calculate the indicators
        self.calculate_sma([21, 50, 200])
        self.calculate_stochastic()
    
        # Apply the MAB strategy
        strategy_signals = self.mab_strategy()
    
        return strategy_signals

class SSGStrategy(Indicators):
    def __init__(self, df):
        super().__init__(df)
    
    def ssg_strategy(self):
        # Create a new DataFrame to store the strategy signals
        strategy_signals = pd.DataFrame(index=self.df.index)
        strategy_signals['SSG_Signal'] = 0  # 1 for long, -1 for short, 0 for no signal

        # Iterate over the DataFrame rows
        for i in range(2, len(self.df)):
            # Check if the Fast D is in the >80 zone or coming from that zone and decreasing
            if self.df['%FastD'][i] > 80 or (self.df['%FastD'][i] > 80 and self.df['%FastD'][i-1] > 80 and self.df['%FastD'][i-2] > 80 and self.df['%FastD'][i] < self.df['%FastD'][i-1]):
                # Check if the Double Slow K is at 100 and stays there while Fast D is < 80
                if self.df['DoubleSlowK'][i] > 80 or self.df['DoubleSlowK'][i-1] > 80 or self.df['DoubleSlowK'][i-2] > 80 and self.df['%FastD'][i] < 80: #THIS CAUSES EMPTY SSG SIGNALS
                    strategy_signals['SSG_Signal'][i] = -1  # Short signal
            # Check if the Fast D is in the <20 zone or coming from that zone and increasing
            elif self.df['%FastD'][i] < 20 or (self.df['%FastD'][i] < 20 and self.df['%FastD'][i-1] < 20 and self.df['%FastD'][i-2] < 20 and self.df['%FastD'][i] > self.df['%FastD'][i-1]):
                # Check if the Double Slow K is at 0 and stays there while Fast D is > 20
                if self.df['DoubleSlowK'][i] < 20 or self.df['DoubleSlowK'][i-1] < 20 or self.df['DoubleSlowK'][i-2] < 20 and self.df['%FastD'][i] > 20: #THIS CAUSES EMPTY SSG SIGNALS
                    strategy_signals['SSG_Signal'][i] = 1  # Long signal

        return strategy_signals
    
    def run_strategy(self):
        # Calculate the indicators
      
    
        # Apply the SSG strategy
        strategy_signals = self.ssg_strategy()
    
        return strategy_signals
    
    
class HPatternStrategy(Indicators):
    def __init__(self, df):
        super().__init__(df)
    def check_fast_d_condition(self, i):
        """
        Checks if the Fast D is in the >80 zone or coming from that zone and decreasing.
        """
        return (self.df['%D'][i] > 80 or
                (self.df['%D'][i] > 80 and self.df['%D'][i-1] > 80 and self.df['%D'][i-2] > 80 and
                 self.df['%D'][i] < self.df['%D'][i-1]))
   
    def h_pattern(self):
        # Create a new DataFrame to store the strategy signals
        strategy_signals = pd.DataFrame(index=self.df.index)
        strategy_signals['HPattern_Signal'] = 0  # 1 for long, -1 for short, 0 for no signal

        # Iterate over the DataFrame rows
        for i in range(2, len(self.df)):
            # Check if the price hits a high point (back leg)
            if self.df['high'][i] > self.df['high'][i-1]:
                # Check if the price drops for a while then goes back up close to the same level (front leg)
                if self.df['low'][i] < self.df['low'][i-1]  and self.df['close'][i] > self.df['close'][i-1]:
                        # Check if the Fast D is in the >80 zone or coming from that zone and decreasing
                    if self.check_fast_d_condition(i):
                        strategy_signals['HPattern_Signal'][i] = -1  # Short signal
                # Check if the price hits a low point (back leg)
                elif self.df['low'][i] < self.df['low'][i-1] :
                    # Check if the price goes up for a while then goes back down close to the same level (front leg)
                    if self.df['high'][i] > self.df['high'][i-1]  and self.df['close'][i] < self.df['close'][i-1] :
                            # Check if the Fast D is in the <20 zone or coming from that zone and increasing
                        if self.check_fast_d_condition(i):
                            strategy_signals['HPattern_Signal'][i] = 1  # Long signal

        return strategy_signals
    
    def run_strategy(self):
        # Calculate the indicators
      
    
        # Apply the H Pattern strategy
        strategy_signals = self.h_pattern()
    
        return strategy_signals
  
class NewStrategy(Indicators):
    def __init__(self, df):
        super().__init__(df)
        
    def new_strategy(self):
        # Create a new DataFrame to store the strategy signals
        strategy_signals = pd.DataFrame(index=self.df.index)
        strategy_signals['New_Signal'] = 0  # 1 for long, -1 for short, 0 for no signal

        # Iterate over the DataFrame rows
        for i in range(3, len(self.df)):
            # Factor 3: Highs and Lows Comparison
            if self.check_highs_before_pattern(i):
                if self.check_horizontal_highs(i):
                    if self.check_three_bars_highs(i):
                        if self.oscillator_condition(i, direction='short'):
                            strategy_signals['New_Signal'][i] = -1  # Short signal

            elif self.check_lows_before_pattern(i):
                if self.check_horizontal_lows(i):
                    if self.check_three_bars_lows(i):
                        if self.oscillator_condition(i, direction='long'):
                            strategy_signals['New_Signal'][i] = 1  # Long signal

        return strategy_signals

    def check_highs_before_pattern(self, i):
        """
        Checks if the highs of each price bar before the pattern are higher than the previous one.
        """
        return (self.df['high'][i] > self.df['high'][i-1] or self.df['high'][i] > self.df['high'][i-2])

    def check_lows_before_pattern(self, i):
        """
        Checks if the lows of each price bar before the pattern are lower than the previous one.
        """
        return (self.df['low'][i] < self.df['low'][i-1] or self.df['low'][i] < self.df['low'][i-2])

    def check_horizontal_highs(self, i):
        """
        Checks if at least 3 price bars on the 5-minute chart have highs within $0.10 of each other.
        """
        return (abs(self.df['high'][i] - self.df['high'][i-1]) <= 0.10 and
                abs(self.df['high'][i-1] - self.df['high'][i-2]) <= 0.10 and
                abs(self.df['high'][i-2] - self.df['high'][i-3]) <= 0.10)

    def check_horizontal_lows(self, i):
        """
        Checks if at least 3 price bars on the 5-minute chart have lows within $0.10 of each other.
        """
        return (abs(self.df['low'][i] - self.df['low'][i-1]) <= 0.10 and
                abs(self.df['low'][i-1] - self.df['low'][i-2]) <= 0.10 and
                abs(self.df['low'][i-2] - self.df['low'][i-3]) <= 0.10)

    def check_three_bars_highs(self, i):
        """
        Checks if the first bar where the high price is not higher than the previous bar is the first of the 3 required bars.
        """
        return (self.df['high'][i-2] <= self.df['high'][i-3] and
                self.df['high'][i-1] <= self.df['high'][i-2] and
                abs(self.df['high'][i-1] - self.df['high'][i-2]) <= 0.10)

    def check_three_bars_lows(self, i):
        """
        Checks if the first bar where the low price is not lower than the previous bar is the first of the 3 required bars.
        """
        return (self.df['low'][i-2] >= self.df['low'][i-3] and
                self.df['low'][i-1] >= self.df['low'][i-2] and
                abs(self.df['low'][i-1] - self.df['low'][i-2]) <= 0.10)

    def oscillator_condition(self, i, direction):
        """
        Checks if the oscillator is >80 (short) or <20 (long)
        """
        if direction == 'short':
            return (self.df['%FastD'][i] > 80 or self.df['%FastD'][i-1] > 80 or self.df['%FastD'][i-2] > 80)
        elif direction == 'long':
            return (self.df['%FastD'][i] < 20 or self.df['%FastD'][i-1] < 20 or self.df['%FastD'][i-2] < 20)

    def run_strategy(self):
        # Calculate the indicators
        self.calculate_sma([21, 50, 200])
        self.calculate_stochastic()

        # Apply the new strategy
        strategy_signals = self.new_strategy()

        return strategy_signals


st.title("Backtesting App")


if st.button("Run Backtest"):
       with st.spinner('Running backtest...'):
            data_feeder = DataFeeder('QQQ', '64e5d79c83ed4ff49b32db1b1a60627d', '1min')
            mab_strategy = MABStrategy(data_feeder.df)
            mab_signals = mab_strategy.run_strategy()
 
 
            ssg_strategy = SSGStrategy(data_feeder.df)
            ssg_signals = ssg_strategy.run_strategy()
 
            #
 
            # Create HPatternStrategy object and run strategy
            h_pattern_strategy = HPatternStrategy(data_feeder.df)
            h_pattern_signals = h_pattern_strategy.run_strategy()
 
            # Printing non-zero signals
            data_feeder5 = DataFeeder('QQQ', '64e5d79c83ed4ff49b32db1b1a60627d', '5min')
 
            # Create a Trip5Strategy object
            trip5_strategy = NewStrategy(data_feeder5.df)
            trip5_signals = trip5_strategy.run_strategy()
 
             
            # Combine signals
            signals = pd.concat( [data_feeder.df['date'] , mab_signals, ssg_signals, h_pattern_signals, trip5_signals], axis=1)
            signals.columns = ['date' , 'MAB_Signal', 'SSG_Signal', 'HPattern_Signal', 'Trip5_Signal']
 
            # Create combined signals
            signals['Buy_Signal'] = (signals[['MAB_Signal', 'SSG_Signal', 'HPattern_Signal', 'Trip5_Signal']] == 1).sum(axis=1) >= ss
            signals['Sell_Signal'] = (signals[['MAB_Signal', 'SSG_Signal', 'HPattern_Signal', 'Trip5_Signal']] == -1).sum(axis=1) >= ss

            for index, row in signals.iterrows():
                current_price = data_feeder.df.loc[index, 'close']
                
                if row['Buy_Signal'] and position is None and row['Sell_Signal'] == False:
                    position = 'Long'
                    entry_price = current_price
                    shares = cash // current_price
                    cash -= shares * current_price
                    trades.append({'Date': index, 'Action': 'Buy', 'Price': entry_price, 'Shares': shares})
                
                elif row['Sell_Signal'] and position is None and row['Buy_Signal'] == False:
                    position = 'Short'
                    entry_price = current_price
                    shares = cash // current_price
                    cash += shares * current_price
                    trades.append({'Date': index, 'Action': 'Sell Short', 'Price': entry_price, 'Shares': shares})
            
                elif row['Buy_Signal'] and position == 'Short':
                    exit_price = current_price
                    cash -= shares * exit_price
                    position = None
                    trades.append({'Date': index, 'Action': 'Buy to Cover', 'Price': exit_price, 'Shares': shares})
                    shares = 0
            
                elif row['Sell_Signal'] and position == 'Long':
                    exit_price = current_price
                    cash += shares * exit_price
                    position = None
                    trades.append({'Date': index, 'Action': 'Sell', 'Price': exit_price, 'Shares': shares})
                    shares = 0
                
                elif position == 'Long':
                    if current_price <= entry_price * (1 - stop_loss):
                        cash += shares * current_price
                        position = None
                        trades.append({'Date': index, 'Action': 'Sell (Stop-Loss)', 'Price': current_price, 'Shares': shares})
                        shares = 0
            
                elif position == 'Short':
                    if current_price >= entry_price * (1 + stop_loss):
                        cash -= shares * current_price
                        position = None
                        trades.append({'Date': index, 'Action': 'Buy to Cover (Stop-Loss)', 'Price': current_price, 'Shares': shares})
                        shares = 0
            
                # Calculate portfolio value
                if position == 'Long':
                    portfolio_value.append(cash + (shares * current_price))
                elif position == 'Short':
                    portfolio_value.append(cash - (shares * current_price))
                else:
                    portfolio_value.append(cash)
            # Convert portfolio value list to DataFrame for performance metrics calculation
            portfolio_df = pd.DataFrame(portfolio_value, index=signals.index, columns=['Portfolio_Value'])
            portfolio_df['Returns'] = portfolio_df['Portfolio_Value'].pct_change().dropna()
            
            # Calculate basic performance metrics
            total_return = (portfolio_df['Portfolio_Value'].iloc[-1] / portfolio_df['Portfolio_Value'].iloc[0]) - 1

            sharpe_ratio = (portfolio_df['Returns'].mean() / portfolio_df['Returns'].std()) * np.sqrt(252)
            max_drawdown = (portfolio_df['Portfolio_Value'].cummax() - portfolio_df['Portfolio_Value']).max()
            st.success('Backtest completed!')
            st.write(f"Total Return: {total_return:.2%}")

            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            st.write(f"Max Drawdown: {max_drawdown:.2f}")
            
            # Plotting Portfolio Value and Trades
            plt.figure(figsize=(14, 7))
            
            # Plot Portfolio Value
            plt.plot(portfolio_df.index, portfolio_df['Portfolio_Value'], label='Portfolio Value', color='blue')
            plt.plot(portfolio_df.index, data_feeder.df['close'], label='Price', color='red')
            st.pyplot(plt)
            
            # Plot Trades
            buy_trades = [trade for trade in trades if trade['Action'].startswith('Buy')]
            sell_trades = [trade for trade in trades if trade['Action'].startswith('Sell')]
            stop_loss_trades = [trade for trade in trades if trade['Action'].startswith('Sell (Stop-Loss)')]
            plt.show()
            
            printsignalsM = np.where(signals['MAB_Signal'] == 1, "Buy",  np.where(signals['MAB_Signal']==-1 , "Sell" , 0))
            #printsignalsMd = pd.concat( signals['date'] ,  printsignalsM  ,axis = 1)
            printsignalsS = np.where(signals['SSG_Signal'] == 1, "Buy", np.where(signals['SSG_Signal'] == -1, "Sell",  0))
            #printsignalsSd = pd.concat( signals['date'] ,  printsignalsS  ,axis = 1)

            printsignalsH = np.where(signals['HPattern_Signal'] == 1, "Buy", np.where(signals['HPattern_Signal'] == -1, "Sell" , 0 ))
            #printsignalsHd = pd.concat( signals['date'] ,  printsignalsH  ,axis = 1)

            printsignalsT = np.where(signals['Trip5_Signal'] == 1, "Buy", np.where(signals['Trip5_Signal'] == -1, "Sell" , 0 ))
            #printsignalsTd = pd.concat( signals['date'] ,  printsignalsT  ,axis = 1)

            signal_series = {
            'MAB_Signal': pd.Series(printsignalsM, index=data_feeder.df.index),
            'SSG_Signal': pd.Series(printsignalsS, index=data_feeder.df.index),
            'HPattern_Signal': pd.Series(printsignalsH, index=data_feeder.df.index),
            'Trip5_Signal': pd.Series(printsignalsT, index=data_feeder.df.index)
            }

# Create a DataFrame from the signals
            signals_df = pd.DataFrame(signal_series)
            all_sig = pd.DataFrame(index = data_feeder.df.index)
            all_sig = pd.concat( [data_feeder.df['date']  , signals_df ]   , axis=1)
            
            
            
            # For the trades
            st.subheader("Trades")
            for trade in trades:
                st.write(f"{str(data_feeder.df.loc[trade['Date'], 'date'])} {trade['Action']} {str(trade['Price'])}")
            
            # For the signals
            st.subheader("Signals")
            st.write(all_sig)
            
            
            # Print trades
            for trade in trades:
                print(str(data_feeder.df.loc[trade['Date'], 'date']) + " " + trade['Action'] + " " + str(trade['Price']))


st.write("For Prediction " )
ticker = st.text_input("Enter stock ticker:", "QQQ")
api_key = st.text_input("Enter API key:", "64e5d79c83ed4ff49b32db1b1a60627d")
interval = st.selectbox("Select interval:", ["1min", "5min", "15min", "30min", "60min", "daily"])
size = st.number_input("Set size: (How many vals to pull )", min_value=210, value=210, step=10)


if st.button("Run Analysis"):
    
        # User inputs


    # Create DataFeeder object
    data_feeder = DataFeederML(ticker, api_key, interval,size)
    
    # Calculate indicators and strategies
    mab_strategy = MABStrategy(data_feeder.df)
    mab_signals = mab_strategy.run_strategy()
    
    ssg_strategy = SSGStrategy(data_feeder.df)
    ssg_signals = ssg_strategy.run_strategy()
    
    h_pattern_strategy = HPatternStrategy(data_feeder.df)
    h_pattern_signals = h_pattern_strategy.run_strategy()
    
    data_feeder5 = DataFeederML(ticker, api_key, '5min', size)
    trip5_strategy = NewStrategy(data_feeder5.df)
    trip5_signals = trip5_strategy.run_strategy()

    signals = pd.concat( [mab_signals, ssg_signals, h_pattern_signals, trip5_signals], axis=1)
    signals.columns = [ 'MAB_Signal', 'SSG_Signal', 'HPattern_Signal', 'Trip5_Signal']

    # Create combined signals
    #signals['Buy_Signal'] = (signals[['MAB_Signal', 'SSG_Signal', 'HPattern_Signal', 'Trip5_Signal']] == 1).sum(axis=1) >= 1
    #signals['Sell_Signal'] = (signals[['MAB_Signal', 'SSG_Signal', 'HPattern_Signal', 'Trip5_Signal']] == -1).sum(axis=1) >= 1

    

    s = signals[['MAB_Signal', 'SSG_Signal', 'HPattern_Signal', 'Trip5_Signal']]
    X = data_feeder.df[['open' , 'close' , 'low' ,  'high' , 'SMA_21','SMA_50','SMA_200', '%FastD' ,'DoubleSlowK']]
    models = {}
    
    # Signals to loop through
    signals_list = ['MAB_Signal', 'SSG_Signal', 'HPattern_Signal', 'Trip5_Signal']
    
    # Load each model
    for signal in signals_list:
        model = joblib.load(f'{signal}_model.pkl')
        models[signal] = model
    # Initialize a DataFrame to store the predictions
    predictions = {}
    
    # Loop through each signal to make predictions
    for signal in signals_list:
        model = models[signal]
        predictions[signal] = model.predict(X)
    
    # Convert predictions dictionary to DataFrame
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.apply(lambda x: np.where(x == 1, 'Buy', np.where(x == -1, 'Sell', 'No Signal')))
    st.write("Machine Learning Predictions : ->")

    st.write(predictions_df)
    st.write("Calculated Signals:")
    st.write(signals)

    st.write("Latest Values (Machine Learning Predictions) : ->")
    # Display the predictions
    st.write(predictions_df.tail())
    st.write("Latest Values (Calculated Signals) : ->")

    st.write(signals.tail())
