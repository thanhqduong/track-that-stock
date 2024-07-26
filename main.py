import pandas as pd
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

import math
from scipy.stats import norm

import mysql.connector
from mysql.connector import Error

# --- Page Configuration ---
st.set_page_config(
    page_title="S&P Tracker",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)


class DistanceStrategy:
    """
    Class for creation of trading signals following the strategy by Gatev, E., Goetzmann, W. N., and Rouwenhorst, K. G.
    in "Pairs trading:  Performance of a relative-value arbitrage rule." (2006)
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615.
    """

    def __init__(self):
        """
        Initialize Distance strategy.
        """

        # Internal parameters
        self.min_normalize = None  # Minimum values for each price series used for normalization
        self.max_normalize = None  # Maximum values for each price series used for normalization
        self.pairs = None  # Created pairs after the form_pairs stage
        self.train_std = None  # Historical volatility for each chosen pair portfolio
        self.normalized_data = None  # Normalized test dataset
        self.portfolios = None  # Pair portfolios composed from test dataset
        self.train_portfolio = None  # Pair portfolios composed from train dataset
        self.trading_signals = None  # Final trading signals
        self.num_crossing = None  # Number of zero crossings from train dataset

    def form_pairs(self, train_data, method='standard', industry_dict=None, num_top=5, skip_top=0, selection_pool=50,
                   list_names=None):
        """
        Forms pairs based on input training data.

        This method includes procedures from the pairs formation step of the distance strategy.

        First, the input data is being normalized using max and min price values for each series:
        Normalized = (Price - Min(Price)) / (Max(Price) - Min(Price))

        Second, the normalized data is used to find a pair for each element - another series of
        prices that would have a minimum sum of square differences between normalized prices.
        Only unique pairs are picked in this step (pairs ('AA', 'BD') and ('BD', 'AA') are assumed
        to be one pair ('AA', 'BD')).
        During this step, if one decides to match pairs within the same industry group, with the
        industry dictionary given, the sum of square differences is calculated only for the pairs
        of prices within the same industry group.

        Third, based on the desired number of top pairs to chose and the pairs to skip, they are
        taken from the list of created pairs in the previous step. Pairs are sorted so that ones
        with a smaller sum of square distances are placed at the top of the list.

        Finally, the historical volatility for the portfolio of each chosen pair is calculated.
        Portfolio here is the difference of normalized prices of two elements in a pair.
        Historical volatility will later be used in the testing(trading) step of the
        distance strategy. The formula for calculating a portfolio price here:
        Portfolio_price = Normalized_price_A - Normalized_price_B

        Note: The input dataframe to this method should not contain missing values, as observations
        with missing values will be dropped (otherwise elements with fewer observations would
        have smaller distance to all other elements).

        :param train_data: (pd.DataFrame/np.array) Dataframe with training data used to create asset pairs.
        :param num_top: (int) Number of top pairs to use for portfolio formation.
        :param skip_top: (int) Number of first top pairs to skip. For example, use skip_top=10
            if you'd like to take num_top pairs starting from the 10th one.
        :param list_names: (list) List containing names of elements if Numpy array is used as input.
        :param method: (str) Methods to use for sorting pairs [``standard`` by default, ``variance``,
                             ``zero_crossing``].
        :param selection_pool: (int) Number of pairs to use before sorting them with the selection method.
        :param industry_dict: (dict) Dictionary matching ticker to industry group.
        """

        # If np.array given as an input
        if isinstance(train_data, np.ndarray):
            train_data = pd.DataFrame(train_data, columns=list_names)

        # Normalizing input data
        normalized, self.min_normalize, self.max_normalize = self.normalize_prices(train_data)

        # Dropping observations with missing values (for distance calculation)
        normalized = normalized.dropna(axis=0)

        # If industry dictionary is given, pairs are matched within the same industry group
        all_pairs = self.find_pair(normalized, industry_dict)

        # Choosing needed pairs to construct a portfolio
        self.pairs = self.sort_pairs(all_pairs, selection_pool)

        # Calculating historical volatility of pair portfolios (diffs of normalized prices)
        self.train_std = self.find_volatility(normalized, self.pairs)

        # Creating portfolios for pairs chosen in the pairs formation stage with train dataset
        self.train_portfolio = self.find_portfolios(normalized, self.pairs)

        # Calculating the number of zero crossings from the dataset
        self.num_crossing = self.count_number_crossing()

        # In case of a selection method other than standard or industry is used, sorting paris
        # based on the method
        self.selection_method(method, num_top, skip_top)

        # Storing only the necessary values for pairs selected in the above
        self.num_crossing = {pair: self.num_crossing[pair] for pair in self.pairs}
        self.train_std = {pair: self.train_std[pair] for pair in self.pairs}
        self.train_portfolio = self.train_portfolio[self.train_portfolio.columns
                                                        .intersection([str(pair) for pair in self.pairs])]

    def selection_method(self, method, num_top, skip_top):
        """
        Select pairs based on the method. This module helps sorting selected pairs for the given method
        in the formation period.

        :param method: (str) Methods to use for sorting pairs [``standard`` by default, ``variance``,
                             ``zero_crossing``].
        :param num_top: (int) Number of top pairs to use for portfolio formation.
        :param skip_top:(int) Number of first top pairs to skip. For example, use skip_top=10
            if you'd like to take num_top pairs starting from the 10th one.
        """

        if method not in ['standard', 'zero_crossing', 'variance']:
            # Raise an error if the given method is inappropriate.
            raise Exception("Please give an appropriate method for sorting pairs between ‘standard’, "
                            "‘zero_crossing’, or 'variance'")

        if method == 'standard':

            self.pairs = self.pairs[skip_top:(skip_top + num_top)]

        elif method == 'zero_crossing':

            # Sorting pairs from the dictionary by the number of zero crossings in a descending order
            sorted_pairs = sorted(self.num_crossing.items(), key=lambda x: x[1], reverse=True)

            # Picking top pairs
            pairs_selected = sorted_pairs[skip_top:(skip_top + num_top)]

            # Removing the number of crossings, so we have only tuples with elements
            pairs_selected = [x[0] for x in pairs_selected]

            self.pairs = pairs_selected

        else:

            # Sorting pairs from the dictionary by the size of variance in a descending order
            sorted_pairs = sorted(self.train_std.items(), key=lambda x: x[1], reverse=True)

            # Picking top pairs
            pairs_selected = sorted_pairs[skip_top:(skip_top + num_top)]

            # Removing the variance, so we have only tuples with elements
            pairs_selected = [x[0] for x in pairs_selected]

            self.pairs = pairs_selected

    def trade_pairs(self, test_data, divergence=2):
        """
        Generates trading signals for formed pairs based on new testing(trading) data.

        This method includes procedures from the trading step of the distance strategy.

        First, the input test data is being normalized with the min and max price values
        from the pairs formation step (so we're not using future data when creating signals).
        Normalized = (Test_Price - Min(Train_Price)) / (Max(Train_Price) - Min(Train_Price))

        Second, pair portfolios (differences of normalized price series) are constructed
        based on the chosen top pairs from the pairs formation step.

        Finally, for each pair portfolio trading signals are created. The logic of the trading
        strategy is the following: we open a position when the portfolio value (difference between
        prices) is bigger than divergence * historical_standard_deviation. And we close the
        position when the portfolio price changes sign (when normalized prices of elements cross).

        Positions are being opened in two ways. We open a long position on the first element
        from pair and a short position on the second element. The price of a portfolio is then:

        Portfolio_price = Normalized_price_A - Normalized_price_B

        If Portfolio_price > divergence * st_deviation, we open a short position on this portfolio.

        IF Portfolio_price < - divergence * st_deviation, we open a long position on this portfolio.

        Both these positions will be closed once Portfolio_price reaches zero.

        :param test_data: (pd.DataFrame/np.array) Dataframe with testing data used to create trading signals.
            This dataframe should contain the same columns as the dataframe used for pairs formation.
        :param divergence: (float) Number of standard deviations used to open a position in a strategy.
            In the original example, 2 standard deviations were used.
        """

        # If np.array given as an input
        if isinstance(test_data, np.ndarray):
            test_data = pd.DataFrame(test_data, columns=self.min_normalize.index)

        # If the pairs formation step wasn't performed
        if self.pairs is None:
            raise Exception("Pairs are not defined. Please perform the form_pairs() step first.")

        # Normalizing the testing data with min and max values obtained from the training data
        self.normalized_data, _, _ = self.normalize_prices(test_data, self.min_normalize, self.max_normalize)

        # Creating portfolios for pairs chosen in the pairs formation stage
        self.portfolios = self.find_portfolios(self.normalized_data, self.pairs)

        # Creating trade signals for pair portfolios
        self.trading_signals = self.signals(self.portfolios, self.train_std, divergence)

    def get_signals(self):
        """
        Outputs generated trading signals for pair portfolios.

        :return: (pd.DataFrame) Dataframe with trading signals for each pair.
            Trading signal here is the target quantity of portfolios to hold.
        """

        return self.trading_signals

    def get_portfolios(self):
        """
        Outputs pair portfolios used to generate trading signals.

        :return: (pd.DataFrame) Dataframe with portfolios for each pair.
        """

        return self.portfolios

    def get_scaling_parameters(self):
        """
        Outputs minimum and maximum values used for normalizing each price series.

        Formula used for normalization:
        Normalized = (Price - Min(Price)) / (Max(Price) - Min(Price))

        :return: (pd.DataFrame) Dataframe with columns 'min_value' and 'max_value' for each element.
        """

        scale = pd.DataFrame()

        scale['min_value'] = self.min_normalize
        scale['max_value'] = self.max_normalize

        return scale

    def get_pairs(self):
        """
        Outputs pairs that were created in the pairs formation step and sorted by the method.

        :return: (list) List containing tuples of two strings, for names of elements in a pair.
        """

        return self.pairs

    def get_num_crossing(self):
        """
        Outputs pairs that were created in the pairs formation step with its number of zero crossing.

        :return: (dict) Dictionary with keys as pairs and values as the number of zero
            crossings for pairs.
        """

        return self.num_crossing

    def count_number_crossing(self):
        """
        Calculate the number of zero crossings for the portfolio dataframe generated with train dataset.

        As the number of zero crossings in the formation period does have some usefulness in predicting
        future convergence, this method calculates the number of times the normalized spread crosses the
        value zero which measures the frequency of divergence and convergence between two securities.

        :return: (dict) Dictionary with keys as pairs and values as the number of zero
            crossings for pairs.
        """

        # Creating a dictionary for number of zero crossings
        num_zeros_dict = {}

        # Iterating through pairs
        for pair in self.train_portfolio:
            # Getting names of individual elements from dataframe column names
            pair_val = pair.strip('\')(\'').split('\', \'')
            pair_val = tuple(pair_val)

            # Check if portfolio price crossed zero
            portfolio = self.train_portfolio[pair].to_frame()
            pair_mult = portfolio * portfolio.shift(1)

            # Get the number of zero crossings for the portfolio
            num_zero_crossings = len(portfolio[pair_mult.iloc[:, 0] <= 0].index)

            # Adding the pair's number of zero crossings to the dictionary
            num_zeros_dict[pair_val] = num_zero_crossings

        return num_zeros_dict

    def plot_portfolio(self, num_pair):
        """
        Plots a pair portfolio (difference between element prices) and trading signals
        generated for it.

        :param num_pair: (int) Number of the pair from the list to use for plotting.
        :return: (plt.Figure) Figure with portfolio plot and trading signals plot.
        """

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 7))
        fig.suptitle('Distance Strategy results for portfolio' + self.trading_signals.columns[num_pair])

        axs[0].plot(self.portfolios[self.trading_signals.columns[num_pair]])
        axs[0].title.set_text('Portfolio value (the difference between element prices)')

        axs[1].plot(self.trading_signals[self.trading_signals.columns[num_pair]], '#b11a21')
        axs[1].title.set_text('Number of portfolio units to hold')

        return fig

    def plot_pair(self, num_pair):
        """
        Plots prices for a pair of elements and trading signals generated for their portfolio.

        :param num_pair: (int) Number of the pair from the list to use for plotting.
        :return: (plt.Figure) Figure with prices for pairs plot and trading signals plot.
        """

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 7))
        fig.suptitle('Distance Strategy results for pair' + self.trading_signals.columns[num_pair])

        pair_val = self.trading_signals.columns[num_pair].strip('\')(\'').split('\', \'')
        pair_val = tuple(pair_val)

        axs[0].plot(self.normalized_data[pair_val[0]], label="Long asset in a portfolio - " + pair_val[0])
        axs[0].plot(self.normalized_data[pair_val[1]], label="Short asset in a portfolio - " + pair_val[1])
        axs[0].legend()
        axs[0].title.set_text('Price of elements in a portfolio.')

        axs[1].plot(self.trading_signals[self.trading_signals.columns[num_pair]], '#b11a21')
        axs[1].title.set_text('Number of portfolio units to hold')

        return fig

    @staticmethod
    def normalize_prices(data, min_values=None, max_values=None):
        """
        Normalizes given dataframe of prices.

        Formula used:
        Normalized = (Price - Min(Price)) / (Max(Price) - Min(Price))

        :param data: (pd.DataFrame) Dataframe with prices.
        :param min_values: (pd.Series) Series with min values to use for price scaling.
            If None, will be calculated from the given dataset.
        :param max_values: (pd.Series) Series with max values to use for price scaling.
            If None, will be calculated from the given dataset.
        :return: (pd.DataFrame, pd.Series, pd.Series) Dataframe with normalized prices
            and series with minimum and maximum values used to normalize price series.
        """

        # If normalization parameters are not given, calculate
        if (max_values is None) or (min_values is None):
            max_values = data.max()
            min_values = data.min()

        # Normalizing the dataset
        data_copy = data.copy()
        normalized = (data_copy - min_values) / (max_values - min_values)

        return normalized, min_values, max_values

    @staticmethod
    def find_pair(data, industry_dict=None):
        """
        Finds the pairs with smallest distances in a given dataframe.

        Closeness measure here is the sum of squared differences in prices.
        Duplicate pairs are dropped, and elements in pairs are sorted in alphabetical
        order. So pairs ('AA', 'BC') and ('BC', 'AA') are treated as one pair ('AA', 'BC').

        :param data: (pd.DataFrame) Dataframe with normalized price series.
        :param industry_dict: (dictionary) Dictionary matching ticker to industry group.
        :return: (dict) Dictionary with keys as closest pairs and values as their distances.
        """

        # Creating a dictionary
        pairs = {}

        # Iterating through each element in dataframe
        for ticker in data:

            # Removing the chosen element from the dataframe
            data_excluded = data.drop([ticker], axis=1)

            # Removing tickers in different industry group if the industry dictionary is given
            if industry_dict is not None:
                # Getting the industry group for the ticker
                industry_group = industry_dict[ticker]
                # Getting the tickers within the same industry group
                tickers_same_industry = [ticker for ticker, industry in industry_dict.items()
                                         if industry == industry_group]
                # Removing other tickers in different industry group
                data_excluded = data_excluded.loc[:, data_excluded.columns.isin(tickers_same_industry)]

            # Calculating differences between prices
            data_diff = data_excluded.sub(data[ticker], axis=0)

            # Calculating the sum of square differences
            sum_sq_diff = (data_diff ** 2).sum()

            # Iterating through second elements
            for second_element in sum_sq_diff.index:
                # Adding all new pairs to the dictionary
                pairs[tuple(sorted((ticker, second_element)))] = sum_sq_diff[second_element]

        return pairs

    @staticmethod
    def sort_pairs(pairs, num_top=5, skip_top=0):
        """
        Sorts pairs of elements and returns top_num of closest ones.

        The skip_top parameter can be used to skip a number of first top portfolios.
        For example, if we'd like to pick pairs number 10-15 from the top list, we set
        num_top = 5, skip_top = 10.

        :param pairs: (dict) Dictionary with keys as pairs and values as distances
            between elements in a pair.
        :param num_top: (int) Number of closest pairs to take.
        :param skip_top: (int) Number of top closest pairs to skip.
        :return: (list) List containing sorted pairs as tuples of strings, representing
            elements in a pair.
        """

        # Sorting pairs from the dictionary by distances in an ascending order
        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=False)

        # Picking top pairs
        top_pairs = sorted_pairs[skip_top:(skip_top + num_top)]

        # Removing distance values, so we have only tuples with elements
        top_pairs = [x[0] for x in top_pairs]

        return top_pairs

    @staticmethod
    def find_volatility(data, pairs):
        """
        Calculates historical volatility of portfolios(differences of prices)
        for set of pairs.


        :param data: (pd.DataFrame) Dataframe with price series to use for calculation.
        :param pairs: (list) List of tuples with two elements to use for calculation.
        :return: (dict) Dictionary with keys as pairs of elements and values as their
            historical volatility.
        """

        # Creating a dictionary
        volatility_dict = {}

        # Iterating through pairs of elements
        for pair in pairs:
            # Getting two price series for elements in a pair
            par = data[list(pair)]

            # Differences between picked price series
            par_diff = par.iloc[:, 0] - par.iloc[:, 1]

            # Calculating standard deviation for difference series
            st_div = par_diff.std()

            # Adding pair and volatility to dictionary
            volatility_dict[pair] = st_div

        return volatility_dict

    @staticmethod
    def find_portfolios(data, pairs):
        """
        Calculates portfolios (difference of price series) based on given prices dataframe
        and set of pairs to use.

        When creating a portfolio, we long one share of the first element and short one share
        of the second element.

        :param data: (pd.DataFrame) Dataframe with price series for elements.
        :param pairs: (list) List of tuples with two str elements to use for calculation.
        :return: (pd.DataFrame) Dataframe with pairs as columns and their portfolio
            values as rows.
        """

        # Creating a dataframe
        portfolios = pd.DataFrame()

        # Iterating through pairs
        for pair in pairs:
            # Difference between price series - a portfolio
            par_diff = data.loc[:, pair[0]] - data.loc[:, pair[1]]
            portfolios[str(pair)] = par_diff

        return portfolios

    @staticmethod
    def signals(portfolios, variation, divergence):
        """
        Generates trading signals based on the idea described in the original paper.

        A position is being opened when the difference between prices (portfolio price)
        diverges by more than divergence (two in the original paper) historical standard
        deviations. This position is being closed once pair prices are crossing (portfolio
        price reaches zero).

        Positions are being opened in both buy and sell directions.

        :param portfolios: (pd.DataFrame) Dataframe with portfolio price series for pairs.
        :param variation: (dict) Dictionary with keys as pairs and values as the
            historical standard deviations of their pair portfolio.
        :param divergence: (float) Number of standard deviations used to open a position.
        :return: (pd.DataFrame) Dataframe with target quantity to hold for each portfolio.
        """

        # Creating a signals dataframe
        signals = pd.DataFrame()

        # Iterating through pairs
        for pair in portfolios:
            # Getting names of individual elements from dataframe column names
            pair_val = pair.strip('\')(\'').split('\', \'')
            pair_val = tuple(pair_val)

            # Historical standard deviation for a pair
            st_dev = variation[pair_val]

            # Check if portfolio price crossed zero
            portfolio = portfolios[pair].to_frame()
            pair_mult = portfolio * portfolio.shift(1)

            # Entering a short position when portfolio is higher than divergence * st_dev
            short_entry_index = portfolio[portfolio.iloc[:, 0] > divergence * st_dev].index
            short_exit_index = portfolio[pair_mult.iloc[:, 0] <= 0].index

            # Entering a long position in the opposite situation
            long_entry_index = portfolio[portfolio.iloc[:, 0] < -divergence * st_dev].index
            long_exit_index = portfolio[pair_mult.iloc[:, 0] <= 0].index

            # Transforming long and short trading signals into one signal - target quantity
            portfolio['long_units'] = np.nan
            portfolio['short_units'] = np.nan
            portfolio.iloc[0, portfolio.columns.get_loc('long_units')] = 0
            portfolio.iloc[0, portfolio.columns.get_loc('short_units')] = 0

            portfolio.loc[long_entry_index, 'long_units'] = 1
            portfolio.loc[long_exit_index, 'long_units'] = 0
            portfolio.loc[short_entry_index, 'short_units'] = -1
            portfolio.loc[short_exit_index, 'short_units'] = 0

            portfolio.fillna(method='pad', inplace=True)
            portfolio['target_quantity'] = portfolio['long_units'] + portfolio['short_units']

            # Adding target quantity to signals dataframe
            signals[str(pair)] = portfolio['target_quantity']

        return signals

# Define styles for the boxes
box_style = """
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
"""

# Define styles for text sizes
big_text_style = "font-size: 26px;"
small_text_style = "font-size: 20px; font-weight: bold;"

@st.cache_data
def arima_model(step, time_series):
    # Find the best ARIMA model using auto_arima
    model = auto_arima(time_series, 
                    start_p=0, start_q=0,
                    max_p=4, max_q=4,
                    m=1, # Set the seasonal period (e.g., 12 for monthly data)
                    seasonal=False, # Enable seasonal modeling
                    trace=True, # Print the search progress
                    error_action='ignore', # Ignore warnings
                    suppress_warnings=True,
                    stepwise=True) # Use stepwise search for faster execution

    model.fit(time_series)

    predictions = model.predict(n_periods=step)
    return predictions

@st.cache_data
def linear_model(step, ts, data_lr):
    
    model = LinearRegression()

    # Train the model
    model.fit(data_lr, ts)

    future_dates = pd.date_range(start = date_today, end = '2030-01-01', freq = 'D')
    future_df = pd.DataFrame()
    future_df = pd.DataFrame({'Date': future_dates})
    future_df.set_index('Date', inplace=True)


    future_df['Index_Day'] = [i.day for i in future_df.index]
    future_df['Index_Month'] = [i.month for i in future_df.index]
    future_df['Index_Year'] = [i.year for i in future_df.index]
    future_df['Index_Week'] = [i.isocalendar().week for i in future_df.index]
    future_df['Index_Quarter'] = [i.quarter for i in future_df.index]
    future_df['Index_Day_Week'] = [i.weekday() for i in future_df.index]
    future_df['Index_Series'] = np.arange(1+len(ts),len(future_df)+1+len(ts))

    x_predict = future_df[['Index_Day', 'Index_Month', 'Index_Year', 'Index_Series', 'Index_Week', 'Index_Quarter', 'Index_Day_Week']]
    predictions = model.predict(x_predict)
    return list(predictions)[:step]

@st.cache_data
def lstm_model(timeseries, predict_num, look_back=7, lstm_units=25, epochs=20):
    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    timeseries_scaled = scaler.fit_transform(np.array(timeseries).reshape(-1, 1))
    
    # Convert timeseries into supervised learning problem
    def create_dataset(timeseries, look_back=1):
        X, Y = [], []
        for i in range(len(timeseries)-look_back):
            X.append(timeseries[i:(i+look_back), 0])
            Y.append(timeseries[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    # Create the supervised learning data
    X, Y = create_dataset(timeseries_scaled, look_back)
    
    # Reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Fit the model
    model.fit(X, Y, epochs=epochs, batch_size=1, verbose=0)
    
    # Use the model to forecast
    forecast = []
    current_batch = X[-1]  # Start prediction from last full sequence
    
    for i in range(predict_num):
        # Predict the next value
        predicted_value = model.predict(current_batch.reshape((1, look_back, 1)))[0, 0]
        
        # Append the predicted value to forecast
        forecast.append(predicted_value)
        
        # Update current_batch to include the predicted value
        current_batch = np.append(current_batch[1:], [[predicted_value]], axis=0)
    
    # Inverse transform forecasted values to original scale
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    
    return list(forecast.flatten())




# --- Sample Data (Replace with your actual data) ---
# date_today = datetime.today() - timedelta(365*3)
# days = pd.date_range(date_today, date_today + timedelta(365*3 - 1), freq='D')
# np.random.seed(1)
# df = pd.DataFrame({'Date': days})
# df.set_index('Date', inplace=True)
# df['A'] = 100 + np.cumsum(np.random.randn(len(days)))
# df['AAPL'] = 150 + np.cumsum(np.random.randn(len(days)))
# df['AAP'] = 200 + np.cumsum(np.random.randn(len(days)))


# --- Initialize session state for selected tickers ---
if 'selected_tickers' not in st.session_state:
    st.session_state['selected_tickers'] = []

if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = pd.DataFrame()



def create_connection():
    """Create a connection to the MySQL database."""
    try:
        connection = mysql.connector.connect(
            host='aws-screener.cn422y4ww81v.ap-southeast-2.rds.amazonaws.com',
            user='admin',
            password='Matkhauaws100402',
            database="aws_screener"
        )
        if connection.is_connected():
            print('Successful')
            return connection
    except Error as e:
        print(e)
        return None

def select_all_records(_connection):
    """Select all records from the stock_data table."""
    try:
        cursor = _connection.cursor()
        cursor.execute("SELECT * FROM stock_data")
        records = cursor.fetchall()
        return records
    except Exception as e:
        print(f"Error: {e}")
        return None

def records_to_dataframe(records):
    """Convert a list of records to a DataFrame with dates as index, tickers as columns, and prices as values."""
    df = pd.DataFrame(records, columns=['TIME', 'TICKER', 'PRICE'])
    df = df.pivot(index='TIME', columns='TICKER', values='PRICE')
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data
def get_stock_data():
    if len(st.session_state['stock_data']) > 0:
        return st.session_state['stock_data']
    connection = create_connection()
    records = select_all_records(connection)
    df = records_to_dataframe(records)

    df['Index_Day'] = [i.day for i in df.index]
    df['Index_Month'] = [i.month for i in df.index]
    df['Index_Year'] = [i.year for i in df.index]
    df['Index_Week'] = [i.isocalendar().week for i in df.index]
    df['Index_Quarter'] = [i.quarter for i in df.index]
    df['Index_Day_Week'] = [i.weekday() for i in df.index]
    df['Index_Series'] = np.arange(1,len(df)+1)
    connection.close()
    return df
    

data = get_stock_data()


date_today = (max(data.index) + timedelta(1)).to_pydatetime()

# df['Index_Day'] = [i.day for i in df.index]
# df['Index_Month'] = [i.month for i in df.index]
# df['Index_Year'] = [i.year for i in df.index]
# df['Index_Week'] = [i.isocalendar().week for i in df.index]
# df['Index_Quarter'] = [i.quarter for i in df.index]
# df['Index_Day_Week'] = [i.weekday() for i in df.index]
# df['Index_Series'] = np.arange(1,len(df)+1)

# data = df

@st.cache_data
def forecast(model_type, ts, data_lr):
    step = (datetime(2030,1,1) - date_today).days
    if model_type == "arima":
        return arima_model(step, ts)
    if model_type == "lr":
        return linear_model(step, ts, data_lr)
    return lstm_model(ts, step)

@st.cache_data
def get_portfolio(ticker1, ticker2, scaling_parameters, trading_signals):

    pair_scales = scaling_parameters.loc[[ticker1, ticker2]]

    # So the scaling parameters for 'CMA' and 'RF' are
    maxmin_1 = pair_scales.loc[ticker1][1] - pair_scales.loc[ticker1][0]
    maxmin_2 = pair_scales.loc[ticker2][1] - pair_scales.loc[ticker2][0]

    scale_1 = (test_data[ticker1][0] / (test_data[ticker1][0] + test_data[ticker2][0])) * (maxmin_2 / (maxmin_1 + maxmin_2))
    scale_2 = (test_data[ticker2][0] / (test_data[ticker1][0] + test_data[ticker2][0])) * (maxmin_1 / (maxmin_1 + maxmin_2))

    test_data_returns = (test_data / test_data.shift(1) - 1)[1:]

    weight_1 = scale_1 / (scale_1 + scale_2)
    weight_2 = 1 - weight_1

    portfolio_returns_scaled = test_data_returns[ticker1] * weight_1 - test_data_returns[ticker2] * weight_2
    portfolio_returns_scaled = portfolio_returns_scaled * (trading_signals[f"('{ticker1}', '{ticker2}')"].shift(1))
    portfolio_price_scaled = (portfolio_returns_scaled + 1).cumprod()

    return (portfolio_price_scaled - 1).fillna(0)


@st.cache_data
def black_scholes(S, K, T, r, sigma):

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return call_price, put_price


available_tickers = sorted([x for x in list(data.columns) if "Index" not in x])

sp_idt = {'Industrials': 'AAL ADP ALLE AME AOS AXON BA BLDR BR CAT CHRW CMI CPRT CSX CTAS DAL DE DOV EFX EMR ETN EXPD FAST FDX FTV GD GE GNRC GWW HII HON HUBB HWM IEX IR ITW J JBHT JCI LDOS LHX LMT LUV MAS MMM NDSN NOC NSC ODFL PAYC PAYX PCAR PH PNR PWR ROK ROL RSG RTX SNA SWK TDG TT TXT UAL UNP UPS URI VRSK WAB WM XYL',
 'Health Care': 'A ABBV ABT ALGN AMGN BAX BDX BIIB BIO BMY BSX CAH CI CNC COO COR CRL CTLT CVS DGX DHR DVA DXCM ELV EW GILD HCA HOLX HSIC HUM IDXX INCY IQV ISRG JNJ LH LLY MCK MDT MOH MRK MTD PFE PODD REGN RMD RVTY STE SYK TECH TFX TMO UHS UNH VRTX VTRS WAT WST ZBH ZTS',
 'Information Technology': 'AAPL ACN ADBE ADI ADSK AKAM AMAT AMD ANET ANSS APH AVGO CDNS CDW CRM CSCO CTSH ENPH EPAM FFIV FICO FSLR FTNT GDDY GEN GLW HPE HPQ IBM INTC INTU IT JBL JNPR KEYS KLAC LRCX MCHP MPWR MSFT MSI MU NOW NTAP NVDA NXPI ON ORCL PANW PTC QCOM QRVO ROP SMCI SNPS STX SWKS TDY TEL TER TRMB TXN TYL VRSN WDC ZBRA',
 'Utilities': 'AEE AEP AES ATO AWK CMS CNP D DTE DUK ED EIX ES ETR EVRG EXC FE LNT NEE NI NRG PCG PEG PNW PPL SO SRE VST WEC XEL',
 'Financials': 'ACGL AFL AIG AIZ AJG ALL AMP AON AXP BAC BEN BK BLK BRO BX C CB CBOE CFG CINF CME COF CPAY DFS EG FDS FI FIS FITB GL GPN GS HBAN HIG ICE IVZ JKHY JPM KEY KKR L MA MCO MET MKTX MMC MS MSCI MTB NDAQ NTRS PFG PGR PNC PRU PYPL RF RJF SCHW SPGI STT SYF TFC TROW TRV USB V WFC WRB WTW',
 'Materials': 'ALB AMCR APD AVY BALL CE CF DD ECL EMN FCX FMC IFF IP LIN LYB MLM MOS NEM NUE PKG PPG SHW STLD VMC',
 'Consumer Discretionary': 'AMZN APTV AZO BBWI BBY BKNG BWA CCL CMG CZR DECK DHI DPZ DRI EBAY ETSY EXPE F GM GPC GRMN HAS HD HLT KMX LEN LKQ LOW LULU LVS MAR MCD MGM MHK NCLH NKE NVR ORLY PHM POOL RCL RL ROST SBUX TJX TPR TSCO TSLA ULTA WYNN YUM',
 'Real Estate': 'AMT ARE AVB BXP CBRE CCI CPT CSGP DLR DOC EQIX EQR ESS EXR FRT HST INVH IRM KIM MAA O PLD PSA REG SBAC SPG UDR VICI VTR WELL WY',
 'Communication Services': 'CHTR CMCSA DIS EA GOOG GOOGL IPG LYV META MTCH NFLX NWS NWSA OMC PARA T TMUS TTWO VZ WBD',
 'Consumer Staples': 'ADM BG CAG CHD CL CLX COST CPB DG DLTR EL GIS HRL HSY K KDP KHC KMB KO KR LW MDLZ MKC MNST MO PEP PG PM SJM STZ SYY TAP TGT TSN WBA WMT',
 'Energy': 'APA BKR COP CTRA CVX DVN EOG EQT FANG HAL HES KMI MPC MRO OKE OXY PSX SLB TRGP VLO WMB XOM'}

sp_idt = {x: sp_idt[x].split() for x in sp_idt}

FORECAST = pd.DataFrame({'Date': pd.date_range(date_today, datetime(2029,12,31), freq='D')})
FORECAST.set_index('Date', inplace=True)

def add_line_graph(fig, name, data, data_index):
    fig.add_trace(go.Scatter(x=data_index, y=data, mode='lines', name=name, hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}'))


# --- Navigation Menu ---
selected = option_menu(
    menu_title="S&P Tracker",
    options=["Dashboard", "Indicators", "Forecasting", "Stat Arb", "Options Pricing", "Notes"],
    orientation="horizontal",
    styles={
        "nav-link-selected": {"background-color": "gray"}
    }
)

# --- Content ---


# --- Content ---
if selected == "Dashboard":
    st.write("### S&P 500 Dashboard")

    # --- Sidebar ---
    st.sidebar.header("Ticker Selection")

    # Searchable Ticker Selection
    ticker_input = st.sidebar.text_input("Enter Ticker Symbol")
    filtered_tickers = [
        ticker for ticker in available_tickers if ticker_input.upper() in ticker
    ]

    selected_ticker = st.sidebar.selectbox(
        "Select Ticker", options=filtered_tickers
    )

    # Add button
    if st.sidebar.button("Add"):
        if selected_ticker not in st.session_state['selected_tickers']:
            st.session_state['selected_tickers'].append(selected_ticker)

    # --- Display and manage selected tickers ---
    with st.sidebar:
        for i, ticker in enumerate(st.session_state['selected_tickers']):
            col1, col2 = st.columns([9, 1])  # Adjust column ratios as needed
            col1.info(ticker)
            if col2.button("X", key=f"remove_{i}"):
                st.session_state['selected_tickers'].pop(i)
                st.rerun()  # Rerun to update the display

    # --- Create the Plotly Figure ---
    fig = go.Figure()

    # Normalize data for each selected ticker
    for ticker in st.session_state['selected_tickers']:
        min_val = data[ticker].min()
        max_val = data[ticker].max()
        normalized_data = (data[ticker] - min_val) / (max_val - min_val)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=normalized_data, 
            mode='lines',
            name=ticker,
            hovertemplate=f'Date: %{{x|%Y-%m-%d}}<br>Price: %{{customdata:.2f}}' 
        ))

        # Add original data as customdata
        fig.update_traces(customdata=data[ticker])

    # --- Customize Layout ---
    fig.update_layout(
        xaxis=dict(
            tickformat="%b %Y", 
            dtick="M2",         
            showgrid=False,      
        ),
        yaxis=dict(
            showticklabels=False, 
            showgrid=False,       
        ),
        plot_bgcolor='white',   
        paper_bgcolor='white',  
        font=dict(color='black'),
        legend=dict(
            orientation="h",     
            yanchor="bottom",    
            y=1.02,               
            xanchor="right",     
            x=1                   
        )
    )

    # --- Display the chart ---
    st.plotly_chart(fig, use_container_width=True)


elif selected == "Indicators":
    st.write("### Indicators")

    # --- Sidebar ---
    st.sidebar.header("Ticker Selection")

    # Searchable Ticker Selection
    ticker_input = st.sidebar.text_input("Enter Ticker Symbol")
    filtered_tickers = [
        ticker for ticker in available_tickers if ticker_input.upper() in ticker
    ]

    selected_ticker = st.sidebar.selectbox(
        "Select Ticker", options=filtered_tickers
    )

    # Checkboxes for additional lines
    show_sma20 = st.sidebar.checkbox("Show 20-day SMA")
    show_sma50 = st.sidebar.checkbox("Show 50-day SMA")
    show_sma100 = st.sidebar.checkbox("Show 100-day SMA")

    show_cma = st.sidebar.checkbox("Show CMA")

    show_ewma20 = st.sidebar.checkbox("Show 20-day EWMA")
    show_ewma50 = st.sidebar.checkbox("Show 50-day EWMA")
    show_ewma100 = st.sidebar.checkbox("Show 100-day EWMA")


    # --- Main Content (Plot) ---
    fig = go.Figure()

    data_ts = data[selected_ticker]

    # Add the selected ticker's data to the plot
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data_ts,
        mode='lines',
        name=selected_ticker,
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}'
    ))

    # Add SMA lines based on checkbox selections
    if show_sma20:
        add_line_graph(fig, "SMA20", data_ts.rolling(20).mean(), data.index)
    if show_sma50:
        add_line_graph(fig, "SMA50", data_ts.rolling(50).mean(), data.index)
    if show_sma100:
        add_line_graph(fig, "SMA100", data_ts.rolling(100).mean(), data.index)
    
    if show_cma:
        add_line_graph(fig, "CMA", data_ts.expanding(1).mean(), data.index)

    if show_ewma20:
        add_line_graph(fig, "EWMA20", data_ts.ewm(span=20).mean(), data.index)
    if show_ewma50:
        add_line_graph(fig, "EWMA50", data_ts.ewm(span=50).mean(), data.index)
    if show_ewma100:
        add_line_graph(fig, "EWMA100", data_ts.ewm(span=100).mean(), data.index)

    # --- Customize Layout ---
    fig.update_layout(
        xaxis=dict(
            tickformat="%b %Y",
            dtick="M2",
            showgrid=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # --- Display the chart ---
    st.plotly_chart(fig, use_container_width=True)


elif selected == "Forecasting":
    st.write("### Forecasting")

    # --- Sidebar ---
    st.sidebar.header("Ticker Selection")

    # Searchable Ticker Selection
    ticker_input = st.sidebar.text_input("Enter Ticker Symbol")
    filtered_tickers = [
        ticker for ticker in available_tickers if ticker_input.upper() in ticker
    ]

    selected_ticker = st.sidebar.selectbox(
        "Select Ticker", options=filtered_tickers
    )

    # Checkboxes for additional lines
    arima = st.sidebar.checkbox("ARIMA")
    lr = st.sidebar.checkbox("Linear Regression")
    lstm = st.sidebar.checkbox("LSTM (will take some times to compute)")

    today = datetime.today()  # Correct way to get today's date
    forecast_date = st.sidebar.date_input(
        "Forecast from now to",
        min_value=today + timedelta(1),
        max_value = datetime(2029,12,31),
        value=today + timedelta(days=30)  # Default to 30 days from today
    )


    # --- Main Content (Plot) ---
    fig = go.Figure()

    # Add the selected ticker's data to the plot
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[selected_ticker],
        mode='lines',
        name=selected_ticker,
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}'
    ))

    date_idx = pd.date_range(date_today, forecast_date, freq='D')

    data_lr = data[['Index_Day', 'Index_Month', 'Index_Year', 'Index_Series', 'Index_Week', 'Index_Quarter', 'Index_Day_Week']]


    # Add SMA lines based on checkbox selections
    if arima:
        print(list(FORECAST.keys()))
        if selected_ticker + " - arima" not in FORECAST:
            FORECAST[selected_ticker + " - arima"] = forecast("arima", data[selected_ticker], data_lr)
        add_line_graph(fig, "ARIMA", list(FORECAST[selected_ticker + " - arima"])[: len(date_idx)], date_idx)
        print(list(FORECAST.keys()))
    if lr:
        print(list(FORECAST.keys()))
        if selected_ticker + " - lr" not in FORECAST:
            FORECAST[selected_ticker + " - lr"] = forecast("lr", data[selected_ticker], data_lr)
        add_line_graph(fig, "LR", list(FORECAST[selected_ticker + " - lr"])[: len(date_idx)], date_idx)
        print(list(FORECAST.keys()))
    if lstm:
        print(list(FORECAST.keys()))
        if selected_ticker + " - lstm" not in FORECAST:
            FORECAST[selected_ticker + " - lstm"] = forecast("lstm", data[selected_ticker], data_lr)
        add_line_graph(fig, "LSTM", list(FORECAST[selected_ticker + " - lstm"])[: len(date_idx)], date_idx)
        print(list(FORECAST.keys()))

    # --- Customize Layout ---
    fig.update_layout(
        xaxis=dict(
            tickformat="%b %Y",
            dtick="M2",
            showgrid=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # --- Display the chart ---
    st.plotly_chart(fig, use_container_width=True)


elif selected == "Stat Arb":

    st.write("### Statistical Arbitrage")

    # --- Sidebar ---
    st.sidebar.header("Industry Selection")

    # Dropdown for ticker selection
    selected_industry = st.sidebar.selectbox(
        "Select Industry",
        options=['Industrials', 'Health Care', 'Information Technology', 'Utilities', 'Financials', 'Materials', 'Consumer Discretionary', 'Real Estate', 'Communication Services', 'Consumer Staples', 'Energy']
    )

    full_data = data[sp_idt[selected_industry]]
    train_data = full_data[full_data.index < datetime(2024,1,1)]
    test_data = full_data[full_data.index >= datetime(2024,1,1)]

    strategy = DistanceStrategy()

    strategy.form_pairs(train_data, num_top=10)

    scaling_parameters = strategy.get_scaling_parameters()
    pairs = strategy.get_pairs()
    historical_std = strategy.train_std

    st.write("Those are the top 10 correlated pairs based on historical data from 2020-2024.")

    selected_pair = st.selectbox(
        "Select Pair", options=pairs
    )

    strategy.trade_pairs(test_data, divergence=1.5)

    portfolio_series = strategy.get_portfolios()
    trading_signals = strategy.get_signals()
    normalized_prices = strategy.normalized_data

    st.write("##### Price data of two stock from 2020-2024:")

    fig = go.Figure()

    # data_ts = data[selected_ticker]

    for ticker in selected_pair:
        min_val = train_data[ticker].min()
        max_val = train_data[ticker].max()
        normalized_data = (train_data[ticker] - min_val) / (max_val - min_val)

        fig.add_trace(go.Scatter(
            x=train_data.index,
            y=normalized_data, 
            mode='lines',
            name=ticker,
            hovertemplate=f'Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}' 
        ))

    # --- Customize Layout ---
    fig.update_layout(
        xaxis=dict(
            tickformat="%b %Y",
            dtick="M2",
            showgrid=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # --- Display the chart ---
    st.plotly_chart(fig, use_container_width=True)

    return_portfolio = get_portfolio(selected_pair[0], selected_pair[1], scaling_parameters, trading_signals)

    st.write("##### Pair trading strategy return of two stocks since 2024-01-01 (threshold = 1.5 * stdev)")

    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=return_portfolio.index, 
        y=return_portfolio*100, 
        mode='lines', 
        name="PnL", 
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Change: %{y:.2f}%'
        ))

    # add_line_graph(fig, "PnL", return_portfolio, return_portfolio.index)


    fig.update_layout(
        xaxis=dict(
            tickformat="%b %Y",
            dtick="M2",
            showgrid=False,
        ),
        yaxis=dict(
            title='Strategy return (%)',  # Add a title for the y-axis
            showticklabels=True,   # Show tick labels on the y-axis
            showgrid=True,        # Disable the grid lines on the y-axis
            zeroline=True,        # Optionally, hide the zero line
            ticks='outside'        # Optionally, show ticks outside the plot area
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )


    # # --- Display the chart ---
    st.plotly_chart(fig, use_container_width=True)


elif selected == "Options Pricing":

    st.write("### Option Pricing")

    # --- Sidebar ---
    st.sidebar.header("Option Parameters")

    # Searchable Ticker Selection
    ticker_input = st.sidebar.text_input("Enter Ticker Symbol")
    filtered_tickers = [
        ticker for ticker in available_tickers if ticker_input.upper() in ticker
    ]

    selected_ticker_option = st.sidebar.selectbox(
        "Select Ticker", options=filtered_tickers
    )

    # Strike Price
    strike_price = st.sidebar.number_input(
        "Strike Price ($)",
        min_value=0.01,
        step=0.01,
        value=50.00  # Example default value
    )

    # Risk-Free Rate
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.01,
        step=0.01,
        value=5.00  # Default to 5%
    ) / 100  # Convert percentage to decimal

    # Maturity Date
    today = datetime.today()  # Correct way to get today's date
    maturity_date = st.sidebar.date_input(
        "Maturity Date",
        min_value=today,
        max_value = datetime(2029,12,31),
        value=today + timedelta(days=30)  # Default to 30 days from today
    )

    # Volatility
    volatility = st.sidebar.number_input(
        "Volatility",
        min_value=0.01,
        step=0.01,
        value=0.25  # Default volatility
    )

    # Min/Max Volatility Range Slider
    volatility_range = st.sidebar.slider(
        "Volatility Range for heatmap",
        min_value=0.01,
        max_value=1.0,
        value=(0.1, 0.35),  # Default min/max values
        step=0.01
    )

    vol_diff = (volatility_range[1] - volatility_range[0]) / 5

    mtrt_date = datetime(maturity_date.year, maturity_date.month, maturity_date.day)

    # --- Main Content ---
    S = (data[selected_ticker_option].sort_index())[-1]
    K = strike_price
    T = int((mtrt_date - date_today).days + 1)/365
    r = risk_free_rate
    sigma = volatility

    call_price, put_price = black_scholes(S, K, T, r, sigma)

    st.markdown(
            "<span style='{}'>Current stock price: ${}</span>".format(big_text_style, round(S, 2)),
            unsafe_allow_html=True
        )
    
    x_labels = [S*0.95, S*0.97, S*0.99, S*1.01, S*1.03, S*1.05]
    # y_labels = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E', 'Category F']
    y_labels = [volatility_range[0] + vol_diff*multiplier for multiplier in range(6)]

    x_labels = [round(x, 2) for x in x_labels]
    y_labels = [round(x,2) for x in y_labels][::-1]

    heatmap_data_call = []
    heatmap_data_put = []

    for _y_ in y_labels:
        call_row, put_row = [], []
        for _x_ in x_labels:
            cp, pp = black_scholes(_x_, K, T, r, _y_)
            call_row.append(cp)
            put_row.append(pp)
        heatmap_data_call.append(call_row)
        heatmap_data_put.append(put_row)
        

    # Round the values to 2 decimal places
    heatmap_data_call = np.round(heatmap_data_call, 2)
    heatmap_data_put = np.round(heatmap_data_put, 2)

    # Create two columns
    col1, col2 = st.columns(2)

    # Call price column
    with col1:
        st.markdown(
            "<div style='{}'><span style='{}'>Call price</span><br><span style='{}'>${}</span></div>".format(box_style, big_text_style, small_text_style, round(call_price, 2)),
            unsafe_allow_html=True
        )
            # Generate random data for the heatmap
        st.markdown("")
        st.markdown("")

        # Display the heatmap
        fig, ax = plt.subplots()
        heatmap = ax.imshow(heatmap_data_call, cmap='RdYlGn', vmin=heatmap_data_call.min(), vmax=heatmap_data_call.max())

        # Add number annotations
        for i in range(len(heatmap_data_call)):
            for j in range(len(heatmap_data_call[i])):
                text = ax.text(j, i, f'{heatmap_data_call[i, j]:.2f}',
                            ha='center', va='center', color='black')

        # Add colorbar
        plt.colorbar(heatmap)

        # Set custom tick labels for x-axis and y-axis
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

        # Add labels and title
        ax.set_xlabel('Spot Price $')
        ax.set_ylabel('Volatility')
        ax.set_title('Call Price Heatmap')

        # Show plot in Streamlit
        st.pyplot(fig)

    # Put price column
    with col2:
        st.markdown(
            "<div style='{}'><span style='{}'>Put price</span><br><span style='{}'>${}</span></div>".format(box_style, big_text_style, small_text_style, round(put_price, 2)),
            unsafe_allow_html=True
        )

        st.markdown("")
        st.markdown("")

        # Display the heatmap
        fig, ax = plt.subplots()
        heatmap = ax.imshow(heatmap_data_put, cmap='RdYlGn', vmin=heatmap_data_put.min(), vmax=heatmap_data_put.max())

        # Add number annotations
        for i in range(len(heatmap_data_put)):
            for j in range(len(heatmap_data_put[i])):
                text = ax.text(j, i, f'{heatmap_data_put[i, j]:.2f}',
                            ha='center', va='center', color='black')

        # Add colorbar
        plt.colorbar(heatmap)

        # Set custom tick labels for x-axis and y-axis
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

        # Add labels and title
        ax.set_xlabel('Spot Price $')
        ax.set_ylabel('Volatility')
        ax.set_title('Put Price Heatmap')

        # Show plot in Streamlit
        st.pyplot(fig)

    # st.write("## Option Price Calculation (Placeholder)")
    # st.write(f"**Ticker:** {selected_ticker_option}")
    # st.write(f"**Strike Price:** ${strike_price:.2f}")
    # st.write(f"**Risk-Free Rate:** {risk_free_rate*100:.2f}%")
    # st.write(f"**Maturity Date:** {maturity_date.strftime('%Y-%m-%d')}")
    # st.write(f"**Volatility:** {volatility:.2f}")
    # st.write(f"**Volatility Range:** {volatility_range}")

    # ... (Add your option pricing logic here) ...

elif selected == "Notes":
    st.write("### Notes")
    st.markdown("This is a personal project from [Thanh Duong](https://www.linkedin.com/in/thanhqduong/) - I welcome any suggestions!")
    st.write("The data is obtained from Yahoo Finance.")
    st.write("Due to the limited storage capacity of my database, I have chosen not to store a large amount of data that requires extensive computation, which has resulted in a slower web app.")
