import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import getpass
import zipfile
import traceback
import multiprocessing
import dyharrel.dyharrel.data_calls_util as dcu
import warnings
import shutil
import time
import random
import datetime
import requests
import pandas
import numpy
import urllib
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class StockTickerCompendium:
    def __init__(self):
        self.stock_list_dataframe = dcu.stock_list_dataframe()
        s_and_p_dataframe = dcu.s_and_p_dataframe()
        country_list = self.stock_list_dataframe['Country'].to_list()
        ticker_list = self.stock_list_dataframe['Symbol'].to_list()
        name_list = self.stock_list_dataframe['Name'].to_list()
        sector_list = self.stock_list_dataframe['Sector'].to_list()
        industry_list = self.stock_list_dataframe['Industry'].to_list()
        ipo_list = self.stock_list_dataframe['IPO Year'].to_list()
        self.s_and_p_list = s_and_p_dataframe['Symbol'].to_list()
        self.ticker_dict = {
            ticker: {
                'company': name_list[ticker_index],
                'sector': sector_list[ticker_index],
                'industry': industry_list[ticker_index],
                'ipo_year': ipo_list[ticker_index],
                'country': country_list[ticker_index]
            } for ticker_index, ticker in enumerate(ticker_list)

        }
        self.country_list = sorted(
            list(set(country_list))
        )
        self.sector_list = sorted(
            list(set(sector_list))
        )
        self.industry_list = sorted(
            list(set(industry_list))
        )
        self.ipo_year_list = sorted(
            list(set(ipo_list))
        )
        self.stooq_country_crosswalk = {
            'Unites States': 'US'
        }


class OnlineStockDataOrigins:
    def __init__(self):
        # where TICKER is an abbreviation identifying traded share,
        # and INTERVAL defines the time interval between quotations.
        # You can find at Stooq information about tickers for all
        # shares for which data are accessible. For Polish shares,
        # it will be usually something like CCC, ALE, etc.
        # In the case of US shares, it is required to add country code after the dot,
        # e.g., APPL.US. Interval is one letter and can take the following values:
        # d - daily, w - weekly, m - monthly, q - quarterly, and y - yearly.
        self.stooq_origin = "https://stooq.com/q/d/l/?s=TICKER&i=INTERVAL"

    def build_stooq_url(self, stock_ticker, time_interval='d'):
        return self.stooq_origin.replace(
            "TICKER", stock_ticker).replace(
            "INTERVAL", time_interval
        )


STOCK_TICKER_COMPENDIUM = StockTickerCompendium()
GOT_INTERNET = True


class StooqStockDataframe:
    def __init__(
            self,
            stock_ticker,
            date_start,
            date_end=datetime.datetime.today()
    ):
        self.stock_ticker = str(stock_ticker).strip()
        self.date_start = dcu.check_date(date_start)
        self.date_end = dcu.check_date(date_end)
        stock_country = STOCK_TICKER_COMPENDIUM.ticker_dict[self.stock_ticker.upper()]['country']
        if stock_country == 'United States':
            stock_country_code = '.US'
        else:
            try:
                stock_country_code = "." + STOCK_TICKER_COMPENDIUM.stooq_country_crosswalk[
                    stock_country
                ].strip()
            except KeyError:
                #warnings.warn(f'WARNING: stock_country_code not built out for ticker {stock_ticker}.')
                stock_country_code = ''
        self.stock_ticker = f"{self.stock_ticker}{stock_country_code}".upper()
        try:
            dataframe_file_path = f"C:\\stooq\\dataframes\\{self.stock_ticker.upper().replace('.US', '')}"
            file_exists = os.path.isfile(dataframe_file_path)
            assert file_exists == True
            file_date = datetime.datetime.fromtimestamp(
                os.path.getmtime(dataframe_file_path)).strftime('%Y-%m-%d %H:%M:%S')
            file_date = dcu.check_date(file_date)
            assert file_date >= dcu.check_date(self.date_end)
            self.dataframe = pandas.read_pickle(dataframe_file_path)
            self.dataframe.sort_index()
            self.dataframe.loc[str(date_end)]
            # filter by dates
            self.dataframe = self.dataframe.loc[
                (self.dataframe.index >= pandas.Timestamp(date_start))
                & (self.dataframe.index <= pandas.Timestamp(date_end))
                ]
            #print(dataframe_file_path)
        except (AssertionError, KeyError):
            try:
                raw_data_file_path = f"C:\\stooq\\raw_data\\stocks\\{self.stock_ticker.lower()}.txt"
                file_exists = os.path.isfile(raw_data_file_path)
                assert file_exists == True
                file_date = datetime.datetime.fromtimestamp(
                    os.path.getmtime(raw_data_file_path)).strftime('%Y-%m-%d %H:%M:%S')
                file_date = dcu.check_date(file_date)
                assert file_date >= self.date_end
                # Get dataframe from raw data
                raw_dataframe = pandas.read_csv(
                    raw_data_file_path,
                    index_col=False
                )
                raw_dataframe = raw_dataframe[raw_dataframe.columns[[2, 4, 5, 6, 7, 8]]]
                raw_dataframe.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                raw_dataframe[raw_dataframe.columns[0]] = \
                    pandas.to_datetime(raw_dataframe[raw_dataframe.columns[0]], format='%Y%m%d')
                raw_dataframe.set_index(raw_dataframe.columns[0], inplace=True)
                raw_dataframe.sort_index()
                raw_dataframe.loc[str(date_end)]
                # filter by dates
                returned_dataframe = raw_dataframe.loc[
                    (raw_dataframe.index >= pandas.Timestamp(date_start))
                    & (raw_dataframe.index <= pandas.Timestamp(date_end))
                    ]

                self.dataframe = returned_dataframe.apply(pandas.to_numeric, errors='coerce')
                raw_dataframe = None
                #print(raw_data_file_path)
            except (AssertionError, KeyError):
                self.requested_url = OnlineStockDataOrigins(
                ).build_stooq_url(f"{self.stock_ticker}")
                self.dataframe = dcu.get_online_stock_data(
                    online_stock_csv_url=self.requested_url,
                    date_start=self.date_start,
                    date_end=self.date_end
                )
                #print(self.requested_url)


class Stock:
    def __init__(
            self,
            stock_ticker,
            date_start=datetime.datetime.today()-datetime.timedelta(days=20*365),
            date_end=datetime.datetime.today(),
            online_source='stooq',
            day_forecast=1,
            **kwargs
    ):
        global GOT_INTERNET
        self.ticker = str(stock_ticker).strip().upper()
        self.day_forecast = day_forecast
        if GOT_INTERNET:
            if str(online_source).strip().lower() == 'stooq':
                try:
                    online_stock_source = StooqStockDataframe(
                        stock_ticker,
                        date_start=date_start,
                        date_end=date_end,
                        **kwargs
                    )
                    self.date_start = online_stock_source.date_start
                    self.date_end = online_stock_source.date_end
                    self.dataframe = online_stock_source.dataframe
                    pandas.to_pickle(online_stock_source.dataframe, f'C:\\stooq\\dataframes\\{self.ticker}')
                except urllib.error.URLError:
                    warnings.warn("Unable to access internet. Attempting local load.")
                    GOT_INTERNET = False
        if not GOT_INTERNET:
            self.date_start = dcu.check_date(date_start)
            self.date_end = dcu.check_date(date_end)
            self.dataframe = pandas.read_pickle(f'C:\\stooq\\dataframes\\{self.ticker}')
        os.makedirs('C:\\stooq\\dataframes', exist_ok=True)
        try:

            self.company_name = STOCK_TICKER_COMPENDIUM.ticker_dict[self.ticker]['company']
            self.ipo_year = STOCK_TICKER_COMPENDIUM.ticker_dict[self.ticker]['ipo_year']
            self.industry = STOCK_TICKER_COMPENDIUM.ticker_dict[self.ticker]['industry']
            self.sector = STOCK_TICKER_COMPENDIUM.ticker_dict[self.ticker]['sector']
            self.country = STOCK_TICKER_COMPENDIUM.ticker_dict[self.ticker]['country']
        except KeyError:
            self.company_name = ''
            self.ipo_year = ''
            self.industry = ''
            self.sector = ''
            self.country = ''
        online_stock_source = None

    def add_rate_of_change(self, column_name, day_offset=1, encode=True):
        if isinstance(self.day_forecast, int) and self.day_forecast > 0:
            day_offset = self.day_forecast
        self.dataframe[f'ROC_{column_name}'] = \
            self.dataframe[column_name].diff(periods=day_offset).shift(periods=-day_offset)
        if encode:
            self.dataframe[f'ROC_{column_name}'] = \
                self.dataframe[f'ROC_{column_name}'].apply(
                    lambda x: 1 if x > 0 else 0)
        for day in range(1, day_offset + 1):
            self.dataframe.iloc[
                -day, self.dataframe.columns.get_loc(f'ROC_{column_name}')
            ] = 0

    def add_fred_data(self, fred_ticker):
        if GOT_INTERNET:
            fred_url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_ticker}' \
                       f'&fq=Daily%2C%20Close&fam=avg'
            fred_dataframe = dcu.get_online_stock_data(fred_url, self.date_start, self.date_end)
            self.dataframe = \
                pandas.merge(
                    self.dataframe,
                    fred_dataframe,
                    how='left',
                    left_index=True,
                    right_index=True
                )

        fred_dataframe = None





class StockPeers:
    def __init__(
            self,
            stock,
            same_country=True,
            same_industry=True,
            same_sector=True
    ):
        self.analyzed_stock = stock.ticker
        self.peer_list = []
        if same_country and same_industry and same_sector:
            for ticker, ticker_key in STOCK_TICKER_COMPENDIUM.ticker_dict.items():
                if ticker_key['country'] == stock.country \
                        and ticker_key['industry'] == stock.industry \
                        and ticker_key['sector'] == stock.sector:
                    self.peer_list.append(str(ticker).strip().upper())

        if not same_country and same_industry and same_sector:
            for ticker, ticker_key in STOCK_TICKER_COMPENDIUM.ticker_dict.items():
                if ticker_key['industry'] == stock.industry \
                        and ticker_key['sector'] == stock.sector:
                    self.peer_list.append(str(ticker).strip().upper())
        if same_country and not same_industry and same_sector:
            for ticker, ticker_key in STOCK_TICKER_COMPENDIUM.ticker_dict.items():
                if ticker_key['country'] == stock.country \
                        and ticker_key['sector'] == stock.sector:
                    self.peer_list.append(str(ticker).strip().upper())
        if same_country and same_industry and not same_sector:
            for ticker, ticker_key in STOCK_TICKER_COMPENDIUM.ticker_dict.items():
                if ticker_key['country'] == stock.country \
                        and ticker_key['industry'] == stock.industry:
                    self.peer_list.append(str(ticker).strip().upper())
        try:
            self.peer_list.remove(self.analyzed_stock)
        except ValueError:
            pass


class StockPredictionWarehouse:
    def __init__(
            self,
            predicting_stock: Stock,
            target_columns: list,
            additional_stock_data: list = []
    ):
        self.predicting_stock_ticker = predicting_stock.ticker
        self.target_columns = \
            [f'{self.predicting_stock_ticker}_{target_column}'
             for target_column in target_columns]
        self.predicting_dataframe = predicting_stock.dataframe.copy(deep=True)
        self.predicting_dataframe.columns = \
            [f'{self.predicting_stock_ticker}_{column}'
             for column in self.predicting_dataframe.columns]
        if additional_stock_data:
            for single_stock in additional_stock_data:
                additional_dataframe = single_stock.dataframe.copy(deep=True)
                if additional_dataframe.shape[0] < 150:
                    '''
                    warnings.warn(
                        f"Stock {single_stock.ticker} has less than a year's worth of data."
                        f" It will not be added."
                    )
                    '''
                else:
                    additional_dataframe.columns = \
                        [f'{single_stock.ticker}_{column}'
                         for column in additional_dataframe.columns]
                    self.predicting_dataframe = \
                        pandas.merge(
                            self.predicting_dataframe,
                            additional_dataframe,
                            how='left',
                            left_index=True,
                            right_index=True
                        )
                    additional_dataframe = None
        # Add datetime columns
        self.predicting_dataframe = (
            self.predicting_dataframe
                .assign(day=self.predicting_dataframe.index.day)
                .assign(month=self.predicting_dataframe.index.month)
                .assign(day_of_week=self.predicting_dataframe.index.dayofweek)
                .assign(week_of_year=self.predicting_dataframe.index.isocalendar().week)
        )
        cyclisized = {'day_of_week': [7, 0], 'month': [12, 1], 'week_of_year': [52, 0]}
        # Loop through all column feature names, add additional cyclical features
        for cyclisized_name, cyclisized_range in cyclisized.items():
            dataframe = generate_cyclical_features(
                self.predicting_dataframe,
                cyclisized_name,
                cyclisized_range[0],
                cyclisized_range[1]
            )
        # Drop blank values
        self.predicting_dataframe = self.predicting_dataframe.dropna()
        self.target_dataframe = self.predicting_dataframe[self.target_columns]
        self.predicting_dataframe = self.predicting_dataframe.drop(columns=self.target_columns)

class StockExpress:
    def __init__(
            self,
            stock_ticker,
            target_column,
            use_peer_list = True,
            use_fred_list = False,
            add_fred_tickers = ['T5YIFR', 'T5YIE', 'DGS10'],
            add_stock_tickers = [],#'ES.C', 'SU.C'],
            target_type='ROC',
            date_start=datetime.datetime.today() - datetime.timedelta(days=20 * 365),
            date_end=datetime.datetime.today(),
            random_sampler=True,
            sample_size=200,
            day_forecast=1
    ):

        date_end = dcu.check_date(date_end)
        # 'T5YIFR'
        self.stock = Stock(stock_ticker, date_start=date_start, date_end=date_end, day_forecast=day_forecast)
        if use_fred_list:
            for fred_data in add_fred_tickers:
                self.stock.add_fred_data(fred_ticker=fred_data)
        if target_type == 'ROC':
            self.stock.add_rate_of_change(target_column)
        if use_peer_list:
            list = StockPeers(self.stock).peer_list + add_stock_tickers
            if random_sampler:
                try:
                    list = random.sample(list, sample_size)
                except ValueError:
                    list = list
        else:
            list = add_stock_tickers
        self.peer_data = generate_stocks(list, date_start=date_start, date_end=date_end, day_forecast=day_forecast)
        self.data_warehouse = StockPredictionWarehouse(self.stock, ['ROC_Close'], self.peer_data)
        self.ml_data = stock_warehouse_train_test_split(self.data_warehouse)
        #print(f"Start Date: {self.stock.date_start}")
        #print(f"  End Date: {self.stock.date_end}")

    def refresh_ml_data(self):
        self.ml_data = stock_warehouse_train_test_split(self.data_warehouse)


# function that generates cyclical features, given a datetime column
def generate_cyclical_features(df_in, cyclisized_name, period, start_num=0):
    kwargs = {
        f'sin_{cyclisized_name}':
            lambda x: numpy.sin(2 * numpy.pi * (df_in[cyclisized_name] - start_num) / period),
        f'cos_{cyclisized_name}':
            lambda x: numpy.cos(2 * numpy.pi * (df_in[cyclisized_name] - start_num) / period)
    }
    return df_in.assign(**kwargs).drop(columns=[cyclisized_name])

def stock_warehouse_train_test_split(
        stock_warehouse: StockPredictionWarehouse,
        test_size=0.35,
        **kwargs
):

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        stock_warehouse.predicting_dataframe.iloc[:-1],
        stock_warehouse.target_dataframe.iloc[:-1],
        test_size=test_size,
        **kwargs
    )
    X_predict = stock_warehouse.predicting_dataframe.iloc[-1:]
    X_predict.index
    return {
        'Y_train': Y_train,
        'Y_test': Y_test,
        'X_train': X_train,
        'X_test': X_test,
        'X_predict': X_predict
    }

def generate_stocks(
        stock_ticker_list,
        **kwargs
):
    stock_list = []
    for stock_ticker in stock_ticker_list:
        stock_list.append(Stock(stock_ticker, **kwargs))
    return stock_list

def analyze_data(stock_train_test_data, analyze_type='logistic_regression', show_graphs=True,
                 lstm_density=1,
                 lstm_batch_size=16,
                 lstm_epochs=50,
                 lstm_optimizer='sgd',
                 lstm_loss_function='mae',
                 lstm_hidden_layer=4
                 ):
    analyzed_model = {}
    scalar = StandardScaler()
    X_train = scalar.fit_transform(stock_train_test_data['X_train'])
    X_test = scalar.fit_transform(stock_train_test_data['X_test'])
    X_predict = scalar.fit_transform(stock_train_test_data['X_predict'])
    Y_train = stock_train_test_data['Y_train'].values
    Y_test = stock_train_test_data['Y_test'].values
    X_validate, Y_validate = X_test[:-10], Y_test[:-10]
    if analyze_type == 'logistic_regression':
        model = LogisticRegression(
            solver='saga',
            C=1.0,
            max_iter=200,
            random_state=0).fit(X_train, Y_train)
        if show_graphs:
            cm = confusion_matrix(Y_test, model.predict(X_test))

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cm)
            ax.grid(False)
            ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
            ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
            ax.set_ylim(1.5, -0.5)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
            plt.show()

            #print(classification_report(Y_test, model.predict(X_test)))
        try:
            test_score = model.score(X_test, Y_test)
            train_score = model.score(X_train, Y_train)
            tomorrow_predict = model.predict(X_predict)
            report = classification_report(Y_test, model.predict(X_test))
        except AttributeError:
            test_score = ''
            train_score = ''
            tomorrow_predict = ''
            report = ''
        analyzed_model = {
            'model': model,
            'test_score': test_score,
            'train_score': train_score,
            'prediction': tomorrow_predict,
            'report':  report
        }
    if analyze_type == 'lstm':
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        X_validate = X_validate.reshape((X_validate.shape[0], 1, X_validate.shape[1]))
        X_predict = X_predict.reshape((X_predict.shape[0], 1, X_predict.shape[1]))
        # Define the model
        model = keras.Sequential()
        model.add(layers.LSTM(lstm_hidden_layer, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(lstm_density))
        ##print(model.summary())
        # Compile the model
        model.compile(
            loss=lstm_loss_function,
            optimizer=lstm_optimizer,
            metrics=["accuracy"],
        )
        # Fit the model
        history = model.fit(
            X_train,
            Y_train,
            validation_data=(X_test, Y_test),
            batch_size=lstm_batch_size,
            epochs=lstm_epochs,
            verbose=0
        )
        history = model.fit(
            X_train,
            Y_train,
            validation_data=(X_validate, Y_validate),
            batch_size=lstm_batch_size,
            epochs=lstm_epochs,
            verbose=0
        )
        raw_prediction = model.predict(X_predict, verbose=0)[0][0]
        accuracy = history.history['accuracy'][-1]
        #print(accuracy)
        #print(raw_prediction)
        prediction_result = 1 if raw_prediction > 0.5 else 0
        #print(f'raw_prediction: {raw_prediction}')
        #print(f"    prediction: {prediction_result}")
        analyzed_model = {
            'model': model,
            'raw_prediction': raw_prediction,
            'prediction': prediction_result,
            'accuracy': accuracy
        }


    return analyzed_model

def unpack_stooq_data(
        in_folder=f"C:\\Users\\{getpass.getuser()}\\Downloads",
        data_zip_file="d_us_txt.zip"
):
    # Check if folder exists
    assert os.path.isdir(in_folder) == True
    # Check if stooq zip file exist
    zip_file_path = f"{in_folder}\\{data_zip_file}"
    assert os.path.isfile(zip_file_path) == True
    # Set the directory to save it in
    unzipped_directory = f"{in_folder}\\unzipped_stooq_data"
    # Delete the directory if it exists
    if os.path.isdir(unzipped_directory):
        shutil.rmtree(unzipped_directory)
    # Create empty directory
    os.makedirs(unzipped_directory, exist_ok=True)
    # Unzip File
    with zipfile.ZipFile(zip_file_path, 'r') as zipped_object:
        all_zipped_members = zipped_object.namelist()
        for archived_member in all_zipped_members:
            # Check if text file. If not, ensure directory exists
            if '.txt' not in archived_member:
                os.makedirs(f"{unzipped_directory}\\{archived_member}", exist_ok=True)
            else:
                # Get file name
                file_name = archived_member[archived_member.rindex('/')+1:]
                # Get directory name
                directory_name = archived_member.replace(archived_member[archived_member.rindex('/'):], '')
                # Ensure directory exists
                copy_directory = f"{unzipped_directory}\\{directory_name}"
                os.makedirs(copy_directory, exist_ok=True)
                # Copy file to directory
                try:
                    zipfile.ZipFile.extract(zipped_object, member=archived_member, path=copy_directory)
                except Exception:
                    warnings.warn(f"Error during copying of file: {file_name}. "
                                  f"Exception below: \n\n{traceback.format_exc()}")

def copy_unzipped_stooq_data(
        *,
        data_type='stock',
        source_directory=f"C:\\Users\\{getpass.getuser()}\\Downloads\\unzipped_stooq_data"
):
    if data_type =='stock':
        target_folder = 'C:\\stooq\\raw_data\\stocks'
        os.makedirs(target_folder, exist_ok=True)
        data_folders = [
            f"{source_directory}\\data\\daily\\us\\nasdaq stocks",
            f"{source_directory}\\data\\daily\\us\\nyse stocks",
            f"{source_directory}\\data\\daily\\us\\nysemkt stocks"
        ]
        for single_folder in data_folders:
            for root, dirs, files in os.walk(single_folder, topdown=False):
                for name in files:
                    if name.endswith('.txt'):
                        source_location = os.path.join(root, name)
                        shutil.copy2(source_location, target_folder)


def remove_folder(target=f"C:\\Users\\{getpass.getuser()}\\Downloads\\unzipped_stooq_data"):
    shutil.rmtree(target)

def stooq_zip_ingestion():
    print('_'*100)
    print('Starting unpacking of zip file ...')
    print('_' * 100)
    unpack_stooq_data()
    print('Zip file successfully unpacked!')
    print('_' * 100)
    print('Starting copying of data into stooq directories ...')
    print('_' * 100)
    copy_unzipped_stooq_data()
    print('Data successfully copied!!')
    print('_' * 100)
    print('Starting removal of unpacked data ...')
    print('_' * 100)
    remove_folder()
    print('Folder successfully removed!!')

def report_worker(stock_ticker, end_date, day_forecast=1):
    print(f'working on {stock_ticker}')
    try:
        # Create StockExpress
        stock_express = StockExpress(stock_ticker, 'Close', date_end=end_date, day_forecast=day_forecast)
        stock_close = stock_express.data_warehouse.predicting_dataframe[
            f'{str(stock_ticker).upper()}_Close'
        ].loc[str(end_date)]
        # Create Model
        stock_model = analyze_data(
            stock_express.ml_data,
            analyze_type='lstm',
            lstm_batch_size=24,
            lstm_hidden_layer=128,
            lstm_optimizer='adam'
        )
        returned_dict = {
            'accuracy': stock_model['accuracy'],
            'raw_prediction': stock_model['raw_prediction'],
            'prediction': stock_model['prediction'],
            'close_value': stock_close
        }
    except Exception:
        returned_dict = None
    finally:
        print(f'finished {stock_ticker}')
        stock_express = None
        stock_model = None
        return returned_dict

def generate_daily_report(end_date=datetime.datetime.today(), stock_queue_length=100, pool_workers=10, day_forecast=1):
    end_date=dcu.check_date(end_date)
    report_root = "C:\\stooq\\reports"
    os.makedirs(report_root, exist_ok=True)
    report_folder = f"{report_root}\\daily_report_{end_date}"
    text_file_path = f"{report_folder}\\full_report"
    # Ensure report folder exists
    os.makedirs(report_folder, exist_ok=True)
    # Create empty dataframe
    stock_dataframe = pandas.DataFrame(columns=['Ticker',
                                                'Company',
                                                'Close Value',
                                                'Accuracy',
                                                'Raw Score',
                                                'Score',
                                                'Suggestion'])

    stock_queue = []
    for single_stock in STOCK_TICKER_COMPENDIUM.ticker_dict.keys():
        stock_queue.append(single_stock)
        if len(stock_queue) == stock_queue_length:
            # Create tuple list
            passed_in_args = []
            for queued_stock in stock_queue:
                passed_in_args.append((queued_stock, end_date, day_forecast))
            # Create pool and save results
            with multiprocessing.Pool(pool_workers) as p:
                pooled_results = p.starmap(report_worker, passed_in_args)
            # Update based on results
            for stock_position, queued_stock in enumerate(stock_queue):
                returned_stock_dict = pooled_results[stock_position]
                if returned_stock_dict is not None:
                    stock_dataframe.loc[len(stock_dataframe.index)] = [
                        queued_stock,
                        STOCK_TICKER_COMPENDIUM.ticker_dict[queued_stock]['company'],
                        returned_stock_dict['close_value'],
                        returned_stock_dict['accuracy'],
                        returned_stock_dict['raw_prediction'],
                        returned_stock_dict['prediction'],
                        'BUY' if returned_stock_dict['prediction'] == 1 else 'SELL'
                    ]
            stock_dataframe.to_csv(text_file_path)
            print('_'*50)
            # Clear queue
            stock_queue = []
    # Sort the dataframe by the accuracy, then the raw_prediction
    stock_dataframe = stock_dataframe.sort_values(['Accuracy', 'Raw Score'], ascending=[True, True])

    df_good = stock_dataframe.loc[(stock_dataframe['Accuracy'] >= 0.98) & (stock_dataframe['Suggestion'] == 'BUY')]
    df_bad = stock_dataframe.loc[(stock_dataframe['Accuracy'] >= 0.98) & (stock_dataframe['Suggestion'] == 'SELL')]
    df_good = df_good.sort_values(['Raw Score'], ascending=[False])
    df_bad = df_bad.sort_values(['Raw Score'], ascending=[True])
    # Save all three dataframes as csv to report folder
    stock_dataframe.to_csv(text_file_path)
    df_good.to_csv(f"{text_file_path}_BUY")
    df_bad.to_csv(f"{text_file_path}_SELL")

def generate_s_and_p_report(end_date=datetime.datetime.today(), stock_queue_length=50, pool_workers=10, day_forecast=1):
    end_date=dcu.check_date(end_date)
    report_root = "C:\\stooq\\reports"
    os.makedirs(report_root, exist_ok=True)
    report_folder = f"{report_root}\\s_and_p_500_report_{end_date}"
    text_file_path = f"{report_folder}\\{day_forecast}_day_forecast_report"
    # Ensure report folder exists
    os.makedirs(report_folder, exist_ok=True)
    # Create empty dataframe
    stock_dataframe = pandas.DataFrame(columns=['Ticker',
                                                'Company',
                                                'Close Value',
                                                'Accuracy',
                                                'Raw Score',
                                                'Score',
                                                'Suggestion'])

    stock_queue = []
    for single_stock in STOCK_TICKER_COMPENDIUM.s_and_p_list:
        stock_queue.append(single_stock)
        if len(stock_queue) == stock_queue_length:
            # Create tuple list
            passed_in_args = []
            for queued_stock in stock_queue:
                passed_in_args.append((queued_stock, end_date, day_forecast))
            # Create pool and save results
            with multiprocessing.Pool(pool_workers) as p:
                pooled_results = p.starmap(report_worker, passed_in_args)
            # Update based on results
            for stock_position, queued_stock in enumerate(stock_queue):
                returned_stock_dict = pooled_results[stock_position]
                if returned_stock_dict is not None:
                    stock_dataframe.loc[len(stock_dataframe.index)] = [
                        queued_stock,
                        STOCK_TICKER_COMPENDIUM.ticker_dict[queued_stock]['company'],
                        returned_stock_dict['close_value'],
                        returned_stock_dict['accuracy'],
                        returned_stock_dict['raw_prediction'],
                        returned_stock_dict['prediction'],
                        'BUY' if returned_stock_dict['prediction'] == 1 else 'SELL'
                    ]
            stock_dataframe.to_csv(text_file_path)
            # Clear queue
            stock_queue = []
    # Sort the dataframe by the accuracy, then the raw_prediction
    stock_dataframe = stock_dataframe.sort_values(['Accuracy', 'Raw Score'], ascending=[False, False])
    # Save two new dataframes, representing the top 100 and bottom 100 bids
    df_good = stock_dataframe.loc[(stock_dataframe['Accuracy'] >= 0.98) & (stock_dataframe['Suggestion'] == 'BUY')]
    df_bad = stock_dataframe.loc[(stock_dataframe['Accuracy'] >= 0.98) & (stock_dataframe['Suggestion'] == 'SELL')]
    df_good = df_good.sort_values(['Raw Score'], ascending=[False])
    df_bad = df_bad.sort_values(['Raw Score'], ascending=[True])
    # Save all three dataframes as csv to report folder
    stock_dataframe.to_csv(text_file_path)
    df_good.to_csv(f"{text_file_path}_BUY")
    df_bad.to_csv(f"{text_file_path}_SELL")


if __name__ == '__main__':
    generate_daily = True
    generate_s_and_p = True
    if generate_s_and_p:
        generate_s_and_p_report(day_forecast=1)
        # generate_s_and_p_report(day_forecast=2)
        # generate_s_and_p_report(day_forecast=3)
    if generate_daily:
        generate_daily_report()























