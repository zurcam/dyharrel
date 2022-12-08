'''
Doc string here ...
'''
import datetime
import time
import random
import pandas
import dateutil.parser as dp
import requests


class OnlineStockData:
    def __init__(self):
        self.nasdaq_screener = 'https://www.nasdaq.com/market-activity/stocks/screener'


def stock_list_dataframe():

    return pandas.read_csv(
        f"{str(__file__).replace('data_calls_util.py', 'nasdaq_screener.csv')}",
        index_col=False,
        na_filter=False
    )


def s_and_p_dataframe():
    return pandas.read_csv(
        f"{str(__file__).replace('data_calls_util.py', 's_and_p_500_companies.csv')}",
        index_col=False,
        na_filter=False
    )


def check_date(checked_date, business_days=True):
    """ Checks that given date can be parsed as a date. """
    try:
        checked_date = dp.parse(str(checked_date), fuzzy=False).date()
    except ValueError:
        raise Exception(f"{str(checked_date)} cannot be converted into datetime.")
    finally:
        if business_days:
            day_of_week = checked_date.weekday()
            spread_check = day_of_week - 4
            if spread_check > 0:
                checked_date = checked_date - datetime.timedelta(days=spread_check)

    return checked_date


def get_online_stock_data(
        online_stock_csv_url,
        date_start,
        date_end
):
    time.sleep(random.randint(random.randint(20, 40), random.randint(70, 120)) / 10)
    raw_dataframe = pandas.read_csv(
        online_stock_csv_url,
        index_col=False
    )
    raw_dataframe[raw_dataframe.columns[0]] = \
        pandas.to_datetime(raw_dataframe[raw_dataframe.columns[0]])
    raw_dataframe.set_index(raw_dataframe.columns[0], inplace=True)
    raw_dataframe.sort_index()
    # filter by dates
    returned_dataframe = raw_dataframe.loc[
        (raw_dataframe.index >= pandas.Timestamp(date_start))
        & (raw_dataframe.index <= pandas.Timestamp(date_end))
    ]
    returned_dataframe = returned_dataframe.apply(pandas.to_numeric, errors='coerce')
    raw_dataframe = None
    return returned_dataframe










