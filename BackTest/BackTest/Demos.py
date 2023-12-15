import pandas as pd
import BackTest.Preprocessing as Preprocessing
import os

def get_reversal_signal(n:int = 5, start_time: pd.Timestamp = pd.Timestamp('2020-01-02 00:00:00'), end_time: pd.Timestamp = pd.Timestamp('2022-12-29 00:00:00'), Stock_list:list or str = 'all'):
    """
    n: to make a n-day reversal signal.
    start_time: The first signal date (in fact, strictly speaking, the first batch of signals occurring n days after this date)
    end_time: The end signal date.
    Stock_list: if it's 'all': every stock in the signal period will have a signal per day. Else: only selected stocks will have signal.

    return: a standard df input with three columns: date, stk_id and signal.
    """
    if not os.path.exists('./data/data.h5'):
        Preprocessing.update()   
    basic_data = pd.read_hdf('./data/data.h5')
    basic_data['selected_1'] = basic_data['date'].apply(lambda x: x >= start_time)
    basic_data['selected_2'] = basic_data['date'].apply(lambda x: x <= end_time)
    if Stock_list == 'all':
        basic_data['selected_3'] = 1
    else:
        basic_data['selected_3'] = basic_data['stk_id'].apply(lambda x: x in Stock_list)
    basic_data['selected'] = basic_data['selected_1'] * basic_data['selected_2'] * basic_data['selected_3']
    basic_data = basic_data[basic_data['selected'] == 1]
    basic_data = basic_data.sort_values('date')
    basic_data['signal'] = -basic_data.groupby('stk_id')['lead_return'].shift().rolling(n).mean()
    return basic_data[['date', 'stk_id', 'signal']]


def get_MACD_signal(short:int = 5, long = 20, start_time: pd.Timestamp = pd.Timestamp('2020-01-02 00:00:00'), end_time: pd.Timestamp = pd.Timestamp('2022-12-29 00:00:00'), Stock_list:list or str = 'all'):
    """
    n: to make a n-day reversal signal.
    start_time: The first signal date (in fact, strictly speaking, the first batch of signals occurring n days after this date)
    end_time: The end signal date.
    Stock_list: if it's 'all': every stock in the signal period will have a signal per day. Else: only selected stocks will have signal.

    return: a standard df input with three columns: date, stk_id and signal.
    """
    if not os.path.exists('./data/data.h5'):
        Preprocessing.update()   
    basic_data = pd.read_hdf('./data/data.h5')
    basic_data['selected_1'] = basic_data['date'].apply(lambda x: x >= start_time)
    basic_data['selected_2'] = basic_data['date'].apply(lambda x: x <= end_time)
    if Stock_list == 'all':
        basic_data['selected_3'] = 1
    else:
        basic_data['selected_3'] = basic_data['stk_id'].apply(lambda x: x in Stock_list)
    basic_data['selected'] = basic_data['selected_1'] * basic_data['selected_2'] * basic_data['selected_3']
    basic_data = basic_data[basic_data['selected'] == 1]
    basic_data = basic_data.sort_values('date')
    basic_data['lag_return'] = -basic_data.groupby('stk_id')['lead_return'].shift()
    basic_data['short_ema'] = basic_data['lag_return'].ewm(short).mean()
    basic_data['long_ema'] = basic_data['lag_return'].ewm(long).mean()
    basic_data['signal'] = basic_data['short_ema'] - basic_data['long_ema']
    return basic_data[['date', 'stk_id', 'signal']]


def standard_input_demo():
    demo = pd.read_hdf('./demo_signal_df.h5', 'demo')
    return demo