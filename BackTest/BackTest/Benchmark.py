import numpy as np
import pandas as pd
from BackTest.Preprocessing import truncate
from typing import Union

def get_benchmark_return(standard_input: pd.DataFrame, stock_list: Union[list, None] = None):
    """
    standard_input should be a pandas.DataFrame Object with columns ['date', 'stk_id', 'signal(s)...']
    return: pd.Series which shows the dily benchmark lead return in the period of dates with signals.
    """
    basic_data = pd.read_hdf('./data/data.h5')
    start_time, end_time = truncate(standard_input)
    basic_data = basic_data[basic_data['date'] >= start_time]
    basic_data = basic_data[basic_data['date'] <= end_time]
    if stock_list != None:
        basic_data['filter'] = basic_data['stk_id'].apply(lambda x: x in stock_list)
        basic_data = basic_data[basic_data['filter'] == True]
    basic_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    basic_data.dropna(inplace = True)
    benchmark_lead_return = basic_data.groupby('date').apply(lambda x: (x['lead_return'] * x['float_market_value']).sum() / x['float_market_value'].sum())
    return benchmark_lead_return