import pandas as pd
import numpy as np
from BackTest.Benchmark import get_benchmark_return
from typing import Union

def is_input_standard(standard_input: pd.DataFrame,single = False):
    """
    This method is used to check whether the signal input is of the standard format as i asked.
    return: boolean.
    """
    to_check_columns = standard_input.columns
    if 'date' in to_check_columns[:2] and 'stk_id' in to_check_columns[:2] and len(to_check_columns) > 2:
        if single:
            if len(to_check_columns) == 3:
                return 1
            else:
                return 0
        else:
            return 1
    else:
        return 0
    

def get_Rsquared(standard_input: pd.DataFrame, stock_list: Union[None, list] = None):
    """
    This method is used to get the Rsquared of ONE signal and the lead-one-day-return.
    """
    signal_name = standard_input.columns[2]
    basic_data = pd.read_hdf('./data/data.h5')
    benchmark_return = get_benchmark_return(standard_input, stock_list = stock_list)
    benchmark_return.name = 'Benchmark'
    basic_data = pd.merge(basic_data, benchmark_return, 'left', 'date')
    basic_data['exc_lead_return'] = basic_data['lead_return'] - basic_data['Benchmark']
    merged_df = pd.merge(basic_data, standard_input, 'inner', ['stk_id', 'date'])
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_df.dropna(inplace= True)
    corr_0 =  merged_df[[signal_name, 'lead_return', 'exc_lead_return']].corr().iloc[0,1]
    corr_1 =  merged_df[[signal_name, 'lead_return', 'exc_lead_return']].corr().iloc[0,2]
    return corr_0 * abs(corr_0), corr_1 * abs(corr_1)


def get_trade_days(signal: pd.DataFrame):
    date = signal['date'].sort_values()
    start_date = date.iloc[0]
    end_date = date.iloc[-1]
    span = (end_date - start_date).days
    return span

def get_corr_matrix(signal: pd.DataFrame):
    processed_signal = signal.iloc[:, 2:]
    processed_signal.replace([np.inf, -np.inf], np.nan)
    processed_signal.fillna(0)
    return processed_signal.corr()