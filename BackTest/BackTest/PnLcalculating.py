import numpy as np
import pandas as pd
from BackTest.Benchmark import get_benchmark_return

def get_daily_Pnl(signal_input: pd.DataFrame):
    """
    the procedure that check whether signal_input is of standard format has already been done in BackTest Module
    this input_df has and only has three columns: date, stk_id and ONLY ONE signal.

    return: pd.Series
    """
    signal_input.replace([np.inf, -np.inf], np.nan, inplace=True)
    signal_input.dropna(inplace= True)
    signal_name = signal_input.columns[2]

    basic_data = pd.read_hdf('./data/data.h5')
    benchmark_return = get_benchmark_return(signal_input)
    benchmark_return.name = 'Benchmark'
    basic_data = pd.merge(basic_data, benchmark_return, 'left', 'date')
    basic_data['exc_lead_return'] = basic_data['lead_return'] - basic_data['Benchmark']
    merged_df = pd.merge(basic_data, signal_input, 'left', ['stk_id', 'date'])

    Pnl_series = merged_df.groupby('date').apply(lambda x: (x[signal_name] * x['lead_return']).sum() / (x[signal_name]).abs().sum())
    exc_Pnl_series = merged_df.groupby('date').apply(lambda x: (x[signal_name] * x['exc_lead_return']).sum() / (x[signal_name]).abs().sum())
    return Pnl_series, exc_Pnl_series

def get_daily_Pnl_LS(groups: int, signal_input: pd.DataFrame):
    """
    the procedure that check whether signal_input is of standard format has already been done in BackTest Module
    this input_df has and only has three columns: date, stk_id and ONLY ONE signal.

    return: (list_1, list_2, list_3, list_4), each list has length == groups, 
    list_1 contains the R_square of all groups, list_2 contains the exc_R_square of all groups, 
    list_3 coontains the daily pnl for each group of this signal and the last list contains the excess daily pnl.

    Specially, if groups == 1, this function will just call the simple function.
    """
    if groups == 1:
        return get_daily_Pnl(signal_input)
    else:
        signal_input.replace([np.inf, -np.inf], np.nan, inplace=True)
        signal_input.dropna(inplace= True)
        signal_name = signal_input.columns[2]

        basic_data = pd.read_hdf('./data/data.h5')
        benchmark_return = get_benchmark_return(signal_input)
        benchmark_return.name = 'Benchmark'
        basic_data = pd.merge(basic_data, benchmark_return, 'left', 'date')
        basic_data['exc_lead_return'] = basic_data['lead_return'] - basic_data['Benchmark']
        merged_df = pd.merge(basic_data, signal_input, 'left', ['stk_id', 'date'])

        merged_df['group'] = (merged_df[signal_name] + 0.5) // (1 / groups)

        list_1 = []
        list_2 = []
        list_3 = []
        list_4 = []
        for group in range(groups):
            group_df = merged_df[merged_df['group'] == group]
            if group_df.shape[0] == 0:
                list_1.append(np.nan)
                list_2.append(np.nan)
                list_3.append(None)
                list_4.append(None)
            else:
                corr_0= (group_df[signal_name] * group_df['lead_return']).sum() / ((group_df[signal_name] ** 2).sum() ** 0.5 * (group_df['lead_return'] ** 2).sum() ** 0.5)
                group_r2_0 = corr_0 * abs(corr_0)
                corr_1= (group_df[signal_name] * group_df['exc_lead_return']).sum() / ((group_df[signal_name] ** 2).sum() ** 0.5 * (group_df['exc_lead_return'] ** 2).sum() ** 0.5)
                group_r2_1 = corr_1 * abs(corr_1)
                group_pnl = group_df.groupby('date').apply(lambda x: (x[signal_name] * x['lead_return']).sum() / (x[signal_name]).abs().sum())
                group_exc_pnl = group_df.groupby('date').apply(lambda x: (x[signal_name] * x['exc_lead_return']).sum() / (x[signal_name]).abs().sum())
                list_1.append(group_r2_0)
                list_2.append(group_r2_1)
                list_3.append(group_pnl)
                list_4.append(group_exc_pnl)
        return list_1, list_2, list_3, list_4
