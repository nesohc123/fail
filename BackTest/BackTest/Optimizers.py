from statsmodels.api import OLS
import statsmodels.api as sms
import pandas as pd
import numpy as np
from BackTest.Benchmark import get_benchmark_return

def LinearOptimizer(signals_df, train_ratio, stock_list, exc:bool = True):
    signal_names = list(signals_df.columns[2:])
    basic_data = pd.read_hdf('./data/data.h5')
    benchmark_return = get_benchmark_return(signals_df, stock_list = stock_list)
    benchmark_return.name = 'Benchmark'
    basic_data = pd.merge(basic_data, benchmark_return, 'left', 'date')
    basic_data['exc_lead_return'] = basic_data['lead_return'] - basic_data['Benchmark']
    merged_df = pd.merge(basic_data, signals_df, 'inner', ['stk_id', 'date'])
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_df.dropna(inplace= True)
    days = sorted(list(set(merged_df['date'])))
    train_days = int(train_ratio * len(days))
    split_point = days[train_days]
    train_data = merged_df[merged_df['date'] < split_point]
    test_data = merged_df[merged_df['date'] >= split_point]
    if exc == True:
        model = OLS(train_data['exc_lead_return'], sms.add_constant(train_data[signal_names]))
    else:
        model = OLS(train_data['lead_return'], sms.add_constant(train_data[signal_names]))
    result = model.fit()
    params = result.params[1:]
    test_data['composed_signal'] = (test_data[signal_names] * params).sum(axis = 1)
    return test_data[['date', 'stk_id'] + signal_names + ['composed_signal']], params