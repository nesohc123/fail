import BackTest.PnLcalculating as PnLcalculating
import BackTest.Utils as Utils
import BackTest.Visualization as Visualization
import BackTest.Benchmark as Benchmark
import BackTest.Optimizers as Optimizers
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
from typing import Union


class Single_BackTest:    
    def __init__(self) -> None:
        self.isFitted = False
        self.R_squared_0 = None
        self.R_squared_1 = None
        self.real_ret = None
        self.exc_ret = None
        self.ann_real_ret = None
        self.ann_exc_ret = None
        self.real_vol = None
        self.exc_vol = None
        self.drawdown = None
        self.exc_drawdown = None
        self.Sharpe = None
        self.exc_Sharp = None
        self.LS_result = None

    def fit(self, signal_df, groups: Union[int, None] = None, stock_list: Union[list, None] = None, limits_on_long = False):
        if not Utils.is_input_standard(signal_df, True):
            print('This input is not standard, plz check README.md')
            return -1
        max_cpu = os.cpu_count()

        groups_ = signal_df.groupby('date')
        df_ls = []
        signal_name = signal_df.columns[2]
        for date, group in groups_:
            group['date'] = date
            length = group.shape[0]
            group[signal_name] = (group[signal_name].rank() / length - 0.5)
            df_ls.append(group)
        signal_df = pd.concat(df_ls).sort_values('date')

        if stock_list:
            if not limits_on_long:
                signal_df['limits'] = signal_df['stk_id'].apply(lambda x: int(x not in stock_list))
                signal_name = signal_df.columns[2]
                signal_df['limits'] = signal_df['limits'] * (signal_df[signal_name] < 0).astype(int)
                signal_df['limits'] = signal_df['limits'].apply(lambda x: np.nan if x == 1 else 1)
                signal_df[signal_name] *= signal_df['limits']
                signal_df = signal_df.iloc[:,:3]
            else:
                signal_df['limits'] = signal_df['stk_id'].apply(lambda x: int(x not in stock_list))
                signal_name = signal_df.columns[2]
                signal_df['limits'] = signal_df['limits'].apply(lambda x: np.nan if x == 1 else 1)
                signal_df[signal_name] *= signal_df['limits']
                signal_df = signal_df.iloc[:,:3]

        with mp.Pool(min(max_cpu, 4)) as pool:
            result1 = pool.apply_async(Utils.get_Rsquared, args= (signal_df, stock_list))
            result2 = pool.apply_async(PnLcalculating.get_daily_Pnl, args= (signal_df,))
            if not (groups is None):
                result2_LS = pool.apply_async(PnLcalculating.get_daily_Pnl_LS, args= (groups, signal_df,))
            result3 = pool.apply_async(Utils.get_trade_days, args= (signal_df,))
            result4 = pool.apply_async(Benchmark.get_benchmark_return, args= (signal_df, stock_list,))
            pool.close()
            pool.join()

        r0, r1 = result1.get()
        pnl_series, exc_pnl_series = result2.get()
        cum_pnl_series = (1 + pnl_series).cumprod()
        cum_exc_pnl_series = (1 + exc_pnl_series).cumprod()
        peak = cum_pnl_series.expanding().max()
        draw_down = (peak - cum_pnl_series).apply(lambda x: max(0, x)) / (peak + 1e-10)
        exc_peak = cum_exc_pnl_series.expanding().max()
        exc_draw_down = (exc_peak - cum_exc_pnl_series).apply(lambda x: max(0, x)) / (exc_peak + 1e-10)
        span = result3.get()
        benchmark_return = result4.get()

        if not (groups is None):
            self.LS_result = result2_LS.get()

        self.pnl = pnl_series
        
        self.benchmark_return = benchmark_return
        self.cum_pnl = cum_pnl_series
        self.cum_exc_pnl = cum_exc_pnl_series
        self.real_ret = self.cum_pnl.iloc[-1] - 1
        self.exc_ret = self.cum_exc_pnl.iloc[-1] - 1
        self.real_vol = pnl_series.std()
        self.exc_vol = exc_pnl_series.std()
        self.ann_real_ret = self.real_ret / span * 365
        self.ann_exc_ret = self.exc_ret / span * 365
        self.Sharpe = self.real_ret / self.real_vol / 100
        self.exc_Sharp = self.exc_ret / self.exc_vol / 100
        self.R_squared_0 = r0
        self.R_squared_1 = r1
        self.drawdown = draw_down.max()
        self.exc_drawdown = exc_draw_down.max()
        self.isFitted = True
        return 0
    
    def show(self, excess: bool = False, benchmark: bool = True):
        if not self.isFitted:
            print('You need to fit the model first, then we can show you the result.')
            return -1
        Visualization.show(excess, benchmark, self.cum_pnl, self.cum_exc_pnl, self.benchmark_return)
        print(f"the annual return is: {self.ann_real_ret}\nthe excess annual return is :{self.ann_exc_ret}\nthe Sharpe ratio is: {self.Sharpe}\nthe excess Sharpe ratio is: {self.exc_Sharp}\nthe maximum drawdown is: {self.drawdown}\nthe excess return max-drawdown is: {self.exc_drawdown}\nthe R-squared of this signal is: {self.R_squared_0}\nthe excess_Rsquard of this signal is: {self.R_squared_1}")
        return 0 
    
    def LnS_PnL_show(self, excess:bool = True, benchmark:bool = True):
        if not self.isFitted:
            print('You need to fit the model first, then we can show you the result.')
            return -1
        if not self.LS_result:
            print('You need to give groups parameter when use fit method')
            return -1
        Visualization.LnS_Pnl_show(excess, benchmark, self.LS_result, self.benchmark_return)
        print(f"the annual return is: {self.ann_real_ret}\nthe excess annual return is :{self.ann_exc_ret}\nthe Sharpe ratio is: {self.Sharpe}\nthe excess Sharpe ratio is: {self.exc_Sharp}\nthe maximum drawdown is: {self.drawdown}\nthe excess return max-drawdown is: {self.exc_drawdown}\nthe R-squared of this signal is: {self.R_squared_0}\nthe excess_Rsquard of this signal is: {self.R_squared_1}")
        return 0

class Multiple_BackTest(Single_BackTest):
    def __init__(self) -> None:
        super().__init__()
        self.corr_matrix = None
        self.reference_weight = None
        self.R_squared_list = None
        self.R_squared_total = None
        return None
    
    def multiple_signals_fit(self, signal_df: pd.DataFrame, train_ratio: float = 0.8, optimizer:Union[str, callable] = 'Linear', groups: Union[int, None] = None, stock_list: Union[None, list] = None, exc= True, limits_on_long = False):
        if not Utils.is_input_standard(signal_df):
            print('This input is not standard, plz check README.md')
            return -1
        max_cpu = os.cpu_count()
        if optimizer == 'Linear':
            optimizer = Optimizers.LinearOptimizer
        with mp.Pool(min(max_cpu, 2)) as pool:
            result1 = pool.apply_async(Utils.get_corr_matrix, args= (signal_df,))
            result2 = pool.apply_async(optimizer, args= (signal_df, train_ratio, stock_list, exc))
            pool.close()
            pool.join()
        
        self.corr_matrix = result1.get()
        composed_df, self.reference_weight = result2.get()
        
        super().fit(composed_df[['date', 'stk_id', 'composed_signal']], groups=groups, stock_list= stock_list, limits_on_long= limits_on_long)
        if exc:
            self.R_squared_total = self.R_squared_1
        else:
            self.R_squared_total = self.R_squared_0
        
        signals = composed_df.shape[1] - 3
        tmp_df_list = [composed_df.iloc[:,[0,1,i]] for i in range(2,2+signals)]
        with mp.Pool(min(max_cpu, signals)) as pool:
            R_list = pool.map(Utils.get_Rsquared, tmp_df_list)
        if exc:
            self.R_squared_list = [R[1] for R in R_list]
        else:
            self.R_squared_list = [R[0] for R in R_list]
        
        return 0

    def show_corr(self):
        if not self.isFitted:
            print('You need to fit the model first, then we can show you the result.')
            return -1
        else:
            Visualization.show_corr(self.corr_matrix)
            return 0 

