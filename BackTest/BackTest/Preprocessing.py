import feather
import os
import pandas as pd

def update(original_data: str = 'stk_daily.feather'):
    """
    original_data: name of the data file. e.g.: stk_daily.feather (only support feather files)
    return: 0 (successfully run), -1(interrupted), other errorcodes
    RaiseError: FileExistsError, IOError, OSError

    the original data file must be stored in ./data folder
    """
    if not os.path.exists(f'./data/{original_data}'):
        raise FileExistsError("the original_data path doesn't exist")
    
    try:
        stk_daily = feather.read_dataframe(f'./data/{original_data}')
    except Exception as e:
        print(f"An error occurred: {e}")
    
    if os.path.exists('./data/data.h5'):
        print('there has an old version of data.h5, this process will overwrite it, input "Y" to continue.')
        if input() != 'Y':
            print('process interrupted.')
            return -1
        else:
            try:
                os.remove('./data/data.h5')
            except Exception as e:
                print(f"An error occurred: {e}")
    else:
        print('there is no data.h5 file, this process will create one.')
    
    try:
        stk_daily['adj_open'] = stk_daily['open'] * stk_daily['cumadj']
        stk_daily = stk_daily.sort_values(by = ['stk_id', 'date'], ascending = True)
        stk_daily['lead_return'] = stk_daily.groupby('stk_id')['adj_open'].diff().shift(-1)
        stk_daily['lead_return'] /= stk_daily['adj_open']
        selected_df = stk_daily[['open', 'stk_id', 'date', 'lead_return']].dropna()

        map_df = feather.read_dataframe('./data/stk_fin_item_map.feather')
        map_df = map_df.set_index('item')
        item_1 = map_df.loc['基本每股收益', 'field']
        item_2 = map_df.loc['净利润', 'field']
        IS = feather.read_dataframe('./data/stk_fin_income.feather')
        IS = IS.loc[:,['date', 'stk_id', item_1, item_2]]
        IS['float_share'] = IS[item_2] / IS[item_1]

        IS = IS[['date', 'stk_id', 'float_share']]
        IS.dropna(inplace= True)

        new_data = pd.merge(selected_df, IS, 'outer', on = ['date', 'stk_id'])
        new_data = new_data.sort_values('date', ascending= True)
        new_data['float_share'] = new_data.groupby('stk_id')['float_share'].fillna(method='ffill')
        new_data.dropna(inplace = True)
        new_data = new_data.sort_values(by = ['stk_id', 'date'])
        new_data['float_market_value'] = new_data['float_share'] * new_data['open']

        new_data.to_hdf('./data/data.h5', 'basic_data')
    except Exception as e:
        print(f"An error occurred: {e}")

    return 0

def truncate(standard_input: pd.DataFrame):
    """
    standard_input should be a pandas.DataFrame Object with columns ['date', 'stk_id', 'signal(s)...']
    This method should not be explicitly called by users.
    """
    try:
        date = standard_input['date']
    except Exception as e:
        print(f'An error occured:{e}, pls check whether the input is in stantdard form.')
    return min(date), max(date)