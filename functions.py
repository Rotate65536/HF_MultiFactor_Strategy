import pandas as pd
import numpy
import os


def get_fc_df(directory):
    fc_list = []
    for path in os.listdir(directory):
        fc = pd.read_feather(os.path.join(directory, path))
        fc_list.append(fc)
    fc_df = pd.concat(fc_list)
    print(f'fc_df is {type(fc_df)}')
    return fc_df


def get_ohclv_df(directory):
    price_list = []
    for path in os.listdir(directory):
        price = pd.read_feather(os.path.join(directory, path))
        price_list.append(price)
    price_df = pd.concat(price_list)
    price_df = price_df.reset_index(drop=True)
    price_df = price_df[['time', 'code', 'close']]
    price_df = price_df.set_index(['time', 'code']).unstack()['close']
    print(f'price_df is {type(price_df)}')
    return price_df
