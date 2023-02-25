import os

import tqdm
import numba
import pandas as pd
import numpy as np
from functools import wraps


# from gpliu.data import get_trade_date


def try_except(func):
    @wraps(func)
    def decorated(*args, **kargs):
        try:
            return func(*args, **kargs)
        except:
            return np.nan

    return decorated


@try_except
@numba.jit(signature_or_function=numba.float32(numba.float32[:]), nopython=True, cache=True, nogil=True,
           boundscheck=True)
def __sub_QUA__(x):
    x = np.sort(x)[:-10]
    return (np.percentile(x, 10) - np.min(x)) / (np.max(x) - np.min(x) + 1)


@try_except
@numba.jit(signature_or_function=numba.float32(numba.float32[:]), nopython=True, cache=True, nogil=True,
           boundscheck=True)
def __sub_STD__(x):
    x = x[x < np.percentile(x, 10)]
    return np.std(x)


@try_except
@numba.jit(signature_or_function=numba.float32(numba.float32[:]), nopython=True, cache=True, nogil=True,
           boundscheck=True)
def __sub_SKEW__(x):
    x = x[x < np.median(x)]
    return np.mean(((x - np.mean(x)) / np.std(x)) ** 3)


@try_except
@numba.jit(signature_or_function=numba.float32(numba.float32[:]), nopython=True, cache=True, nogil=True,
           boundscheck=True)
def __sub_KURT__(x):
    x = x[x < np.median(x)]
    return np.mean((x - np.mean(x)) ** 4) / np.var(x) ** 2 - 3


@try_except
def __sub_MTS__(x):
    return np.corrcoef(x['amount'], x['money'])[1][0]


@try_except
def __sub_MTE__(x):
    return np.corrcoef(x['amount'], x['close'])[1][0]


@try_except
def __sub_SR__(x, eta=0.2):
    x = x.sort_values(by='amount', ascending=False).iloc[:int(len(x) * eta)]['ret']
    return x.sum()


def next_trade_date(td, date_list, shift=1):
    """
    获取特定时间点的下一个交易日。
    td: str或者datetime，不一定是交易日，但对应信号时间。
    shift: 往后查找的时间，默认往后推一天。
    """
    return get_trade_date(start_date=td, count=shift + 1, date_list=date_list)[-1]


def get_trade_date(start_date, count, date_list):
    i = 0
    for i in range(len(date_list)):
        if date_list[i] == str(start_date):
            break
    if i + count > len(date_list):
        return date_list[i:len(date_list)]
    else:
        return date_list[i:i + count]


class HqFactor:
    def __init__(self,
                 date_list_path='data\\factor_calculation_data\\amount',
                 price_path='data\\factor_calculation_data\\high-freq-ftr2\\{}.ftr',
                 amount_path='data\\factor_calculation_data\\amount\\amount_{}.ftr'
                 ):
        self.date_list = []
        for item in os.listdir(date_list_path):
            self.date_list.append(item.split('_')[1].split('.')[0])

        np.seterr(divide='ignore', invalid='ignore')
        self.price_path = price_path
        self.amount_path = amount_path
        tds = get_trade_date(start_date=self.date_list[0], count=20, date_list=self.date_list)
        data = []
        for td in tqdm.tqdm(tds):
            data.append(self.__load_data__(td))
        self.data = pd.concat(data)
        self.amount = self.data[['code', 'date', 'amount']].copy()
        self.last_date = td

    def __load_data__(self, td):
        amount = pd.read_feather(self.amount_path.format(td))
        price = pd.read_feather(
            self.price_path.format(td), columns=['code', 'datetime', 'open', 'close', 'money']
        ).rename(columns={'datetime': 'Time'})
        price['ret'] = price['close'] / price['open'] - 1
        data_ = pd.merge(amount, price)
        data_['date'] = data_['Time'].apply(lambda x: x.date())
        return data_.drop(columns=['Time', 'open'])

    def __renew_data__(self):
        first_date, last_date = self.data['date'].min(), self.data['date'].max()
        new_data = self.__load_data__(next_trade_date(last_date, shift=1, date_list=self.date_list))
        self.data = pd.concat([self.data[self.data['date'] != first_date], new_data])
        self.amount = self.data[['code', 'date', 'amount']].copy()
        self.last_date = self.data['date'].max()

    def get_data(self):
        return self.data

    def get_factor_date(self):
        return self.last_date

    def QUA(self):
        sub_QUA = self.amount.groupby(['date', 'code']) \
            .apply(lambda x: __sub_QUA__(x['amount'].values)).reset_index()
        qua = sub_QUA.groupby('code').mean()
        qua.columns = ['QUA']
        return qua

    def STD(self):
        sub_STD = self.amount.groupby(['date', 'code']) \
            .apply(lambda x: __sub_STD__(x['amount'].values)).reset_index()
        std = sub_STD.groupby('code').mean()
        std.columns = ['STD']
        return std

    def SKEW(self):
        sub_SKEW = self.amount.groupby(['date', 'code']) \
            .apply(lambda x: __sub_SKEW__(x['amount'].values)).reset_index()
        skew = sub_SKEW.groupby('code').mean()
        skew.columns = ['SKEW']
        return skew

    def KURT(self):
        sub_KURT = self.amount.groupby(['date', 'code']) \
            .apply(lambda x: __sub_KURT__(x['amount'].values)).reset_index()
        kurt = sub_KURT.groupby('code').mean()
        kurt.columns = ['KURT']
        return kurt

    def MTS(self):
        sub_MTS = self.data[['code', 'date', 'amount', 'money']].groupby(['date', 'code']) \
            .apply(__sub_MTS__).reset_index()
        mts = sub_MTS.groupby('code').mean()
        mts.columns = ['MTS']
        return mts

    def MTE(self):
        sub_MTE = self.data[['code', 'date', 'amount', 'close']].groupby(['date', 'code']) \
            .apply(__sub_MTE__).reset_index()
        mte = sub_MTE.groupby('code').mean()
        mte.columns = ['MTE']
        return mte

    def SR(self):
        sr = self.data[['code', 'amount', 'ret']].groupby('code').apply(__sub_SR__)
        sr.name = 'SR'
        return sr

    def get_all_factor(self):
        qua = self.QUA()
        std = self.STD()
        skew = self.SKEW()
        kurt = self.KURT()
        mts = self.MTS()
        mte = self.MTE()
        sr = self.SR()
        factor = pd.concat([qua, std, skew, kurt, mts, mte, sr], axis=1)
        factor['time'] = self.last_date
        return factor.reset_index()
