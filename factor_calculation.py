import os, datetime
from HqFactor import HqFactor

hf = HqFactor(
    date_list_path='data\\factor_calculation_data\\amount',
    price_path='data\\factor_calculation_data\\high-freq-ftr2\\{}.ftr',
    amount_path='data\\factor_calculation_data\\amount\\amount_{}.ftr'
)
output_path = 'data/factor_mainforce'
os.mkdir(output_path)
if __name__ == '__main__':
    factor_date = ''
    while factor_date != '2022-01-28':
        factor = hf.get_all_factor()
        factor_date = hf.get_factor_date()
        factor.to_feather(os.path.join(output_path, f'mainforce_{factor_date}.ftr'))
        print(f'mainforce_{factor_date}.ftr finished at {datetime.datetime.now()}')
        hf.__renew_data__()
    pass
