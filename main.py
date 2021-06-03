import pandas as pd
import numpy as np
from pandas_datareader import data as web
from matplotlib import pyplot as plt
import datetime as dt
import config as cfg

def main():
    startdate = dt.date(2021,5,27)
    enddate = dt.date(2021,6,1)
    apikey = cfg.ALPHA_VANTAGE_API_KEY
    df_appl = web.DataReader('AAPL', 'av-daily', startdate, enddate, api_key=apikey)
    df_msft = web.DataReader('MSFT', 'av-daily', startdate, enddate, api_key=apikey)
    df_fb = web.DataReader('FB', 'av-daily', startdate, enddate, api_key=apikey)
    print(df_appl.head()['close'])
    print(df_msft.head()['close'])
    print(df_fb.head()['close'])
    
    df_mix = pd.DataFrame({'Apple': df_appl['close'],'Microsoft': df_msft['close'], 'Facebook': df_fb['close']})
    df_mix.plot(figsize=(8,6),fontsize=18)
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1, fontsize=18)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()