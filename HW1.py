# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 01:11:06 2019

@author: Qiao Guanzhuo
"""

import pandas as pd
from pandas_datareader import data as pdrd
import pandas_datareader.yahoo.options as pdro
import datetime as dt
import math as m
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
import time
import os



os.chdir('D:\\Grad 2\\621\\assignment\\HM1\\data') 


#1.1
def download(stocklist):
    starttime = dt.datetime(2018,1,1)
    endtime = dt.date.today()
    for stock in stocklist:
        equity_data = pdrd.DataReader(stock, data_source='yahoo',start=starttime,end=endtime)
        equity_data.to_csv('{}.{} {} equity.csv'.format(endtime.month,endtime.day,stock))
        option_name = pdro.Options(stock)
        option_data = option_name.get_forward_data(3,call=True,put=True)
        option_data.to_csv('{}.{} {} option.csv'.format(endtime.month,endtime.day,stock))


#1.2
#download(['amzn','spy','^vix'])
AMZN_option1 = pd.read_csv('2.11 amzn option.csv')
SPY_option1 = pd.read_csv('2.11 spy option.csv')
VIX_option1 = pd.read_csv('2.11 ^vix option.csv')
AMZN_option2 = pd.read_csv('2.12 amzn option.csv')
SPY_option2 = pd.read_csv('2.12 spy option.csv')
VIX_option2 = pd.read_csv('2.12 ^vix option.csv')


Data1 = AMZN_option1.append(SPY_option1).append(VIX_option1)
Data1 = Data1.set_index(['Underlying',Data1.index])
Data2 = AMZN_option2.append(SPY_option2).append(VIX_option2)
Data2 = Data2.set_index(['Underlying',Data2.index])
del AMZN_option1,SPY_option1 ,VIX_option1,AMZN_option2,SPY_option2 ,VIX_option2
#clean data
Data1 = Data1.drop_duplicates()
Data2 = Data2.drop_duplicates()
Data1 = Data1[Data1.notnull()]
Data2 = Data2[Data2.notnull()]
Data1 = Data1[['Strike','Expiry','Type','Last','Bid','Ask','Vol','Underlying_Price']]
Data2 = Data2[['Strike','Expiry','Type','Last','Bid','Ask','Vol','Underlying_Price']]

#bonus
def download_combine_equity(stocklist,stock_startt, stock_endt):
    equitylist = []
    for stock in stocklist:
        equitylist.append(pdrd.DataReader(stock, data_source='yahoo',start=stock_startt,end=stock_endt))
    equitydata = pd.merge(equitylist[0],equitylist[1],how='inner',left_index=True,right_index = True,suffixes=('_'+stocklist[0],'_'+stocklist[1]))
    for i in range(2,len(equitylist)):
        equitylist[i].columns = [name+'_'+stocklist[i] for name in equitylist[i]]
        equitydata = pd.merge(equitydata,equitylist[i],how='inner',left_index=True,right_index = True)
    equitydata.to_csv('combined equity data.csv')
download_combine_equity(['amzn','spy','^vix'],'2019-01-01', '2019-03-01')
def combine_options(options_file_list):
    optionslist=[]
    Download_Date = []
    for file_name in options_file_list:
        Download_Date.append(file_name.split(' ')[0])
        optionslist.append(pd.read_csv(file_name))
        optionslist[-1]['Download_Date']=Download_Date[-1]
    optionsdata = pd.concat(optionslist,ignore_index=True,)
    optionsdata = optionsdata[['Download_Date']+list(optionsdata.columns[optionsdata.columns != 'Download_Date'])]
    optionsdata.to_csv('combined options data.csv')
combine_options(['2.11 amzn option.csv','2.11 spy option.csv',
                               '2.11 ^vix option.csv','2.12 amzn option.csv',
                               '2.12 spy option.csv','2.12 ^vix option.csv'])
pd.read_csv('combined equity data.csv').head()
pd.read_csv('combined options data.csv').head()
pd.read_csv('combined options data.csv').tail()
#1.3

#1.4
rate=2.4/100
asset_price = pd.DataFrame({'rate':[rate,rate],
                            'AMZN_price':[Data1.loc[('AMZN',0),'Underlying_Price'],
                                         Data2.loc[('AMZN',0),'Underlying_Price']],
                            'SPY_price':[Data1.loc[('SPY',0),'Underlying_Price'],
                                         Data2.loc[('SPY',0),'Underlying_Price']],
                            'VIX_price':[Data1.loc[('^VIX',0),'Underlying_Price'],
                                         Data2.loc[('^VIX',0),'Underlying_Price']]},
                            index = [dt.datetime(2019,2,11),dt.datetime(2019,2,12)])
asset_price.index.names=['Date']
asset_price.to_csv('asset price.csv')
pd.read_csv('asset price.csv').head()

#time to maturity
current_date = dt.datetime(2019,2,11)
Data1['Expiry']=  [dt.datetime.strptime(i,'%Y-%m-%d') for i in Data1['Expiry']]
Data1['TtM'] = [(i-current_date).days/365 for i in Data1['Expiry']]
current_date = dt.datetime(2019,2,12)
Data2['Expiry']=  [dt.datetime.strptime(i,'%Y-%m-%d') for i in Data2['Expiry']]
Data2['TtM'] = [(i-current_date).days/365 for i in Data2['Expiry']]
del current_date
#2.5


def BS_Formula(type_opt, r, vol, K, S, T):
    d_1 = float(m.log(S/K)+(r+vol**2/2)*T)/float(vol*m.sqrt(T))
    d_2 = d_1-vol*m.sqrt(T)
    if type_opt == 'call':
        return norm.cdf(d_1)*S-K*m.exp(-r*T)*norm.cdf(d_2)
    else:
        return K*m.exp(-r*T)*norm.cdf(-d_2)-norm.cdf(-d_1)*S
#6

def Bisection(func,tolerance,up,down):
    if np.sign(func(down)) * np.sign(func(up)) > 0:
        return np.nan
    if abs(func(up))<tolerance:
        return up
    if abs(func(down))<tolerance:
        return down
    mid = (down + up)/2
    while ( abs(func(mid)) > tolerance ):
        if ( np.sign(func(down)) * np.sign(func(mid)) < 0 ):
            up = mid
        else:
            down = mid
        mid = (down + up)/2
    return mid

def get_avrprice(bid,ask):
    return 0.5*(bid+ask)
#Bisection(lambda x: 2*x**3-1 , 0.001,-1,1 )
#BS_Formula('call',0.1,0.2,40,42,0.5)

data_6 = Data1['AMZN':'SPY']
data_6 = data_6[data_6['Vol'] != 0]
data_6 = data_6[(data_6['Ask'] != 0) & (data_6['Bid'] != 0)]
data_6 = data_6[data_6['TtM'] != 0]


def get_iv(type_opt, r, K, S, T, P,tolerance,up,down):
    obj_func= lambda x: BS_Formula(type_opt, r, x, K, S, T)-P
    return Bisection(obj_func,tolerance,up,down)
start = time.time()
for ind in data_6.index:
    data_6.loc[ind,'Avr_Price'] = get_avrprice(data_6.loc[ind,'Bid'],data_6.loc[ind,'Ask'])
    data_6.loc[ind,'Implied_vol_bis'] = get_iv(data_6.loc[ind,'Type'], rate, data_6.loc[ind,'Strike']\
                        , data_6.loc[ind,'Underlying_Price'], data_6.loc[ind,'TtM'], data_6.loc[ind,'Avr_Price'],
                        10**(-6), 7, 0.00001)
# =============================================================================
#     data_6.loc[ind,'CalcuPrice'] = BS_Formula(data_6.loc[ind,'Type'], rate, data_6.loc[ind,'Implied_vol_bis'], data_6.loc[ind,'Strike']\
#                         , data_6.loc[ind,'Underlying_Price'], data_6.loc[ind,'TtM'])
# =============================================================================
end = time.time()
data_6 = data_6.dropna()
timespent1 = end-start
#    data_6.loc[ind,'CalcuPrice'] = BS_Formula(data_6.loc[ind,'Type'], rate, data_6.loc[ind,'IV'], data_6.loc[ind,'Strike']\
#                        , data_6.loc[ind,'Underlying_Price'], data_6.loc[ind,'TtM'])

amzn_at = data_6[(data_6['Strike']/data_6['Underlying_Price']>0.95) & (data_6['Strike']/data_6['Underlying_Price']<1.05)]['AMZN':'AMZN']
spy_at = data_6[(data_6['Strike']/data_6['Underlying_Price']>0.95) & (data_6['Strike']/data_6['Underlying_Price']<1.05)]['SPY':'SPY']
amzn_call_in_avriv =data_6['AMZN':'AMZN'].loc[(data_6['Strike']/data_6['Underlying_Price']<0.95) & 
                   (data_6['Type']=='call')].Implied_vol_bis.mean()

amzn_call_out_avriv=data_6['AMZN':'AMZN'].loc[(data_6['Strike']/data_6['Underlying_Price']>1.05) & 
                   (data_6['Type']=='call')].Implied_vol_bis.mean()

amzn_put_out_avriv =data_6['AMZN':'AMZN'].loc[(data_6['Strike']/data_6['Underlying_Price']<0.95) & 
                   (data_6['Type']=='put')].Implied_vol_bis.mean()

amzn_put_in_avriv=data_6['AMZN':'AMZN'].loc[(data_6['Strike']/data_6['Underlying_Price']>1.05) & 
                   (data_6['Type']=='put')].Implied_vol_bis.mean()

spy_call_in_avriv=data_6['SPY':'SPY'].loc[(data_6['Strike']/data_6['Underlying_Price']<0.95) & 
                   (data_6['Type']=='call')].Implied_vol_bis.mean()

spy_call_out_avriv=data_6['SPY':'SPY'].loc[(data_6['Strike']/data_6['Underlying_Price']>1.05) & 
                   (data_6['Type']=='call')].Implied_vol_bis.mean()

spy_put_out_avriv=data_6['SPY':'SPY'].loc[(data_6['Strike']/data_6['Underlying_Price']<0.95) & 
                   (data_6['Type']=='put')].Implied_vol_bis.mean()

spy_put_in_aviv=data_6['SPY':'SPY'].loc[(data_6['Strike']/data_6['Underlying_Price']>1.05) & 
                   (data_6['Type']=='put')].Implied_vol_bis.mean()
amzn_call_at_avriv = amzn_at[amzn_at['Type']=='call'].Implied_vol_bis.mean()
amzn_put_at_avriv = amzn_at[amzn_at['Type']=='put'].Implied_vol_bis.mean()
spy_call_at_avriv = spy_at[spy_at['Type']=='call'].Implied_vol_bis.mean()
spy_put_at_avriv = spy_at[spy_at['Type']=='put'].Implied_vol_bis.mean()

Avr_Iv = pd.DataFrame([[amzn_call_in_avriv,amzn_put_in_avriv],
                       [amzn_call_at_avriv,amzn_put_at_avriv],
                       [amzn_call_out_avriv,amzn_put_out_avriv],
                       [spy_call_in_avriv,spy_put_in_aviv],
                       [spy_call_at_avriv,spy_put_at_avriv],
                       [spy_call_out_avriv,spy_put_out_avriv]],
                        index = [['AMZN','AMZN','AMZN','SPY','SPY','SPY'],
                                 ['in-the money','at-the-money','out-the money',
                                  'in-the money','at-the-money','out-the money']],
                        columns=['Call','Put'])
Avr_Iv.index.names = ['Options','Moneyness']
Avr_Iv.columns.names = ['Type']
del amzn_call_in_avriv,amzn_call_out_avriv,amzn_put_out_avriv,amzn_put_in_avriv,spy_call_in_avriv,spy_call_out_avriv,\
spy_put_out_avriv,spy_put_in_aviv,amzn_at,spy_at,amzn_call_at_avriv,amzn_put_at_avriv,spy_call_at_avriv,spy_put_at_avriv
#7
data_7 = Data1['AMZN':'SPY']
data_7 = data_7.loc[data_7['Vol'] != 0]
data_7 = data_7[(data_7['Ask'] != 0) & (data_7['Bid'] != 0)]
#data_7 = data_7[(data_7['Strike']/data_7['Underlying_Price']>0.95) & (data_7['Strike']/data_7['Underlying_Price']<1.05)]
data_7 = data_7.loc[data_7['TtM'] != 0]

# =============================================================================
# def Vega(r, vol, K, S, T):
#     d_1 = float(m.log(S/K)+(r+vol**2/2)*T)/float(vol*m.sqrt(T))
#     return S*m.sqrt(T)*norm.pdf(d_1)
# def Vega_sigma(r, vol, K, S, T):
#     d_1 = float(m.log(S/K)+(r+vol**2/2)*T)/float(vol*m.sqrt(T))
#     return -d_1*Vega(r, vol, K, S, T)*(m.sqrt(T)-d_1/vol)
# 
# def newton_method(first_dir, func, guess, tolerance):
#     b= guess
#     while(abs(func(b)>tolerance)):
#         a = b
#         b = a-func(a)/first_dir(a)
#     return b
# def get_iv2(type_opt, r, K, S, T, P,guess, tolerance):
#     original_func = lambda x: BS_Formula(type_opt, r, x, K, S, T)-P
#     fir_dfunc = lambda y: Vega(r, y, K, S, T)
#     return newton_method(original_func,fir_dfunc,guess,tolerance)
# 
# =============================================================================
def secant_method(func, guess1, guess2, tolerance):
    if abs(func(guess1))<tolerance:
        return guess1
    elif abs(func(guess2))<tolerance:
        return guess2
    else:
        new_guess = guess2
    while(abs(func(new_guess))>tolerance):
        k = float(func(guess2)-func(guess1))/float(guess2-guess1)
        new_guess = guess2-func(guess2)/k
        guess1 = guess2
        guess2 = new_guess
    return new_guess
def get_iv2(type_opt, r, K, S, T, P,guess1, guess2, tolerance):
    obj_func = lambda x: BS_Formula(type_opt, r, x, K, S, T) - P
    if obj_func(0.0001)>tolerance:
        return np.nan
    return secant_method(obj_func, guess1, guess2, tolerance)
start = time.time()
for ind2 in data_7.index:
    data_7.loc[ind2,'Avr_Price'] = get_avrprice(data_7.loc[ind2,'Bid'],data_7.loc[ind2,'Ask'])
    data_7.loc[ind2,'Implied_vol_secant'] = get_iv2(data_7.loc[ind2,'Type'], rate, data_7.loc[ind2,'Strike']
                        , data_7.loc[ind2,'Underlying_Price'], data_7.loc[ind2,'TtM'], data_7.loc[ind2,'Avr_Price'],
                        2,1,10**(-6))
    data_7.loc[ind2,'CalcuPrice'] = BS_Formula(data_7.loc[ind2,'Type'], rate, data_7.loc[ind2,'Implied_vol_secant'], data_7.loc[ind2,'Strike']
                        , data_7.loc[ind2,'Underlying_Price'], data_7.loc[ind2,'TtM'])
end = time.time()
timespent2 = end-start
data_7 = data_7.dropna()
data_7.head()

data_7 = Data1['AMZN':'SPY']
data_7 = data_7.loc[data_7['Vol'] != 0]
data_7 = data_7[(data_7['Ask'] != 0) & (data_7['Bid'] != 0)]
#data_7 = data_7[(data_7['Strike']/data_7['Underlying_Price']>0.95) & (data_7['Strike']/data_7['Underlying_Price']<1.05)]
data_7 = data_7.loc[data_7['TtM'] != 0]

def muller_method(func, guess0, guess1, guess2, tolerance):
    if abs(func(guess0))<tolerance:
        return guess0
    elif abs(func(guess1))<tolerance:
        return guess1
    elif abs(func(guess2))<tolerance:
        return guess2
    else:
        new_guess = guess2
    while(abs(func(new_guess))>tolerance):
        a = ((func(guess2)-func(guess1))/(guess2-guess1)-\
        (func(guess1)-func(guess0))/(guess1-guess0))/(guess2-guess0)
        b = (func(guess1)-func(guess0))/(guess1-guess0)+\
        (func(guess2)-func(guess0))/(guess2-guess0)-\
        (func(guess1)-func(guess2))/(guess1-guess2)
        c = func(guess0)
        delta = b**2-4*a*c
        if delta<0:
            return np.nan
        new_guess1 = guess0-2*c/(b-m.sqrt(b**2-4*a*c))
        new_guess2 = guess0-2*c/(b+m.sqrt(b**2-4*a*c))
        if abs(func(new_guess1))<abs(func(new_guess2)):
            new_guess = new_guess1
        else:
            new_guess = new_guess2
        guess0 = guess1
        guess1 = guess2
        guess2 = new_guess
    return new_guess
def get_iv3(type_opt, r, K, S, T, P, guess0, guess1, guess2, tolerance):
    obj_func = lambda x: BS_Formula(type_opt, r, x, K, S, T)-P
    if obj_func(0.0001)>tolerance:
        return np.nan
    return muller_method(obj_func, guess0, guess1, guess2, tolerance)

start = time.time()
for ind2 in data_7.index:
    data_7.loc[ind2,'Avr_Price'] = get_avrprice(data_7.loc[ind2,'Bid'],data_7.loc[ind2,'Ask'])
    data_7.loc[ind2,'Implied_vol_muller'] = get_iv3(data_7.loc[ind2,'Type'], rate, data_7.loc[ind2,'Strike']\
                        , data_7.loc[ind2,'Underlying_Price'], data_7.loc[ind2,'TtM'], data_7.loc[ind2,'Avr_Price'],
                        3,2,1,0.000001)
# =============================================================================
#     data_7.loc[ind2,'CalcuPrice2'] = BS_Formula(data_7.loc[ind2,'Type'], rate, data_7.loc[ind2,'Implied_vol_muller'], data_7.loc[ind2,'Strike']\
#                         , data_7.loc[ind2,'Underlying_Price'], data_7.loc[ind2,'TtM'])
# =============================================================================
end = time.time()
timespent3 = end-start
data_7 = data_7.dropna()
data_7.head()
Time_Consumed = pd.DataFrame([[timespent1,timespent2,timespent3]],index=['Time consumed'], columns = ['Bisection(s)','Secant(s)','Muller(s)'])

#8

# =============================================================================
# call = data_6.loc[data_6['Type'] == 'call'] 
# put = data_6.loc[data_6['Type'] == 'put'] 
# call_put = pd.merge(call,put,on=['Expiry','Strike'],how='inner')
# 
# =============================================================================
data_8 = data_6
amzn_call_8 = data_8['AMZN':'AMZN'].loc[data_8.Type=='call']
amzn_put_8 = data_8['AMZN':'AMZN'].loc[data_8.Type=='put']
amzn_result_table = pd.DataFrame(list(set(amzn_call_8.Expiry).intersection(set(amzn_put_8.Expiry))),columns = ['Expiry'])
for ind8 in amzn_result_table.index:
    amzn_result_table.loc[ind8,'Call_IV'] = amzn_call_8[amzn_call_8.Expiry == amzn_result_table.loc[ind8,'Expiry']].Implied_vol_bis.mean()
    amzn_result_table.loc[ind8,'Put_IV'] = amzn_put_8[amzn_put_8.Expiry == amzn_result_table.loc[ind8,'Expiry']].Implied_vol_bis.mean()
amzn_result_table = amzn_result_table.sort_values(by=['Expiry'])
amzn_result_table = amzn_result_table.set_index(keys=['Expiry'])

spy_call_8 = data_8['SPY':'SPY'].loc[data_8.Type=='call']
spy_put_8 = data_8['SPY':'SPY'].loc[data_8.Type=='put']
spy_result_table = pd.DataFrame(list(set(spy_call_8.Expiry).intersection(set(spy_put_8.Expiry))),columns = ['Expiry'])
for ind8 in spy_result_table.index:
    spy_result_table.loc[ind8,'Call_IV'] = spy_call_8[spy_call_8.Expiry == spy_result_table.loc[ind8,'Expiry']].Implied_vol_bis.mean()
    spy_result_table.loc[ind8,'Put_IV'] = spy_put_8[spy_put_8.Expiry == spy_result_table.loc[ind8,'Expiry']].Implied_vol_bis.mean()
spy_result_table = spy_result_table.sort_values(by=['Expiry'])
spy_result_table = spy_result_table.set_index(keys=['Expiry'])

data_8['SPY':'SPY'].loc[data_8.Expiry != dt.datetime(2019,2,13)].Implied_vol_bis.mean()
del amzn_call_8,amzn_put_8,spy_call_8,spy_put_8



#9
data_9 = data_6[(data_6['Strike']/data_6['Underlying_Price']>0.95) & (data_6['Strike']/data_6['Underlying_Price']<1.05)]
def put_call_parity(opt_type,  r, K, S,T, price):
    if opt_type == 'call':
        return price-S+K*m.exp(-r*T)
    else:
        return S-K*m.exp(-r*T)+price
call_9 = data_9.loc[data_9['Type'] == 'call'] 
put_9 = data_9.loc[data_9['Type'] == 'put'] 
call_put_9 = pd.merge(call_9,put_9,on=['Expiry','Strike'],how='outer',suffixes=('_call','_put'))
for ind9 in call_put_9.index:
    call_put_9.loc[ind9,'Calculated_call'] = put_call_parity(call_put_9.loc[ind9,'Type_put'],
              rate, call_put_9.loc[ind9,'Strike'],call_put_9.loc[ind9,'Underlying_Price_put'],
              call_put_9.loc[ind9,'TtM_put'],call_put_9.loc[ind9,'Avr_Price_put'])
    call_put_9.loc[ind9,'Calculated_put'] = put_call_parity(call_put_9.loc[ind9,'Type_call'],
              rate, call_put_9.loc[ind9,'Strike'],call_put_9.loc[ind9,'Underlying_Price_call'],
              call_put_9.loc[ind9,'TtM_call'],call_put_9.loc[ind9,'Avr_Price_call'])
call_put_9 = call_put_9[['Strike','Expiry','Underlying_Price_call','Avr_Price_call','Calculated_call','Type_call',\
                'Type_put','Calculated_put','Avr_Price_put']]
call_put_9 = call_put_9.rename(columns={'Underlying_Price_call' : 'Underlying_Price'})
call_put_9.head()
(abs(call_put_9.Calculated_call-call_put_9.Avr_Price_call)+\
abs(call_put_9.Calculated_put-call_put_9.Avr_Price_put)).mean()/2

#10

# =============================================================================
# amzn_10 = Data1['AMZN':'AMZN']
# amzn_10 = amzn_10.loc[amzn_10['Vol'] != 0]
# amzn_10 = amzn_10.loc[amzn_10['TtM'] != 0]
# amzn_10 = amzn_10.loc[amzn_10.Type == 'call']
# 
# for ind3 in amzn_10.index:
#     amzn_10.loc[ind3,'Avr_Price'] = get_avrprice(amzn_10.loc[ind3,'Bid'],amzn_10.loc[ind3,'Ask'])
#     amzn_10.loc[ind3,'Implied_vol_bis'] = get_iv(amzn_10.loc[ind3,'Type'], rate, amzn_10.loc[ind3,'Strike']
#                         , amzn_10.loc[ind3,'Underlying_Price'], amzn_10.loc[ind3,'TtM'], amzn_10.loc[ind3,'Avr_Price'],
#                         0.000001,7,0.001)
# amzn_10 = amzn_10.dropna()
# 
# =============================================================================

amzn_10 = data_6['AMZN':'AMZN']
amzn_10 = amzn_10.loc[amzn_10.Type == 'call']
amzn_10_215 = amzn_10.loc[amzn_10['Expiry'] == '2019-02-15']
amzn_10_315 = amzn_10.loc[amzn_10['Expiry'] == '2019-03-15']
amzn_10_418 = amzn_10.loc[amzn_10['Expiry'] == '2019-04-18']

plt.plot(amzn_10_215['Strike'],amzn_10_215['Implied_vol_bis'],label = '2-15')
plt.plot(amzn_10_315['Strike'],amzn_10_315['Implied_vol_bis'],label = '3-15')
plt.plot(amzn_10_418['Strike'],amzn_10_418['Implied_vol_bis'],label = '4-18')
plt.legend(loc = 0)
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.title('AMZN Call Volatility Smile')
del amzn_10_215,amzn_10_315,amzn_10_418


spy_10 = Data1['SPY':'SPY']
spy_10 = spy_10.loc[spy_10['Vol'] != 0]
spy_10 = spy_10.loc[spy_10['TtM'] != 0]
spy_10 = spy_10.loc[spy_10.Type == 'put']
for ind2 in spy_10.index:
    spy_10.loc[ind2,'Avr_Price'] = get_avrprice(spy_10.loc[ind2,'Bid'],spy_10.loc[ind2,'Ask'])
    spy_10.loc[ind2,'Implied_vol_bis'] = get_iv(spy_10.loc[ind2,'Type'], rate, spy_10.loc[ind2,'Strike']
                        , spy_10.loc[ind2,'Underlying_Price'], spy_10.loc[ind2,'TtM'], spy_10.loc[ind2,'Avr_Price'],
                        0.000001,7,0.001)
spy_10 = spy_10.dropna()


# =============================================================================
# spy_10 = data_6['SPY':'SPY']
# spy_10 = spy_10.loc[spy_10.Type == 'put']
# =============================================================================

spy_10_215 = spy_10.loc[spy_10['Expiry'] == '2019-02-15']
spy_10_315 = spy_10.loc[spy_10['Expiry'] == '2019-03-15']
spy_10_418 = spy_10.loc[spy_10['Expiry'] == '2019-04-18']

plt.plot(spy_10_215['Strike'],spy_10_215['Implied_vol_bis'],label = '2-15')
plt.plot(spy_10_315['Strike'],spy_10_315['Implied_vol_bis'],label = '3-15')
plt.plot(spy_10_418['Strike'],spy_10_418['Implied_vol_bis'],label = '4-18')
plt.legend(loc = 0)
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.title('AMZN Put Volatility Smile')
del spy_10_215,spy_10_315,spy_10_418

#plt.plot(spy_10_418['Strike'],spy_10_418['Implied_vol_bis'])

from mpl_toolkits.mplot3d import Axes3D
amzn_10_3d = data_6['AMZN':'AMZN']
amzn_10_3d = amzn_10_3d.loc[amzn_10_3d['Vol'] != 0]
amzn_10_3d = amzn_10_3d.loc[amzn_10_3d['TtM'] != 0]
amzn_10_3d = amzn_10_3d.loc[amzn_10_3d.Type == 'call']
amzn_10_3d = amzn_10_3d.sort_values(by=['TtM'])

x = np.array(amzn_10_3d.Strike[1:])
y = np.array(amzn_10_3d.TtM[1:]*365)
z = np.array(amzn_10_3d.Implied_vol_bis[1:])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x,y,z,cmap='plasma')
ax.set_xlabel('Strike Price($)')
ax.set_ylabel('Time to Matuity(Days)')
ax.set_zlabel('Implied Volatility')


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

X = np.linspace(-6, 6, 30)
Y= np.linspace(-6, 6, 30)

X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

#11
def Delta_call(r, vol, K, S, T):
    d_1 = float(m.log(S/K)+(r+vol**2/2)*T)/float(vol*m.sqrt(T))
    return norm.cdf(d_1)
def Gamma(r, vol, K, S, T):
    d_1 = float(m.log(S/K)+(r+vol**2/2)*T)/float(vol*m.sqrt(T))
    return norm.pdf(d_1)/(S*vol*m.sqrt(T))
def Vega(r, vol, K, S, T):
    d_1 = float(m.log(S/K)+(r+vol**2/2)*T)/float(vol*m.sqrt(T))
    return S*m.sqrt(T)*norm.pdf(d_1)

def First_dir(func, small_gap, x):
    return (func(x+small_gap)-func(x))/(small_gap)
def Second_dir(func,small_gap,x):
    return (func(x+small_gap)-2*func(x)+func(x-small_gap))/(small_gap**2)
def get_num_delta_call( r, vol, K, S_with_respct, T,small_gap):
    BS_s = lambda s: BS_Formula('call',r,vol,K,s,T)
    return First_dir(BS_s, small_gap, S_with_respct)
def get_num_gamma( r, vol, K, S_with_respct, T,small_gap):
    BS_s = lambda s: BS_Formula('call',r,vol,K,s,T)
    return Second_dir(BS_s, small_gap, S_with_respct)
def get_num_vega( r, vol_with_respct, K, S, T,small_gap):
    BS_sigma = lambda sigma: BS_Formula('call',r,sigma,K,S,T)
    return First_dir(BS_sigma, small_gap, vol_with_respct)

data_11 = data_6.loc[data_6.Type == 'call']
for ind11 in data_11.index:
    data_11.loc[ind11,'Delta'] = Delta_call(rate,
               data_11.loc[ind11,'Implied_vol_bis'],
               data_11.loc[ind11,'Strike'],
               data_11.loc[ind11,'Underlying_Price'],
               data_11.loc[ind11,'TtM'])
    data_11.loc[ind11,'Gamma'] = Gamma(rate,
               data_11.loc[ind11,'Implied_vol_bis'],
               data_11.loc[ind11,'Strike'],
               data_11.loc[ind11,'Underlying_Price'],
               data_11.loc[ind11,'TtM'])
    data_11.loc[ind11,'Vega'] = Vega(rate,
               data_11.loc[ind11,'Implied_vol_bis'],
               data_11.loc[ind11,'Strike'],
               data_11.loc[ind11,'Underlying_Price'],
               data_11.loc[ind11,'TtM'])
    data_11.loc[ind11,'Numerical_Delta'] =get_num_delta_call(rate,
               data_11.loc[ind11,'Implied_vol_bis'],
               data_11.loc[ind11,'Strike'],
               data_11.loc[ind11,'Underlying_Price'],
               data_11.loc[ind11,'TtM'],10**(-4))
    data_11.loc[ind11,'Numerical_Gamma'] =get_num_gamma(rate,
               data_11.loc[ind11,'Implied_vol_bis'],
               data_11.loc[ind11,'Strike'],
               data_11.loc[ind11,'Underlying_Price'],
               data_11.loc[ind11,'TtM'],10**(-4))
    data_11.loc[ind11,'Numerical_Vega'] =get_num_vega(rate,
               data_11.loc[ind11,'Implied_vol_bis'],
               data_11.loc[ind11,'Strike'],
               data_11.loc[ind11,'Underlying_Price'],
               data_11.loc[ind11,'TtM'],10**(-4))

data_11 = data_11[['Strike','Expiry','Type','Delta','Gamma','Vega','Numerical_Delta','Numerical_Gamma','Numerical_Vega']]
data_11.head()


#12
data_12 = Data2['AMZN':'SPY']
data_12 = data_12.loc[data_12['Vol'] != 0]
data_12 = data_12[(data_12['Ask'] != 0) & (data_12['Bid'] != 0)]
data_12 = data_12.loc[data_12['TtM'] != 0]
#data_12['data12_index'] = data_12.index
data_12 = pd.merge(data_6,data_12,
                   on=['Expiry','Strike','Type'],
                   how='inner').loc[:,['Strike','Expiry','Type',
                   'Last_y','Bid_y','Ask_y','Vol_y','Underlying_Price_y','TtM_y','Implied_vol_bis']]
for ind12 in data_12.index:
    data_12.loc[ind12,'BS_Price'] = BS_Formula(data_12.loc[ind12,'Type'],
               rate, data_12.loc[ind12,'Implied_vol_bis'], 
               data_12.loc[ind12,'Strike'], data_12.loc[ind12,'Underlying_Price_y'],
               data_12.loc[ind12,'TtM_y'])
    data_12.loc[ind12,'Avr_Price'] = get_avrprice(data_12.loc[ind12,'Bid_y'],data_12.loc[ind12,'Ask_y'])
#data_12.to_csv('data12.csv')
data_12 = data_12[['Strike','Expiry','Type','Avr_Price','BS_Price']]
data_12.head()
# =============================================================================
# data_12 = data_12.loc[pd.merge(data_7,data_12,on=['Expiry','Strike','Type'],how='inner').data12_index,:]
# data_12['Implied_vol'] = data_7.loc[data_12.index,'Implied_vol_secant']
# 
# =============================================================================
# part3
#1


def func1(x):
    if x == 0:
        return 1
    else:
        return m.sin(x)/x

def trapezoid_int(func,a,b,n):
    if n<1:
        print('wrong number')
    x = np.linspace(a,b,n+1)
    f_x = np.vectorize(func)(x)
    delta = (b-a)/n
    return delta/2*(f_x.sum()+f_x[1:-1].sum())

trapezoid_int(func1,-(10**6),(10**6),5000000)

def simpson_int(func,a,b,n):
    if n<1:
        print('wrong number')
    x = np.linspace(a,b,n+1)
    f_x = np.vectorize(func)(x)
    delta = (b-a)/n
    return delta/3*(f_x[0]+f_x[-1]+4*f_x[1:-1][::2].sum()+2*f_x[1:-1][1::2].sum())
simpson_int(func1,-(10**6),(10**6),50000000)

#2
def truncation_tra(a,N):
    return abs(trapezoid_int(func1,-a,a,N)-m.pi)
def truncation_sim(a,N):
    return abs(simpson_int(func1,-a,a,N)-m.pi)
def difference(a,N):
    return (trapezoid_int(func1,-a,a,N)-simpson_int(func1,-a,a,N))
N=10**5
a_list = np.arange(100,10**4,50)
trunc1 = [truncation_tra(x,N) for x in a_list]
trunc2 = [truncation_sim(x,N) for x in a_list]
dif = [difference(x,N) for x in a_list]

plt.plot(a_list,trunc1)
plt.xlabel('Interval')
plt.ylabel('Absolute Value of Truncation')
plt.title('Truncation of Trapezoid Method')
plt.plot(a_list,trunc2,)
plt.xlabel('Interval')
plt.ylabel('Absolute Value of Truncation')
plt.title('Truncation of Simpson Method')
plt.plot(a_list,dif)
plt.xlabel('Interval')
plt.ylabel('Difference')
plt.title('Difference between Two Method')


a = 10**5
N_list=np.arange(100,10**4,50)
trunc3 = [truncation_tra(a,n) for n in N_list]
trunc4 = [truncation_sim(a,n) for n in N_list]
dif2 = [difference(a,n) for n in N_list]

plt.plot(N_list,trunc3)
plt.xlabel('Sections')
plt.ylabel('Absolute Value of Truncation')
plt.title('Truncation of Trapezoid Method')
plt.plot(N_list,trunc4)
plt.xlabel('Sections')
plt.ylabel('Absolute Value of Truncation')
plt.title('Truncation of Simpson Method')
plt.plot(N_list,dif2)
plt.xlabel('Sections')
plt.ylabel('Difference')
plt.title('Difference between Two Method')
#3
def Intergral(func, method, a, b, tolerance):  
    n=1
    I_last = 0
    I_now = method(func,a,b,n)
    while (abs(I_now-I_last)>tolerance):   
        n=n+1
        I_last = I_now
        I_now = method(func,a,b,n)
    return I_now, n
result2,k2 = Intergral(func1,simpson_int,-(10000),(10000),10**-4)
result1,k1 = Intergral(func1,trapezoid_int,-(10000),(10000),10**-4)
#4
def g_x(x):
    return 1+m.exp(-x**2)*m.cos(8*x**(2/3))

Intergral(g_x,simpson_int,0,2,10**-4)

Intergral(lambda x: x**2,trapezoid_int,0,2,10**-4)
















