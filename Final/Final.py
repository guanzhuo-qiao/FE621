# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:00:47 2019

@author: Qiao Guanzhuo
"""
import numpy as np
from scipy.stats import norm
import time
from sklearn import linear_model
import os

#Aa
def analytical_asian(S0_,r_,sigma_,K_,T_):
    N = T_*252
    sigma_hat = sigma_*np.sqrt((2*N+1)/(6*(N+1)))
    rho = 0.5*(r_-0.5*sigma_**2+sigma_hat**2)
    d1 = (np.log(S0_/K_)+(rho+0.5*sigma_hat**2)*T_)/(np.sqrt(T_)*sigma_hat)
    d2 = (np.log(S0_/K_)+(rho-0.5*sigma_hat**2)*T_)/(np.sqrt(T_)*sigma_hat)
    Pg = np.exp(-r_*T_)*(S0_*np.exp(rho*T_)*norm.cdf(d1)-K_*norm.cdf(d2))
    return Pg

pg = analytical_asian(S0_=100,r_=0.03,sigma_=0.3,K_=100,T_=5)

#Ab
def Monte_Carlo_arith_asian(m_,T,S0,sig0,r,K,conf_level):
    start = time.time()
    dt = 1/252
    nudt = (r-sig0**2/2)*dt
    sigdt = sig0*np.sqrt(dt)
    dis = np.exp(-r*T)
    sum_C = 0
    sum_C2 = 0
    for i in range(int(m_)):
        lgXt = np.log(S0)
        sum_Xt = S0
        for j in range(int(T*252)):
            z = np.random.randn()
            lgXt += nudt+sigdt*z
            sum_Xt += np.exp(lgXt)
        C = dis*max(sum_Xt/(T*252+1)-K,0)
        sum_C += C
        sum_C2 += C**2
    mean_C = sum_C/m_
    se = np.sqrt((sum_C2-m_*mean_C**2)/(m_-1)/m_)
    end = time.time()
    return mean_C,'[{},{}]'.format(mean_C-se*norm.ppf(conf_level),mean_C+se*norm.ppf(conf_level)),(end-start)
pa_sim = Monte_Carlo_arith_asian(m_=1e6,T=5,S0=100,sig0=0.3,r=0.03,K=100,conf_level = 0.95)

#c
def Monte_Carlo_geo_asian(m_,T,S0,sig0,r,K,conf_level):
    start = time.time()
    dt = 1/252
    nudt = (r-sig0**2/2)*dt
    sigdt = sig0*np.sqrt(dt)
    dis = np.exp(-r*T)
    sum_C = 0
    sum_C2 = 0
    for i in range(int(m_)):
        lgXt = np.log(S0)
        sum_Xt = lgXt
        for j in range(int(T*252)):
            z = np.random.randn()
            lgXt += nudt+sigdt*z
            sum_Xt += lgXt
        C = dis*max(np.exp(sum_Xt/(T*252+1))-K,0)
        sum_C += C
        sum_C2 += C**2
    mean_C = sum_C/m_
    se = np.sqrt((sum_C2-m_*mean_C**2)/(m_-1)/m_)
    end = time.time()
    return mean_C,'[{},{}]'.format(mean_C-se*norm.ppf(conf_level),mean_C+se*norm.ppf(conf_level)),(end-start)
pg_sim = Monte_Carlo_arith_asian(m_=1e3,T=5,S0=100,sig0=0.3,r=0.03,K=100,conf_level = 0.95)
#d
def Monte_Carlo_get_b(m_,T,S0,sig0,r,K):
    dt = 1/252
    nudt = (r-sig0**2/2)*dt
    sigdt = sig0*np.sqrt(dt)
    dis = np.exp(-r*T)
    Xi = []
    Yi = []
    for i in range(int(m_)):
        lgSt = np.log(S0)
        sum_St1 = 0
        sum_St2 = 0
        for j in range(int(T*252)):
            z = np.random.randn()
            lgSt += nudt+sigdt*z
            sum_St1 += np.exp(lgSt)
            sum_St2 += lgSt
        Xi.append(dis*max(sum_St1/(T*252+1)-K,0))
        Yi.append(dis*max(np.exp(sum_St2/(T*252+1))-K,0))
    lr = linear_model.LinearRegression().fit(np.asarray(Xi).reshape(-1,1),np.asarray(Yi).reshape(-1,1))
    b_star = lr.coef_
    return b_star[0,0]

b_star = Monte_Carlo_get_b(m_=1e4,T=5,S0=100,sig0=0.3,r=0.03,K=100)

#e
Eg = pg-pg_sim

#f
pa_star = pa_sim-b_star*Eg

#B
#1
from pandas_datareader import data as pdrd
import datetime as dt
import pandas as pd

os.chdir(r'D:\Grad 2\621\assignment\Final')
stocklist = ['BAC','C','JPM','AFL','XLF']
def download(stocklist):
    starttime = dt.datetime(2012,1,1)
    endtime = dt.date.today()
    for stock in stocklist:
        equity_data = pdrd.DataReader(stock, data_source='yahoo',start=starttime,end=endtime)
        equity_data.to_csv('{} equity.csv'.format(stock))
download(stocklist)

bac_p = pd.read_csv('BAC equity.csv',index_col=0,usecols = [0,6])
c_p = pd.read_csv('C equity.csv',index_col=0,usecols = [0,6])
jpm_p = pd.read_csv('JPM equity.csv',index_col=0,usecols = [0,6])
afl_p = pd.read_csv('AFL equity.csv',index_col=0,usecols = [0,6])
xlf_p = pd.read_csv('XLF equity.csv',index_col=0,usecols = [0,6])

bac = np.log(bac_p/bac_p.shift(1))[1:]
c = np.log(c_p/c_p.shift(1))[1:]
jpm = np.log(jpm_p/jpm_p.shift(1))[1:]
afl = np.log(afl_p/afl_p.shift(1))[1:]
xlf = np.log(xlf_p/xlf_p.shift(1))[1:]

d_t = 1/255
bac_theta2 = np.std(bac,ddof = 1)/np.sqrt(d_t)
bac_theta1 = np.mean(bac)/d_t+0.5*bac_theta2**2

c_theta2 = np.std(c,ddof = 1)/np.sqrt(d_t)
c_theta1 = np.mean(c)/d_t+0.5*c_theta2**2

jpm_theta2 = np.std(jpm,ddof = 1)/np.sqrt(d_t)
jpm_theta1 = np.mean(jpm)/d_t+0.5*jpm_theta2**2

afl_theta2 = np.std(afl,ddof = 1)/np.sqrt(d_t)
afl_theta1 = np.mean(afl)/d_t+0.5*afl_theta2**2

#3

stock_pool = pd.DataFrame([],index =  bac.index)
stock_pool['bac'] = bac
stock_pool['c'] = c
stock_pool['jpm'] = jpm
stock_pool['afl'] = afl
corr_matrix = stock_pool.corr()


#4
def Cholesky(A):
    n = A.shape[0]
    L = np.zeros(A.shape)
    L[0,0] = np.sqrt(A[0,0])
    for i in range(1,n):
        L[i,0] = A[i,0]/L[0,0]
    for i in range(1,n):
        for j in range(1,i+1):
            if i == j:
                L[j,j] = np.sqrt(A[j,j]-np.sum([(L[j,k])**2 for k in range(j)]))
            else:
                L[i,j] = (A[i,j]-np.sum([L[i,k]*L[j,k] for k in range(j)]))/L[j,j]
    return L



def Monte_Carlo3(m_,T,S0,mu0,sig0,A):
    dt = 1/255
    result = []
    L = Cholesky(A)
    for i in range(int(m_)):
        Xt = S0[0]
        Yt = S0[1]
        Zt = S0[2]
        Qt = S0[3]
        for j in range(int(T/dt)):
            row_z = np.random.randn(4)
            z = np.dot(L,row_z)
            z1 = z[0]
            z2 = z[1]
            z3 = z[2]
            z4 = z[3]
            Xt += mu0[0]*Xt*dt+sig0[0]*Xt*z1*np.sqrt(dt)+0.5*sig0[0]**2*(z1**2-1)*dt
            Yt += mu0[1]*Yt*dt+sig0[1]*Yt*z2*np.sqrt(dt)+0.5*sig0[1]**2*(z2**2-1)*dt
            Zt += mu0[2]*Zt*dt+sig0[2]*Zt*z3*np.sqrt(dt)+0.5*sig0[2]**2*(z3**2-1)*dt
            Qt += mu0[2]*Qt*dt+sig0[3]*Zt*z4*np.sqrt(dt)+0.5*sig0[3]**2*(z4**2-1)*dt
        result.append([Xt,Yt,Zt,Qt])
    return result

A_ = np.matrix(corr_matrix)
res = Monte_Carlo3(m_=1000,T=1,
                   S0=[bac_p.iloc[-1,0],c_p.iloc[-1,0],jpm_p.iloc[-1,0],afl_p.iloc[-1,0]],
                   mu0=[bac_theta1[0],c_theta1[0],jpm_theta1[0],afl_theta1[0]],
                   sig0=[bac_theta2[0],c_theta2[0],jpm_theta2[0],afl_theta2[0]],
                   A=A_)
res = pd.DataFrame(res)
np.mean(res)
np.std(res)
res.kurtosis()
res.skew()
#5

xlf_theta2 = np.std(xlf,ddof = 1)/np.sqrt(d_t)
xlf_theta1 = np.mean(xlf)/d_t+0.5*xlf_theta2**2


#6
respond_v = np.asmatrix(xlf)
predictor_v = np.asmatrix(stock_pool)
fit_model = linear_model.LinearRegression().fit(predictor_v, respond_v)

weights = fit_model.coef_

#7

def Monte_Carlo4(m_,T,S0,mu0,sig0,A,weights_,r,opt_type):
    dt = 1/255
    dis = np.exp(-r*T)
    L = Cholesky(A)
    sum_C = 0
    for i in range(int(m_)):
        Xt = S0[0]
        Yt = S0[1]
        Zt = S0[2]
        Qt = S0[3]
        etf = S0[4]
        for j in range(int(T/dt)):
            row_z = np.random.randn(4)
            z = np.dot(L,row_z)
            z1 = z[0]
            z2 = z[1]
            z3 = z[2]
            z4 = z[3]
            z5 = np.random.randn()
            Xt += mu0[0]*Xt*dt+sig0[0]*Xt*z1*np.sqrt(dt)+0.5*sig0[0]**2*(z1**2-1)*dt
            Yt += mu0[1]*Yt*dt+sig0[1]*Yt*z2*np.sqrt(dt)+0.5*sig0[1]**2*(z2**2-1)*dt
            Zt += mu0[2]*Zt*dt+sig0[2]*Zt*z3*np.sqrt(dt)+0.5*sig0[2]**2*(z3**2-1)*dt
            Qt += mu0[3]*Qt*dt+sig0[3]*Qt*z4*np.sqrt(dt)+0.5*sig0[3]**2*(z4**2-1)*dt
            etf += mu0[4]*etf*dt+sig0[4]*etf*z5*np.sqrt(dt)+0.5*sig0[4]**2*(z5**2-1)*dt
        Ut = np.dot(weights_,np.array([Xt,Yt,Zt,Qt]))[0]
        if opt_type == 'call':
            C = max(Ut-etf,0)*dis
        else:
            C = max(etf-Ut,0)*dis
        sum_C += C
    mean_C = sum_C/m_
    return mean_C

Monte_Carlo4(m_=1000,T=1,
             S0=[bac_p.iloc[-1,0],c_p.iloc[-1,0],jpm_p.iloc[-1,0],afl_p.iloc[-1,0],xlf_p.iloc[-1,0]],
             mu0=[bac_theta1[0],c_theta1[0],jpm_theta1[0],afl_theta1[0],xlf_theta1[0]],
             sig0=[bac_theta2[0],c_theta2[0],jpm_theta2[0],afl_theta2[0],xlf_theta2[0]],
             A=A_,
             weights_=weights,
             r = 0.06,
             opt_type = 'call')

#C
spx = pd.read_excel('SPX.xls', header = None)
td_data = spx.iloc[0,:-1]
spx = spx.iloc[1:,:]
spx.columns = spx.iloc[0,:]
spx = spx.reindex(spx.index[1:])


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
def BS_Formula(type_opt, r, vol, K, S, T):
    d_1 = float(np.log(S/K)+(r+vol**2/2)*T)/float(vol*np.sqrt(T))
    d_2 = d_1-vol*np.sqrt(T)
    if type_opt == 'call':
        return norm.cdf(d_1)*S-K*np.exp(-r*T)*norm.cdf(d_2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d_2)-norm.cdf(-d_1)*S

def get_iv(type_opt, r, K, S, T, P,tolerance,up,down):
    obj_func= lambda x: BS_Formula(type_opt, r, x, K, S, T)-P
    return Bisection(obj_func,tolerance,up,down)

for ind in spx.index:
    spx.loc[ind,'Implied_vol_bis'] = get_iv('call', td_data[2]/100, spx.loc[ind,'K']\
                        , td_data[1], spx.loc[ind,'T'], spx.loc[ind,'Price'],
                        10**(-6), 1, 0.00001)
spx_result = spx.dropna()
dd = spx_result[(spx_result['Date']==40074) | (spx_result['Date']== 39920) | (spx_result['Date']== 39948) | (spx_result['Date']== 39983)]

dd_ = dd.drop_duplicates(subset = 'Date',keep = 'first')
dd_date = dd_['Date']
K_list = dd['K']
for i in dd_date:
    B = dd.loc[dd['Date'] == i]
    B = B.loc[:,'K']
    B = B.drop_duplicates()
    K_list = np.intersect1d(K_list,B)
K_list = list(K_list[10:33])
data_c = dd.loc[dd['K'].isin(K_list)]
data_c = data_c.sort_values(by=['Date','K'])
data_c= data_c.drop_duplicates(subset = ['Date','K'],keep = 'first')
data_c.index = range(len(data_c))



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
x_ = np.array(list(data_c.loc[:,'K']))
y = np.asarray(list(data_c.loc[:,'T']))
z = np.array(list(data_c.loc[:,'Implied_vol_bis']))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x_,y,z,c='c')









from scipy.interpolate import CubicSpline
x = np.asarray(K_list[::])
z1 = np.array(z[:20])
z2 = np.array(z[20:40])
z3 = np.array(z[40:60])
z4 = np.array(z[60:80])
cs1 = CubicSpline(x, z1,axis = 1)
cs2 = CubicSpline(x, z2,axis = 1)
cs3 = CubicSpline(x, z3,axis = 1)
cs4 = CubicSpline(x, z4,axis = 1)
cs = [cs1,cs2,cs3,cs4]
xs = np.linspace(K_list[0],K_list[-1],500)
xs2 = list(xs)*4
ys2 = []
zs2 = []
for i in range(len(dd_date)):
    ys2+=[dd_date.iloc[i]]*500
    zs2+=list(cs[i](xs))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(xs2,ys2,zs2,cmap='plasma')


z_1 = zs2[:500]
z_2 = zs2[500:1000]
z_3 = zs2[1000:1500]
z_4 = zs2[1500:2000]

cs_m = []*500
for i in range(len(xs)):
    temp = CubicSpline(list(dd_date),[z_1[i],z_2[i],z_3[i],z_4[i]],axis = 1)
    cs_m.append(temp)
ds = np.linspace(dd_date.min(),dd_date.max(),20)
ys1 = list(ds)*500
xs1 = []
zs1 = []
for i in range(len(xs)):
    xs1+=[xs[i]]*20
    zs1 += list(cs_m[i](ds))
   
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(xs1,ys1,zs1,cmap='plasma')



def local_v(S_,K_,tau_,sigma_,r_,s2t,s2k,s2k2):
    d1 = (np.log(S_/K_)+(r_+0.5*sigma_**2)*tau_)/(sigma_*np.sqrt(tau_))
    up = (2*sigma_*s2t*tau_)+sigma_**2+2*sigma_*r_*tau_*K_*s2k
    down = ((1+K_*d1*s2k*np.sqrt(tau_))**2+K_**2*tau_*sigma_*(s2k2-d1*s2k**2*np.sqrt(tau_)))
    result = np.sqrt(up/down)
    return result


spx_result = spx_result.sort_values(by=['Date','K'])
spx_result.index = range(len(spx_result))

datt = spx.drop_duplicates(subset = 'Date',keep = 'first')['Date']
new_dt = []
for i in datt:
    frame = spx_result.loc[spx_result['Date'] == i]
    frame = frame.drop_duplicates(subset = 'K',keep = 'first')
    s2k_list = (np.asarray(frame['Implied_vol_bis'])[1:]-np.asarray(frame['Implied_vol_bis'])[:-1])/\
    (np.asarray(frame['K'])[1:]-np.asarray(frame['K'])[:-1])
    s2k2_list = (s2k_list[1:]-s2k_list[:-1])/(np.asarray(frame['K'])[1:-1]-np.asarray(frame['K'])[:-2])
    new_frame = frame.iloc[:-2,:]
    new_frame['s2k'] = s2k_list[:-1]
    new_frame['s2k2'] = s2k2_list[::]
    new_dt.append(new_frame)
new_dt = pd.concat(new_dt)
new_dt = new_dt.sort_values(by=['K','Date'])
new_dt.index = range(len(new_dt))
katt = new_dt.drop_duplicates(subset = 'K',keep = 'first')['K']
new_dt2 = []
for i in katt:
    frame = new_dt.loc[new_dt['K'] == i]
    frame = frame.drop_duplicates(subset = 'Date',keep = 'first')
    if len(frame) == 1:
        continue
    else:
        s2t_list = (np.asarray(frame['Implied_vol_bis'])[1:]-np.asarray(frame['Implied_vol_bis'])[:-1])/\
        (np.asarray(frame['T'])[1:]-np.asarray(frame['T'])[:-1])
        new_frame = frame.iloc[:-1,:]
        new_frame['s2t'] = s2t_list
        new_dt2.append(new_frame)
new_dt2 = pd.concat(new_dt2)

new_dt2 = new_dt2.sort_values(by=['Date','K'])
new_dt2.index = range(len(new_dt2))
for i in range(len(new_dt2)):
    new_dt2.loc[i,'lv'] = local_v(S_=td_data[1],K_ = new_dt2.loc[i,'K'],tau_ = new_dt2.loc[i,'T'],
               sigma_ = new_dt2.loc[i,'Implied_vol_bis'],r_ = td_data[2]/100,s2t = new_dt2.loc[i,'s2t'],
               s2k = new_dt2.loc[i,'s2k'],s2k2 = new_dt2.loc[i,'s2k2'])
new_dt2['lv'] = new_dt2['lv'].interpolate()

data_c2 = new_dt2[(new_dt2['Date'].isin(dd_date))&(new_dt2['K'].isin(K_list))]
data_c2.index = range(len(data_c2))

# =============================================================================
# from scipy.interpolate import CubicSpline
# K_list2 = list(data_c2.drop_duplicates(subset = 'K',keep = 'first')['K'])
# x = np.asarray(K_list2[::])
# zd = data_c2['lv']
# z1d = np.array(zd[:18])
# z2d = np.array(zd[18:36])
# z3d = np.array(zd[36:56])
# z4d = np.array(zd[56:])
# cs1d = CubicSpline(x[:18], z1d,axis = 1)
# cs2d = CubicSpline(x[:18], z2d,axis = 1)
# cs3d = CubicSpline(x, z3d,axis = 1)
# cs4d = CubicSpline(x, z4d,axis = 1)
# csd = [cs1d,cs2d,cs3d,cs4d]
# xsd = np.linspace(K_list2[0],K_list2[-1],500)
# xs2d = list(xsd)*4
# ys2d = []
# zs2d = []
# for i in range(len(dd_date)):
#     ys2d+=[dd_date.iloc[i]]*500
#     zs2d+=list(csd[i](xsd))
# 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(xs2d,ys2d,zs2d,cmap='plasma')
# 
# =============================================================================
xs2d = list(data_c2['K'])
ys2d = list(data_c2['T'])
zs2d = list(data_c2['lv'])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(np.asarray(xs2d),np.asarray(ys2d),np.asarray(zs2d),cmap='plasma')





for i in range(len(new_dt2)):
    new_dt2.loc[i,'p_bs'] = BS_Formula(type_opt='call', r = td_data[2]/100, vol = new_dt2.loc[i,'Implied_vol_bis'],
               K = new_dt2.loc[i,'K'], S = td_data[1], T = new_dt2.loc[i,'T'])
    new_dt2.loc[i,'p_lv'] = BS_Formula(type_opt='call', r = td_data[2]/100, vol = new_dt2.loc[i,'lv'],
               K = new_dt2.loc[i,'K'], S = td_data[1], T = new_dt2.loc[i,'T'])

data_c2.to_csv('SPXvolatility.csv')







