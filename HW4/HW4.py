# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:08:38 2019

@author: Qiao Guanzhuo
"""

import numpy as np
from scipy.stats import norm
import time
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


class derivatives(object):
    def __init__(self,Strike,Maturity,Spot_p,r,q,vol,opt_type):
        self.K = Strike
        self.T = Maturity
        self.S0 = Spot_p
        self.r = r
        self.q = q
        self.vol = vol
        self.opt_type = opt_type


def Monte_Carlo1(n_,m_,der):
    start = time.time()
    dt = der.T/n_
    nudt = (der.r-der.q-(der.vol)**2*0.5)*dt
    sigdt = der.vol*np.sqrt(dt)
    dis = np.exp(-der.r*der.T)
    sum_C = 0
    sum_C2 = 0
    for i in range(int(m_)):
        lnSt = np.log(der.S0)
        for j in range(int(n_)):
            z = np.random.randn()
            lnSt += nudt+sigdt*z
        if der.opt_type == "call":
            C = dis*max(np.exp(lnSt)-der.K,0)
        else:
            C = dis*max(-np.exp(lnSt)+der.K,0)
        sum_C += C
        sum_C2 += C**2
    mean_C = sum_C/m_
    se = np.sqrt((sum_C2-m_*mean_C**2)/(m_-1)/m_)
    end = time.time()
    return mean_C, se, (end-start)
  
der1 = derivatives(Strike=100,Maturity = 1,Spot_p = 100,r = 0.06,
                   q = 0.03,vol = 0.2,opt_type = "call")
der2 = derivatives(Strike=100,Maturity = 1,Spot_p = 100,r = 0.06,
                   q = 0.03,vol = 0.2,opt_type = "put")
Monte_Carlo1(n_ = 300,m_ = 1e6,der = der1)

def BS_Formula(type_opt, r, vol, K, S, T, q=0):
    d_1 = float(np.log(S/K)+(r-q+vol**2/2)*T)/float(vol*np.sqrt(T))
    d_2 = d_1-vol*np.sqrt(T)
    if type_opt == 'call':
        return norm.cdf(d_1)*S*np.exp(-q*T)-K*np.exp(-r*T)*norm.cdf(d_2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d_2)-norm.cdf(-d_1)*S*np.exp(-q*T)
BS_Formula(type_opt="call", r = 0.06, vol = 0.2, K=100, S=100, T=1, q=0.03)

result_table1 = pd.DataFrame(np.zeros((2,2)))
result_table2 = pd.DataFrame(np.zeros((2,2)))
result_table3 = pd.DataFrame(np.zeros((2,2)))
for i in range(1,3):
    for j in range(1,3):
        n = i*100+200
        m = j*1e6
        res = Monte_Carlo1(n_ = n,m_ = m,der = der1)
        result_table1.iloc[i-1,j-1] = res[0]
        result_table2.iloc[i-1,j-1] = res[1]
        result_table3.iloc[i-1,j-1] = res[2]

rsult_table4 = pd.DataFrame([Monte_Carlo1(n_ = 300,m_ = 1e6,der = der2)[0],
                             Monte_Carlo1(n_ = 400,m_ = 1e6,der = der2)[0],
                             Monte_Carlo1(n_ = 400,m_ = 2e6,der = der2)[0]])

#2
res_call1 = Monte_Carlo1(n_ = 300,m_ = 10000,der = der1)
res_put1 = Monte_Carlo1(n_ = 300,m_ = 10000,der = der2)

def Monte_Carlo_Anti(n_,m_,der):
    dt = der.T/n_
    nudt = (der.r-der.q-(der.vol)**2*0.5)*dt
    sigdt = der.vol*np.sqrt(dt)
    dis = np.exp(-der.r*der.T)
    sum_C = 0
    sum_C2 = 0
    for i in range(int(m_)):
        lnSt1 = np.log(der.S0)
        lnSt2 = np.log(der.S0)
        for j in range(int(n_)):
            z = np.random.randn()
            lnSt1 += nudt+sigdt*z
            lnSt2 += nudt-sigdt*z
        if der.opt_type == "call":
            C = dis*(max(np.exp(lnSt1)-der.K,0)+max(np.exp(lnSt2)-der.K,0))
        else:
            C = dis*(max(-np.exp(lnSt1)+der.K,0)+max(-np.exp(lnSt2)+der.K,0))
        sum_C += C
        sum_C2 += C**2
    mean_C = sum_C/m_
    se = np.sqrt((sum_C2-m_*mean_C**2)/(m_-1)/m_)
    return mean_C, se




def Black_Scholes_delta(St,t,K,T,sig,r,div):
    d1 = (np.log(St/K)+(r+sig**2/2)*(T-t))/(sig*np.sqrt(T-t))
    return np.exp(-div*(T-t))*norm.cdf(d1)




def Monte_Carlo_Del(n_,m_,der):
    dt = der.T/n_
    nudt = (der.r-der.q-(der.vol)**2*0.5)*dt
    sigdt = der.vol*np.sqrt(dt)
    dis = np.exp(-der.r*der.T)
    erddt = np.exp((der.r-der.q)*dt)
    beta1 = -1
    
    sum_C = 0
    sum_C2 = 0
    for i in range(int(m_)):
        St = der.S0
        cv = 0
        cv2 = 0
        for j in range(int(n_)):
            t = (j)*dt
            delta = Black_Scholes_delta(St = St,t=t,K=der.K,T=der.T,sig=der.vol,r=der.r,div=der.q)
            z = np.random.randn()
            Stn = St*np.exp(nudt+sigdt*z)
            cv += delta*(Stn-St*erddt)*np.exp(der.r*(der.T-t-dt))
            cv2 += (delta-1)*(Stn-St*erddt)*np.exp(der.r*(der.T-t-dt))
            St = Stn
        if der.opt_type == "call":
            C = dis*(max(St-der.K,0)+beta1*cv)
        else:
            C = dis*(max(-St+der.K,0)+beta1*cv2)
        sum_C += C
        sum_C2 += C**2
    mean_C = sum_C/m_
    se = np.sqrt((sum_C2-m_*mean_C**2)/(m_-1)/m_)
    return mean_C, se

res3 = Monte_Carlo_Del(n_=300,m_=1000,der=der1)


def Monte_Carlo_Anti_Del(n_,m_,der):
    start = time.time()
    dt = der.T/n_
    nudt = (der.r-der.q-(der.vol)**2*0.5)*dt
    sigdt = der.vol*np.sqrt(dt)
    dis = np.exp(-der.r*der.T)
    erddt = np.exp((der.r-der.q)*dt)
    beta1 = -1
    
    sum_C = 0
    sum_C2 = 0
    for i in range(int(m_)):
        St1 = der.S0
        St2 = der.S0
        cv1 = 0
        cv2 = 0
        cv3 = 0
        cv4 = 0
        for j in range(int(n_)):
            t = (j)*dt
            delta1 = Black_Scholes_delta(St = St1,t=t,K=der.K,T=der.T,sig=der.vol,r=der.r,div=der.q)
            delta2 = Black_Scholes_delta(St = St2,t=t,K=der.K,T=der.T,sig=der.vol,r=der.r,div=der.q)
            z = np.random.randn()
            Stn1 = St1*np.exp(nudt+sigdt*z)
            Stn2 = St2*np.exp(nudt+sigdt*(-z))
            cv1 += delta1*(Stn1-St1*erddt)*np.exp(der.r*(der.T-t-dt))
            cv2 += delta2*(Stn2-St2*erddt)*np.exp(der.r*(der.T-t-dt))
            cv3 += (delta1-1)*(Stn1-St1*erddt)*np.exp(der.r*(der.T-t-dt))
            cv4 += (delta1-1)*(Stn2-St2*erddt)*np.exp(der.r*(der.T-t-dt))
            St1 = Stn1
            St2 = Stn2
        if der.opt_type == "call":
            C = 0.5*dis*((max(St1-der.K,0)+beta1*cv1)+
                     (max(St2-der.K,0)+beta1*cv2))
        else:
            C = 0.5*dis*((max(-St1+der.K,0)+beta1*cv3)+
                     (max(-St2+der.K,0)+beta1*cv4))
        sum_C += C
        sum_C2 += C**2
    mean_C = sum_C/m_
    se = np.sqrt((sum_C2-m_*mean_C**2)/(m_-1)/m_)
    end = time.time()
    return mean_C, se, (end-start)

Monte_Carlo_Anti_Del(n_=300,m_=1000,der=der1)


# problem 2
#1
nx = 1e7*0.4/80
ny = 1e7*0.3/90000
yuanz = 1e7*0.3*6.1

def Monte_Carlo2(n_,m_,T,X0,Y0,Z0):
    dt = T/n_
    var_list = []
    for i in range(int(m_)):
        Xt = X0
        Yt = Y0
        Zt = Z0
        for j in range(int(n_)):
            t = j*dt
            z1 = np.random.randn()
            z2 = np.random.randn()
            z3 = np.random.randn()
            Xt += 0.01*Xt*dt+0.3*Xt*z1*np.sqrt(dt)
            Yt += 100*(90000+1000*t-Yt)*dt+np.sqrt(Yt)*z2*np.sqrt(dt)
            Zt += 5*(6-Zt)*Zt*dt+0.01*np.sqrt(Zt)*z3*np.sqrt(dt)
        P = nx*Xt+ny*Yt+yuanz/Zt
        var_list.append(P)
    var_list = np.asarray(var_list)
    quantile = np.quantile(var_list,0.01)
    VaR = 1e7-quantile
    cVaR = np.mean(1e7-var_list[var_list<=quantile])
    return var_list, VaR, cVaR


a1,a2,a3 = Monte_Carlo2(n_=np.ceil(10/252/0.001),m_=3e4,T=10/252,X0=80,Y0=90000,Z0=6.1)
#3.1
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

#3.2

def Monte_Carlo3(n_,m_,T,S0,mu0,sig0,A):
    dt = T/n_
    result = []
    L = Cholesky(A)
    for i in range(int(m_)):
        Xt = S0[0]
        Yt = S0[1]
        Zt = S0[2]
        res1 = []
        for j in range(int(n_)):
            row_z = np.random.randn(3)
            z = np.dot(L,row_z)
            z1 = z[0]
            z2 = z[1]
            z3 = z[2]
            Xt += mu0[0]*Xt*dt+sig0[0]*Xt*z1*np.sqrt(dt)
            Yt += mu0[1]*Yt*dt+sig0[1]*Yt*z2*np.sqrt(dt)
            Zt += mu0[2]*Zt*dt+sig0[2]*Zt*z3*np.sqrt(dt)
            res2 = [Xt,Yt,Zt]
            res1.append(res2)
        result.append(res1)
    return result

A_ = np.matrix([[1,0.5,0.2],
                [0.5,1,-0.4],
                [0.2,-0.4,1]])
res = Monte_Carlo3(n_=100,m_=1000,T=100/365,S0=[100,101,98],mu0=[0.03,0.06,0.02],sig0=[0.05,0.2,0.15],A=A_)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for mm in range(10):
    xt = []
    yt = []
    zt = []
    for nn in range(len(res[mm])):
        xt.append(res[mm][nn][0])
        yt.append(res[mm][nn][1])
        zt.append(res[mm][nn][2])
    ax.plot(range(100),np.asarray([mm]*100),xt,color = "red")
    ax.plot(range(100),np.asarray([mm]*100),yt,color = "blue")
    ax.plot(range(100),np.asarray([mm]*100),zt,color = "lightgreen")
ax.set_xlabel('time')
ax.set_ylabel('simulation')
ax.set_zlabel('price')


#3.3
def Monte_Carlo4(n_,m_,T,S0,mu0,sig0,A,a,opt_type,K,r):
    dt = T/n_
    L = Cholesky(A)
    dis = np.exp(-r*T)
    sum_C = 0
    for i in range(int(m_)):
        Xt = S0[0]
        Yt = S0[1]
        Zt = S0[2]
        for j in range(int(n_)):
            row_z = np.random.randn(3)
            z = np.dot(L,row_z)
            Xt = mu0[0]*Xt*dt+sig0[0]*Xt*z[0]*np.sqrt(dt)+Xt
            Yt = mu0[1]*Yt*dt+sig0[1]*Yt*z[1]*np.sqrt(dt)+Yt
            Zt = mu0[2]*Zt*dt+sig0[2]*Zt*z[2]*np.sqrt(dt)+Zt
        Ut = a[0]*Xt+a[1]*Yt+a[2]*Zt
        if opt_type == "call":
            C = dis*max(Ut-K,0)
        else:
            C = dis*max(-Ut+K,0)
        sum_C += C
    result = sum_C/m_
    return result

Monte_Carlo4(n_=100,m_=1000,T=1,S0 = [100,101,98],mu0=[0.03,0.06,0.02],sig0=[0.05,0.2,0.15],A=A_,
             a=[1/3,1/3,1/3],opt_type='call',K=100,r=0.06)


#3.4

def Monte_Carlo5(n_,m_,T,S0,mu0,sig0,A,a,opt_type,K,r,B):
    dt = T/n_
    L = Cholesky(A)
    dis = np.exp(-r*T)
    sum_C = 0
    for i in range(int(m_)):
        Xt = S0[0]
        Yt = S0[1]
        Zt = S0[2]
        indicator = 0
        for j in range(int(n_)):
            row_z = np.random.randn(3)
            z = np.dot(L,row_z)
            Xt = mu0[0]*Xt*dt+sig0[0]*Xt*z[0]*np.sqrt(dt)+Xt
            Yt = mu0[1]*Yt*dt+sig0[1]*Yt*z[1]*np.sqrt(dt)+Yt
            Zt = mu0[2]*Zt*dt+sig0[2]*Zt*z[2]*np.sqrt(dt)+Zt
            if Yt > B:
                indicator = 1
        if indicator == 1:
            C = dis*max(Yt-K,0)
        else:
            Ut = a[0]*Xt+a[1]*Yt+a[2]*Zt
            if opt_type == "call":
                C = dis*max(Ut-K,0)
            else:
                C = dis*max(-Ut+K,0)
        sum_C += C
    result = sum_C/m_
    return result
Monte_Carlo5(n_=100,m_=1000000,T=1,S0 = [100,101,98],mu0=[0.03,0.06,0.02],sig0=[0.05,0.2,0.15],A=A_
             ,a=[1/3,1/3,1/3],opt_type='call',K=100,r=0.06,B=104)
def Monte_Carlo6(n_,m_,T,S0,mu0,sig0,A,a,opt_type,K,r,B):
    dt = T/n_
    L = Cholesky(A)
    dis = np.exp(-r*T)
    sum_C = 0
    for i in range(int(m_)):
        Xt = S0[0]
        Yt = S0[1]
        Zt = S0[2]
        max_Yt = Yt
        max_Zt = Zt
        for j in range(int(n_)):
            row_z = np.random.randn(3)
            z = np.dot(L,row_z)
            Xt = mu0[0]*Xt*dt+sig0[0]*Xt*z[0]*np.sqrt(dt)+Xt
            Yt = mu0[1]*Yt*dt+sig0[1]*Yt*z[1]*np.sqrt(dt)+Yt
            Zt = mu0[2]*Zt*dt+sig0[2]*Zt*z[2]*np.sqrt(dt)+Zt
            if Yt > max_Yt:
                max_Yt = Yt
            if Zt > max_Zt:
                max_Zt = Zt
        if max_Yt>max_Zt:
            C = dis*max(Yt**2-K,0)
        else:
            Ut = a[0]*Xt+a[1]*Yt+a[2]*Zt
            if opt_type == "call":
                C = dis*max(Ut-K,0)
            else:
                C = dis*max(-Ut+K,0)
        sum_C += C
    result = sum_C/m_
    return result
Monte_Carlo6(n_=100,m_=100000,T=1,S0 = [100,101,98],mu0=[0.03,0.06,0.02],sig0=[0.05,0.2,0.15],A=A_
             ,a=[1/3,1/3,1/3],opt_type='call',K=100,r=0.06,B=104)

def Monte_Carlo7(n_,m_,T,S0,mu0,sig0,A,a,opt_type,K,r,B):
    dt = T/n_
    L = Cholesky(A)
    dis = np.exp(-r*T)
    sum_C = 0
    for i in range(int(m_)):
        Xt = S0[0]
        Yt = S0[1]
        Zt = S0[2]
        sum_Yt = Yt
        sum_Zt = Zt
        for j in range(int(n_)):
            row_z = np.random.randn(3)
            z = np.dot(L,row_z)
            Xt = mu0[0]*Xt*dt+sig0[0]*Xt*z[0]*np.sqrt(dt)+Xt
            Yt = mu0[1]*Yt*dt+sig0[1]*Yt*z[1]*np.sqrt(dt)+Yt
            Zt = mu0[2]*Zt*dt+sig0[2]*Zt*z[2]*np.sqrt(dt)+Zt
            sum_Yt += Yt
            sum_Zt += Zt
        if sum_Yt>sum_Zt:
            C = dis*max(sum_Yt/(n_+1)-K,0)
        else:
            Ut = a[0]*Xt+a[1]*Yt+a[2]*Zt
            if opt_type == "call":
                C = dis*max(Ut-K,0)
            else:
                C = dis*max(-Ut+K,0)
        sum_C += C
    result = sum_C/m_
    return result
Monte_Carlo7(n_=100,m_=100000,T=1,S0 = [100,101,98],mu0=[0.03,0.06,0.02],sig0=[0.05,0.2,0.15],A=A_
             ,a=[1/3,1/3,1/3],opt_type='call',K=100,r=0.06,B=104)


def Monte_Carlo_all(n_,m_,T,S0,mu0,sig0,A,a,opt_type,K,r,B):
    dt = T/n_
    L = Cholesky(A)
    dis = np.exp(-r*T)
    sum_C = 0
    for i in range(int(m_)):
        Xt = S0[0]
        Yt = S0[1]
        Zt = S0[2]
        sum_Yt = Yt
        sum_Zt = Zt
        max_Yt = Yt
        max_Zt = Zt
        indicator = 0
        for j in range(int(n_)):
            row_z = np.random.randn(3)
            z = np.dot(L,row_z)
            Xt = mu0[0]*Xt*dt+sig0[0]*Xt*z[0]*np.sqrt(dt)+Xt
            Yt = mu0[1]*Yt*dt+sig0[1]*Yt*z[1]*np.sqrt(dt)+Yt
            Zt = mu0[2]*Zt*dt+sig0[2]*Zt*z[2]*np.sqrt(dt)+Zt
            sum_Yt += Yt
            sum_Zt += Zt
            if Yt > max_Yt:
                max_Yt = Yt
            if Zt > max_Zt:
                max_Zt = Zt
            if Yt > B:
                indicator = 1
        if indicator == 1:
            C = dis*max(Yt-K,0)
        elif max_Yt>max_Zt:
            C = dis*max(Yt**2-K,0)
        elif sum_Yt>sum_Zt:
            C = dis*max(sum_Yt/(n_+1)-K,0)
        else:
            Ut = a[0]*Xt+a[1]*Yt+a[2]*Zt
            if opt_type == "call":
                C = dis*max(Ut-K,0)
            else:
                C = dis*max(-Ut+K,0)
        sum_C += C
    result = sum_C/m_
    return result
Monte_Carlo_all(n_=100,m_=100000,T=1,S0 = [100,101,98],mu0=[0.03,0.06,0.02],sig0=[0.05,0.2,0.15],A=A_
             ,a=[1/3,1/3,1/3],opt_type='call',K=100,r=0.06,B=104)











# Bonus


def Monte_Carlo8(n_,m_,T,S0,V0,k,theta,sig,rho,r,K,f1,f2,f3):
    start = time.time()
    dt = T/n_
    dis = np.exp(-r*T)
    sum_C = 0
    sum_C2 = 0
    for i in range(int(m_)):
        lnSt = np.log(S0)
        Vtt = V0
        Vt = V0
        for j in range(int(n_)):
            z1 = np.random.randn()
            z2 = np.random.randn()
            w1 = z1
            w2 = rho*z1+np.sqrt(1-rho**2)*z2
            Vtt = f1(Vtt)-k*dt*(f2(Vtt)-theta)+sig*f3(Vtt)**0.5*w1*np.sqrt(dt)
            lnSt += (r-0.5*Vt)*dt+np.sqrt(Vt)*w2*np.sqrt(dt)
            Vt = f3(Vtt)
        St = np.exp(lnSt)
        C = dis*max(St-K,0)
        sum_C += C
        sum_C2+= C**2
    mean_C = sum_C/m_
    bias = abs(6.8061-mean_C)
    se = np.sqrt((sum_C2-m_*mean_C**2)/(m_-1)/m_)
    end = time.time()
    return mean_C, bias, se, (end-start)

def fa(x):
    return max(x,0)
def fb(x):
    return abs(x)
def fc(x):
    return x

Monte_Carlo8(n_=1000,m_=10000,T=1,S0=100,V0=0.010201,
             k=6.21,theta=0.019,sig=0.61,rho=-0.7,r=0.0319,
             K=100,f1=fa,f2=fa,f3=fa)

Monte_Carlo8(n_=100,m_=50000,T=1,S0=100,V0=0.010201,
             k=6.21,theta=0.019,sig=0.61,rho=-0.7,r=0.0319,
             K=100,f1=fb,f2=fb,f3=fb)

Monte_Carlo8(n_=100,m_=50000,T=1,S0=100,V0=0.010201,
             k=6.21,theta=0.019,sig=0.61,rho=-0.7,r=0.0319,
             K=100,f1=fc,f2=fc,f3=fb)

Monte_Carlo8(n_=1000,m_=10000,T=1,S0=100,V0=0.010201,
             k=6.21,theta=0.019,sig=0.61,rho=-0.7,r=0.0319,
             K=100,f1=fc,f2=fc,f3=fa)

Monte_Carlo8(n_=1000,m_=10000,T=1,S0=100,V0=0.010201,
             k=6.21,theta=0.019,sig=0.61,rho=-0.7,r=0.0319,
             K=100,f1=fc,f2=fa,f3=fa)


def simpson_int(func,a,b,tol):
    n=10000
    delta = (b-a)/n
    x = np.linspace(a,b,n+1)
    f_x = np.asarray([func(i) for i in x])
    res0 = 0
    res1 = delta/3*(f_x[0]+f_x[-1]+4*f_x[1:-1][::2].sum()+2*f_x[1:-1][1::2].sum())
    while abs(res1-res0)>tol:
        n= n + 10000
        x = np.linspace(a,b,n+1)
        f_x = np.asarray([func(i) for i in x])
        delta = (b-a)/n
        res0 = res1
        res1 = delta/3*(f_x[0]+f_x[-1]+4*f_x[1:-1][::2].sum()+2*f_x[1:-1][1::2].sum())
    return res1

import scipy.integrate as integrate
def C_integral(tau,S0,V0,k,theta,sig,rho,r,K):
    u1 = 0.5
    u2 = -0.5
    a = k*theta
    b1 = k-rho*sig
    b2 = k
    
    def f1(u):
        com = np.complex(b1,-rho*sig*u)
        d1 = np.sqrt((-com)**2-sig**2*(np.complex(0,2*u1*u)-u**2))
        g1 = (com+d1)/(com-d1)
        C1 = np.complex(0,r*u*tau)+a/sig**2*((com+d1)*tau-2*np.log((1-g1*np.exp(d1*tau))/(1-g1)))
        D1 = (com+d1)/sig**2*((1-np.exp(d1*tau))/(1-g1*np.exp(d1*tau)))
        phi1 = np.exp(C1+D1*V0+np.complex(0,u*np.log(S0)))
        res = ((np.exp(np.complex(0,-np.log(K)*u))*phi1)/(np.complex(0,u))).real
        return res
    def f2(u):
        com = np.complex(b2,-rho*sig*u)
        d2 = np.sqrt((-com)**2-sig**2*(np.complex(0,2*u2*u)-u**2))
        g2 = (com+d2)/(com-d2)
        C2 = np.complex(0,r*u*tau)+a/sig**2*((com+d2)*tau-2*np.log((1-g2*np.exp(d2*tau))/(1-g2)))
        D2 = (com+d2)/sig**2*((1-np.exp(d2*tau))/(1-g2*np.exp(d2*tau)))
        phi2 = np.exp(C2+D2*V0+np.complex(0,u*np.log(S0)))
        res = ((np.exp(np.complex(0,-np.log(K)*u))*phi2)/(np.complex(0,u))).real
        return res
    P1 = 0.5+integrate.quad(f1,0.0001,1500)[0]/np.pi
    P2 = 0.5+integrate.quad(f2,0.0001,1500)[0]/np.pi
    result = S0*P1-K*np.exp(-r*tau)*P2
    return result

C_integral(tau=1,S0=100,V0=0.010201,
             k=6.21,theta=0.019,sig=0.61,rho=-0.7,r=0.0319,
             K=100)


simpson_int(func=(lambda x: x**2),a=0,b=100,n=10000)


tau=1
S0=100
V0=0.010201
k=6.21
theta=0.019
sig=0.61
rho=-0.7
r=0.0319
K=100





















