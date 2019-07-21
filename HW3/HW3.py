# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 2019

@author: Qiao Guanzhuo
"""
import scipy.integrate as integrate
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import norm


def transition_p(t,x,x0,alpha,gamma,beta):
    Q = alpha*gamma/2-beta**2/8
    def sigma(x):
        return (alpha*x**2+beta*x+gamma)
    s_x = integrate.quad(lambda xx: 1/sigma(xx),x0,x)[0]
    res = 1/(sigma(x)*math.sqrt(2*math.pi*t))*sigma(x0)/sigma(x)*math.exp(-s_x**2/2/t+Q*t)
    return res#,s_x,1/(sigma(x)*math.sqrt(2*math.pi*t))*sigma(x0)/sigma(x),math.exp(-s_x**2/2/t+Q*t)
#transition_p(t=10,x=45,x0=40,alpha=0.0001,gamma=0,beta=0.1)

X = np.arange(40, 50, 0.1)
Y = np.arange(30, 60, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i][j] = transition_p(t=30,x=Y[i][j],x0=X[i][j],alpha=0.0001,gamma=-0.0025,beta=0.001)

# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel("x_0")
ax.set_ylabel("x")
ax.set_zlabel("P")
#transition_p(t=1,x=1,x0=10,alpha=0.1,gamma=0.1,beta=1)
def explicit_fd_grid(T,N,dx,Nj,x0,alpha,beta,gamma):
    #build_tree(self):
    delta_t = T/N
    def sigma(x):
        return (alpha*x**2+beta*x+gamma)
    p = np.asarray([0]*(2*Nj+1))
    p[Nj] = 1
    X_ = np.asarray([x0+Nj*dx-i*dx for i in range(2*Nj+1)])
    sig_X_ = np.vectorize(sigma)(X_[1:-1])
    p_u = sig_X_**2*delta_t/(2*dx**2)
    p_m = 1-sig_X_**2*delta_t/(1*dx**2)
    p_d = sig_X_**2*delta_t/(2*dx**2)
    #def euro_discount(self):
    N_ = N
    while (N_>0):
        dis_p = (p_u*p[:-2]+p_m*p[1:-1]+p_d*p[2:])
        p_large = dis_p[0]
        p_small = dis_p[-1]
        p = np.concatenate(([p_large],dis_p,[p_small]))
        N_ -= 1
    return p
explicit_fd_grid(T=2,N=1000,dx=1,Nj=40,x0=40,alpha=0.0001,beta=0.1,gamma=0)
nj = 40
p = explicit_fd_grid(T=2,N=1000,dx=1,Nj=nj,x0=50,alpha=0.0001,beta=0.1,gamma=0)

#x0->x
p[nj]
#transition_p(t=2,x=50,x0=50,alpha=0.0001,gamma=0,beta=0.1)
p_=[]
for i in range(50+nj,50-nj-1,-1):
    p_.append(transition_p(t=2,x=i,x0=50,alpha=0.0001,gamma=0,beta=0.1))
p_ = np.asarray(p_)
np.mean((p-p_)**2)

def C(t_, K, x0,alpha,gamma,beta):
    Q = alpha*gamma/2-beta**2/8
    def sigma(x):
        return (alpha*x**2+beta*x+gamma)
    s = abs(integrate.quad(lambda xx: 1/sigma(xx),x0,K)[0])
    dis1 = math.exp(s*math.sqrt(-Q))
    dis2 = math.exp(-s*math.sqrt(-Q))
    d1 = -s/math.sqrt(2*t_)-math.sqrt(-2*Q*t_)
    d2 = d1+2*math.sqrt(-2*Q*t_)
    res = max(x0-K,0)+sigma(K)*sigma(x0)/(2*math.sqrt(-2*Q))*(dis1*norm.cdf(d1)-dis2*norm.cdf(d2))
    return res#, sigma(x0),math.sqrt(sigma(x0)*sigma(K)), s, dis1, dis2,d1,d2, sigma(K)*sigma(x0)/(2*math.sqrt(-2*Q)),(dis1*norm.cdf(d1)-dis2*norm.cdf(d2)),Q,s
C(t_=1, K=90, x0=100,alpha=0,gamma=0,beta=0.03)
def BS_Formula(type_opt, r, vol, K, S, T):
    d_1 = float(math.log(S/K)+(r+vol**2/2)*T)/float(vol*math.sqrt(T))
    d_2 = d_1-vol*math.sqrt(T)
    if type_opt == 'call':
        return norm.cdf(d_1)*S-K*math.exp(-r*T)*norm.cdf(d_2)
    else:
        return K*math.exp(-r*T)*norm.cdf(-d_2)-norm.cdf(-d_1)*S
BS_Formula(type_opt='call', r=0, vol=0.03, K=90, S=100, T=1)
C(t_=1, K=90, x0=100,alpha=0.0001,gamma=-3,beta=0.1)
############################################################################################

def phi(t,mu,sigma):
    return np.exp(np.complex(-sigma**2*t**2/2,t*mu))

def PHI(v,r,T,alpha,s0,sigma_):
    value_phi = phi(np.complex(v,-(alpha+1)),np.log(s0)+(r-sigma_**2/2)*T,sigma_*np.sqrt(T))
    up = np.exp(-r*T)*value_phi
    down = np.complex(alpha**2+alpha-v**2,(2*alpha+1)*v)
    return up/down

def input_generator(v,b,eta,r,T,alpha,s0,sigma_):
    A = np.exp(np.complex(0,b*v))
    B = PHI(v,r,T,alpha,s0,sigma_)
    #B = B.real
    C = eta
    return A*B*C


N = 2000
k = np.log(80)
b_ = np.ceil(k)+5
lambda_ = 2*b_/N
eta_ = 2*np.pi/N/lambda_
alpha_ = 10
vv = np.array([eta_*i for i in range(0,N)])
kk = np.array([np.round(-b_+lambda_*i,2) for i in range(0,N)])
for ind in range(len(kk)):
    if kk[ind] == np.round(k,2):
        break
J = 1
input_array = [0]*N
while J<=N:
    input_array[J-1] = input_generator(v=vv[J-1],b=b_,eta=eta_,r=0.05,T=1,alpha=alpha_,s0=80,sigma_=0.1)
    J=J+1
input_array = np.asarray(input_array)
res = np.fft.fft(input_array)[ind].real*np.exp(-alpha_*kk[ind])/np.pi
BS_Formula(type_opt='call', r=0.05, vol=0.1, K=80, S=80, T=1)


import os
import pandas as pd
import scipy as sp
os.chdir(r'D:\Grad 2\621\assignment\HW3') 
swap_dt = pd.read_excel('2017_2_15_mid.xlsx')
swap_dt = swap_dt.loc[:,["Expiry","Unnamed: 1","3Yr","10Yr"]]
swap_vol = swap_dt.loc[::2,"3Yr"]
swap_strike = swap_dt.loc[1::2,"3Yr"]

def sig_B(p,f,beta_,T_):
    alpha,rho,vega = p
    A = pow(1-beta_,2)*pow(alpha,2)/(24*pow(f,2-2*beta_))
    B = rho*beta_*vega*alpha/(4*pow(f,1-beta_))
    C = (2-3*pow(rho,2))*pow(vega,2)/24
    sig_B = alpha*(1+(A+B+C)*T_)/f**(1-beta_)
    return sig_B

def sum_error(p,f,y,beta_,T_):
    error = np.sum((y-sig_B(p,f,beta_,T_))**2)
    return error

p0 = [0.1,0.1,0.1]
res_1 = sp.optimize.minimize(sum_error,p0,args=(np.asarray(swap_strike)/100,np.asarray(swap_vol)/10000,0.5,3))
res_2 = sp.optimize.minimize(sum_error,p0,args=(np.asarray(swap_strike)/100,np.asarray(swap_vol)/10000,0.7,3))
res_3 = sp.optimize.minimize(sum_error,p0,args=(np.asarray(swap_strike)/100,np.asarray(swap_vol)/10000,0.4,3))

parameters_1 = res_1['x']
parameters_2 = res_2['x']
parameters_3 = res_3['x']
sse_1 = res_1['fun']
sse_2 = res_2['fun']
sse_3 = res_3['fun']
min(sse_1,sse_2,sse_3)

np.mean((sig_B(p=list(parameters_1),f=np.asarray(swap_strike)/100,beta_=0.5,T_=3)-np.asarray(swap_vol)/10000)**2)
res_all = pd.DataFrame([parameters_3,
                        parameters_1,
                        parameters_2],index = ['beta=0.4','beta=0.5','beta=0.7'],columns = ['alpha','rho','nu'])
res_all.loc[:,'sse'] =[sse_3,sse_1,sse_2] 



swap_vol_test = swap_dt.loc[::2,"10Yr"]
swap_strike_test = swap_dt.loc[1::2,"10Yr"]

swap_vol_est = sig_B(parameters_2,np.asarray(swap_strike_test)/100,0.7,10)*10000
np.mean((swap_vol_est-swap_vol_test)**2)
res_bonus = pd.DataFrame([swap_vol_est,swap_vol_test])
res_bonus = res_bonus.T
res_bonus.columns = ['estimation','market data']







