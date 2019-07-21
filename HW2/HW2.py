# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:53:47 2019

@author: Qiao Guanzhuo
"""

import pandas as pd
import math as m
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
import os

#1a
class Payoff(object):
    def __init__(self,Strike):
        self.Strike = Strike
    def getpayoff(self):
        pass
"""
vanilla payoff funciton
"""
class callpayoff(Payoff):
    def __init__(self,Strike):
        Payoff.__init__(self,Strike)
    def getpayoff(self,Price):
        return np.asarray([max(Price-self.Strike,0)])
    def getnodeprice(self,Price,Dis_price):
        return np.asarray(Dis_price)
    def getidentity(self):
        return "callpayoff"

class putpayoff(Payoff):
    def __init__(self,Strike):
        Payoff.__init__(self,Strike)
    def getpayoff(self,Price):
        return np.asarray([max(self.Strike-Price,0)])
    def getnodeprice(self,Price,Dis_price):
        return np.asarray(Dis_price)
    def getidentity(self):
        return "putpayoff"

class tree():
    def __init__(self,T,S,r,sigma,N,payoff,D):
        self.T = T
        self.S = S
        self.r = r
        self.sigma = sigma
        self.N = N
        self.payoff = payoff
        self.D = D
    def build_tree(self):
        pass

class additive_binomial_tree(tree):
    def __init__(self,T,S,r,sigma,N,payoff,D):
        tree.__init__(self,T,S,r,sigma,N,payoff,D)
    def build_tree(self):
        self.delta_t = self.T/self.N
        self.nu = self.r-self.D-0.5*self.sigma**2
        self.x_u = m.sqrt(self.delta_t*self.sigma**2+self.nu**2*self.delta_t**2)
        self.x_d = -self.x_u
        self.p_u = 0.5+0.5*((self.nu*self.delta_t)/self.x_u)
        self.p_d = 1-self.p_u
        self.disc = np.exp(-self.r*self.delta_t)
        self.St = self.S*np.exp(np.asarray([i*self.x_d+(self.N-i)*self.x_u for i in range(self.N+1)]))
        self.C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
    def euro_discount(self):
        self.build_tree()
        while (len(self.C)>1):
            self.dis_C = self.disc*(self.p_u*self.C[:-1]+self.p_d*self.C[1:])# compute discounted value of product
            self.St = np.exp(self.x_d)*self.St[:-1]# compute stock price at that node
            self.C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])# apply the condition on node
        return self.C[0]
    def amer_discount(self):
        self.build_tree()
        while (len(self.C)>1):
            self.dis_C = self.disc*(self.p_u*self.C[:-1]+self.p_d*self.C[1:])
            self.St = np.exp(self.x_d)*self.St[:-1]
            self.exc_C = np.asarray([self.payoff.getpayoff(p) for p in self.St])# by doing this, we already had the condition on the execute price
            self.dis_C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])
            self.C = np.where(self.dis_C < self.exc_C, self.exc_C, self.dis_C) # do the comparation
        return self.C[0]
# =============================================================================
# def callpayoff(Strike,Price):
#     return max(Price-Strike,0)
# def putpayoff(Strike,Price):
#     return max(Strike-Price,0)
# 
# 
# def additive_binomial_tree(K,T,S,r,sigma,N):
#     delta_t = T/N
#     nu = r-0.5*sigma**2
#     x_u = m.sqrt(delta_t*sigma**2+nu**2*delta_t**2)
#     x_d = -x_u
#     p_u = 0.5+0.5*((nu*delta_t)/x_u)
#     p_d = 1-p_u
#     disc = np.exp(-r*delta_t)
#     St = S*np.exp(np.asarray([i*x_d+(N-i)*x_u for i in range(N+1)]))
#     C = np.asarray([callpayoff(K,i) for i in St])
#     while (len(C)>1):
#         C = disc*(p_u*C[:-1]+p_d*C[1:])
#     return C[0]
# =============================================================================


#1b
os.chdir(r'D:\Grad 2\621\assignment\HW2\data') 
Data2 = pd.read_csv('data12.csv',index_col=0)
Data2 = Data2.loc[:1493,:]
Data2 = Data2[(Data2.loc[:,"Expiry"] == "2019-02-22") | (Data2.loc[:,"Expiry"] == "2019-03-22") | (Data2.loc[:,"Expiry"] == "2019-04-18")]
Data2 = Data2[(Data2['Strike']/Data2['Underlying_Price_y']>0.95) & (Data2['Strike']/Data2['Underlying_Price_y']<1.05)]
Data2 = Data2.sort_values(by = ["Expiry","Strike"],ascending=(True,True))
Data2_call = Data2[Data2.Type == "call"]
Data2_put = Data2[Data2.Type == "put"]
r=0.024
for ind1 in Data2_call.index:
    Data2_call.loc[ind1,"Tree_Euro_Price"] = additive_binomial_tree(Data2_call.loc[ind1,"TtM_y"],
                  Data2_call.loc[ind1,"Underlying_Price_y"],r,
                  Data2_call.loc[ind1,"Implied_vol_bis"],400,callpayoff(Data2_call.loc[ind1,"Strike"]),D=0).euro_discount()
    Data2_call.loc[ind1,"Tree_Amer_Price"] = additive_binomial_tree(Data2_call.loc[ind1,"TtM_y"],
                  Data2_call.loc[ind1,"Underlying_Price_y"],r,
                  Data2_call.loc[ind1,"Implied_vol_bis"],400,callpayoff(Data2_call.loc[ind1,"Strike"]),D=0).amer_discount()
for ind2 in Data2_put.index:
    Data2_put.loc[ind2,"Tree_Euro_Price"] = additive_binomial_tree(Data2_put.loc[ind2,"TtM_y"],
                  Data2_put.loc[ind2,"Underlying_Price_y"],r,
                  Data2_put.loc[ind2,"Implied_vol_bis"],400,putpayoff(Data2_put.loc[ind2,"Strike"]),D=0).euro_discount()
    Data2_put.loc[ind2,"Tree_Amer_Price"] = additive_binomial_tree(Data2_put.loc[ind2,"TtM_y"],
                  Data2_put.loc[ind2,"Underlying_Price_y"],r,
                  Data2_put.loc[ind2,"Implied_vol_bis"],400,putpayoff(Data2_put.loc[ind2,"Strike"]),D=0).amer_discount()
Data2_call.index = range(len(Data2_call))
Data2_put.index = range(len(Data2_put))


Data2_call.head()
Data2_put.head()

plt.plot(Data2_call["Avr_Price"])
plt.plot(Data2_call["BS_Price"])
plt.plot(Data2_call["Tree_Euro_Price"])
plt.plot(Data2_call["Tree_Amer_Price"])
plt.xlabel('Price')
plt.ylabel('Index')
plt.title('Prices of Call Options')



#1d
def BS_Formula(type_opt, r, vol, K, S, T, q=0):
    d_1 = float(m.log(S/K)+(r-q+vol**2/2)*T)/float(vol*m.sqrt(T))
    d_2 = d_1-vol*m.sqrt(T)
    if type_opt == 'call':
        return norm.cdf(d_1)*S*m.exp(-q*T)-K*m.exp(-r*T)*norm.cdf(d_2)
    else:
        return K*m.exp(-r*T)*norm.cdf(-d_2)-norm.cdf(-d_1)*S*m.exp(-q*T)

N = [10,20,30,40,50,100,150,200,250,300,350,400]
diff = []
for n in N:
    P_bs = BS_Formula("put",0.06,0.2,100,100,1)
    P_bt = additive_binomial_tree(1,100,0.06,0.2,n,putpayoff(100),D=0).euro_discount()[0]
    diff.append(abs(P_bs-P_bt))

plt.plot(N,diff,'g*-')
plt.xlabel('Eror')
plt.ylabel('Steps')
plt.title('Absolute Error')
for i in range(0,len(N)):
    plt.text(N[i],diff[i],str(round(diff[i],4)), family='serif', style='italic', ha='right', wrap=True)

#1bonus
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

def get_iv_tree(type_opt, r, K, S, T, P, N, tolerance,up,down):
    if type_opt == "call":
        obj_func = lambda x: additive_binomial_tree(T,S,r,x,N,callpayoff(K),D=0).amer_discount()-P
    else:
        obj_func = lambda x: additive_binomial_tree(T,S,r,x,N,putpayoff(K),D=0).amer_discount()-P
    return Bisection(obj_func,tolerance,up,down)

def get_iv_bs(type_opt, r, K, S, T, P,tolerance,up,down):
    obj_func= lambda x: BS_Formula(type_opt, r, x, K, S, T)-P
    return Bisection(obj_func,tolerance,up,down)

for ind3 in Data2.index[:50]:
    Data2.loc[ind3,"IV_tree"] = get_iv_tree(Data2.loc[ind3,"Type"], r,
             Data2.loc[ind3,"Strike"], Data2.loc[ind3,"Underlying_Price_y"], Data2.loc[ind3,"TtM_y"], 
             get_avrprice(Data2.loc[ind3,"Bid_y"],Data2.loc[ind3,"Ask_y"]), 200, 10**-6,1,0.01)
    Data2.loc[ind3,"IV_bs"] = get_iv_bs(Data2.loc[ind3,"Type"], r,
             Data2.loc[ind3,"Strike"], Data2.loc[ind3,"Underlying_Price_y"], Data2.loc[ind3,"TtM_y"], 
             get_avrprice(Data2.loc[ind3,"Bid_y"],Data2.loc[ind3,"Ask_y"]), 10**-6,1,0.01)

#2a
class additive_trinomial_tree(tree):
    def __init__(self,T,S,r,sigma,N,payoff,dx,D):
        tree.__init__(self,T,S,r,sigma,N,payoff,D)
        self.dx = dx
    def build_tree(self):
        self.delta_t = self.T/self.N
        self.nu = self.r-self.D-0.5*self.sigma**2
        self.p_u = 0.5*((self.sigma**2*self.delta_t+self.nu**2*self.delta_t**2)/(self.dx**2)+(self.nu*self.delta_t)/self.dx)
        self.p_m = 1-(self.sigma**2*self.delta_t+self.nu**2*self.delta_t**2)/(self.dx**2)
        self.p_d = 0.5*((self.sigma**2*self.delta_t+self.nu**2*self.delta_t**2)/(self.dx**2)-(self.nu*self.delta_t)/self.dx)
        self.disc = np.exp(-self.r*self.delta_t)
        self.St = self.S*np.exp(np.asarray([self.N*self.dx-i*self.dx for i in range(2*self.N+1)]))
        self.C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
    def euro_discount(self):
        self.build_tree()
        while (len(self.C)>1):
            self.C = self.disc*(self.p_u*self.C[:-2]+self.p_m*self.C[1:-1]+self.p_d*self.C[2:])
        return self.C[0]
    def amer_discount(self):
        self.build_tree()
        while (len(self.C)>1):
            self.C = self.disc*(self.p_u*self.C[:-2]+self.p_m*self.C[1:-1]+self.p_d*self.C[2:])
            self.St = self.St[1:-1]
            self.C_exc = np.asarray([self.payoff.getpayoff(p) for p in self.St])
            self.C = np.where( self.C < self.C_exc, self.C_exc, self.C)
        return self.C[0]

price1 = additive_trinomial_tree(1,100,0.06,0.25,400,callpayoff(100),0.022,0.03).euro_discount()[0]
price2 = BS_Formula("call",0.06,0.25,100,100,1,0.03)
price3 = additive_binomial_tree(1,100,0.06,0.25,400,callpayoff(100),D=0.03).euro_discount()[0]
price4 = additive_trinomial_tree(1,100,0.06,0.25,400,putpayoff(100),0.022,0.03).euro_discount()[0]
price5 = BS_Formula("put",0.06,0.25,100,100,1,0.03)
price6 = additive_binomial_tree(1,100,0.06,0.25,400,putpayoff(100),D=0.03).euro_discount()[0]

price_result1 =  pd.DataFrame([[price1,price2,price3],
                              [price4,price5,price6]],index =["European Call","European Put"],columns = ["Trinomial","BS","Binomial"])



price7 = additive_trinomial_tree(1,100,0.06,0.25,400,callpayoff(100),0.022,0.03).amer_discount()[0]
price8 = additive_binomial_tree(1,100,0.06,0.25,400,callpayoff(100),D=0.03).amer_discount()[0]
price9 = additive_trinomial_tree(1,100,0.06,0.25,400,putpayoff(100),0.022,0.03).amer_discount()[0]
price10 = additive_binomial_tree(1,100,0.06,0.25,400,putpayoff(100),D=0.03).amer_discount()[0]
price_result2 =  pd.DataFrame([[price7,price8],
                              [price9,price10]],index =["European Call","European Put"],columns = ["Trinomial","Binomial"])




N = [10,20,30,40,50,100,150,200,250,300,350,400]
diff = []

for n in [10,20,30,40,50,100,150,200,250,300,350,400]:
    P_bs = BS_Formula("put",0.06,0.25,100,100,1,0.03)
    P_bt = additive_trinomial_tree(1,100,0.06,0.25,n,putpayoff(100),0.25*m.sqrt(3*1/n)+0.01,0.03).euro_discount()[0]
    diff.append(abs(P_bs-P_bt))

plt.plot(N,diff,'g*-')
plt.xlabel('Eror')
plt.ylabel('Steps')
plt.title('Absolute Error')
for i in range(0,len(N)):
    plt.text(N[i],diff[i],str(round(diff[i],4)), family='serif', style='italic', ha='right', wrap=True)

#3a
class up_out_callpayoff(Payoff):
    def __init__(self,Strike,Barrier):
        Payoff.__init__(self,Strike)
        self.Barrier = Barrier
    def getpayoff(self,Price):
        if Price >= self.Barrier:
            return np.asarray([0])
        else:
            return np.asarray([max(Price-self.Strike,0)])
    def getnodeprice(self,Price,Dis_price):
        if Price >=self.Barrier:
            return np.asarray([0])
        else:
            return np.asarray(Dis_price)
    def getidentity(self):
        return "callpayoff"

additive_binomial_tree(0.3,10,0.01,0.2,1000,up_out_callpayoff(Strike = 10,Barrier = 11),D=0).euro_discount()
#3b
def call_ui(r, vol, K, S, T, H, q=0):
    v = r-q-vol**2/2
    def C_bs(x1,x2):
        return BS_Formula("call",r,vol,x2,x1,T,q)
    def P_bs(x1,x2):
        return BS_Formula("put",r,vol,x2,x1,T,q)
    def d_bs(x1,x2):
        return ((m.log(x1/x2)+v*T)/(vol*m.sqrt(T)))
    UI_bs = ((H/S)**(2*v/vol**2))*(P_bs(H**2/S,K)-P_bs(H**2/S,H)+(H-K)*m.exp(-r*T)*norm.cdf(-d_bs(H,S)))+C_bs(S,H)+(H-K)*m.exp(-r*T)*norm.cdf(d_bs(S,H))
    return UI_bs

def call_uo(r, vol, K, S, T, H, q):
    v = r-q-vol**2/2
    def d_bs(x1,x2):
        return ((m.log(x1/x2)+v*T)/(vol*m.sqrt(T)))
    def C_bs(x1,x2):
        return BS_Formula("call",r,vol,x2,x1,T,q)
    UO_bs = C_bs(S,K)-C_bs(S,H)-(H-K)*m.exp(-r*T)*norm.cdf(d_bs(S,H))-((H/S)**(2*v/vol**2))*(C_bs(H**2/S,K)-C_bs(H**2/S,H)-(H-K)*m.exp(-r*T)*norm.cdf(d_bs(H,S)))
    return UO_bs

def call_di(r, vol, K, S, T, H, q):
    v = r-q-vol**2/2
    def C_bs(x1,x2):
        return BS_Formula("call",r,vol,x2,x1,T,q)
    DI_bs = (H/S)**(2*v/vol**2)*C_bs(H**2/S,K)
    return DI_bs

def call_do(r, vol, K, S, T, H, q):
    v = r-q-vol**2/2
    def C_bs(x1,x2):
        return BS_Formula("call",r,vol,x2,x1,T,q)
    DI_bs = C_bs(S,K)-(H/S)**(2*v/vol**2)*C_bs(H**2/S,K)
    return DI_bs

uo = call_uo(r=0.01, vol=0.2, K=10, S=10, T=0.3, H=11, q=0)
#3c
ui = call_ui(r=0.01, vol=0.2, K=10, S=10, T=0.3, H=11, q=0)
c = BS_Formula("call",r=0.01, vol=0.2, K=10, S=10, T=0.3, q=0)
c-uo
#3d
class up_out_putpayoff(Payoff):
    def __init__(self,Strike,Barrier):
        Payoff.__init__(self,Strike)
        self.Barrier = Barrier
    def getpayoff(self,Price):
        if Price >= self.Barrier:
            return np.asarray([0])
        else:
            return np.asarray([max(self.Strike-Price,0)])
    def getnodeprice(self,Price,Dis_price):
        if Price >=self.Barrier:
            return np.asarray([0])
        else:
            return np.asarray(Dis_price)
    def getidentity(self):
        return "putpayoff"
additive_binomial_tree(0.3,10,0.01,0.2,400,up_out_putpayoff(Strike = 10,Barrier = 11),D=0).amer_discount()
additive_binomial_tree(0.3,10,0.01,0.2,400,putpayoff(Strike = 10),D=0).amer_discount()

class up_in_putpayoff(Payoff):
    def __init__(self,Strike,Barrier):
        Payoff.__init__(self,Strike)
        self.Barrier = Barrier
    def getpayoff(self,Price):
        if Price <= self.Barrier:
            return np.array([max(self.Strike-Price,0),0])
        else:
            return np.array([max(self.Strike-Price,0),max(self.Strike-Price,0)])
    def getnodeprice(self,Price,Dis_price):
        if Price <=self.Barrier:
            return Dis_price
        else:
            return np.asarray([Dis_price[0],Dis_price[0]])
    def getidentity(self):
        return "putpayoff"
additive_binomial_tree(0.3,10,0.01,0.2,400,up_in_putpayoff(Strike = 10,Barrier = 11),D=0).amer_discount()[1]

class explicit_fd_grid(tree):
    def __init__(self,T,S,r,sigma,N,payoff,dx,D,Nj):
        tree.__init__(self,T,S,r,sigma,N,payoff,D)
        self.Nj = Nj
        self.dx = dx
    def build_tree(self):
        self.delta_t = self.T/self.N
        self.nu = self.r-self.D-0.5*self.sigma**2
        self.p_u = 0.5*(self.sigma**2*self.delta_t/self.dx**2+self.nu*self.delta_t/self.dx)
        self.p_m = 1-self.sigma**2*self.delta_t/self.dx**2-self.r*self.delta_t
        self.p_d = 0.5*(self.sigma**2*self.delta_t/self.dx**2-self.nu*self.delta_t/self.dx)      
        self.St = self.S*np.exp(np.asarray([self.Nj*self.dx-i*self.dx for i in range(2*self.Nj+1)]))
        self.C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
    def euro_discount(self):
        self.build_tree()
        N = self.N
        while (N>0):
            self.dis_C = (self.p_u*self.C[:-2]+self.p_m*self.C[1:-1]+self.p_d*self.C[2:])# we don't need to discount becasue of the finite difference method
            if self.payoff.getidentity() == "callpayoff":
                C_large = self.dis_C[0]+self.St[0]-self.St[1]
                C_small = self.dis_C[-1]
            else:
                C_large = self.dis_C[0]
                C_small = self.dis_C[-1]-(self.St[-1]-self.St[-2])
            self.dis_C = np.concatenate(([C_large],self.dis_C,[C_small]))
            self.C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])# apply the condition on node
            N -= 1
        return self.C[self.Nj]
    def amer_discount(self):
        self.build_tree()
        N = self.N
        while (N>0):
            self.dis_C = (self.p_u*self.C[:-2]+self.p_m*self.C[1:-1]+self.p_d*self.C[2:])# compute discounted value of product
            if self.payoff.getidentity() == "callpayoff":
                C_large = self.dis_C[0]+self.St[0]-self.St[1]
                C_small = self.dis_C[-1]
            else:
                C_large = self.dis_C[0]
                C_small = self.dis_C[-1]-(self.St[-1]-self.St[-2])
            self.dis_C = self.dis_C = np.concatenate(([C_large],self.dis_C,[C_small]))
            self.dis_C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])# apply the condition on node
            self.exc_C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
            self.C = np.where(self.dis_C < self.exc_C, self.exc_C, self.dis_C) # do the comparation
            N -= 1
        return self.C[self.Nj]
exgrid = explicit_fd_grid(T=1,S=100,r=0.06,sigma=0.2,N=3,payoff = callpayoff(100),dx=0.2,D=0.03,Nj=3)
explicit_fd_grid(T=1,S=100,r=0.06,sigma=0.2,N=3,payoff = callpayoff(100),dx=0.2,D=0.03,Nj=3).euro_discount()
explicit_fd_grid(T=1,S=100,r=0.06,sigma=0.2,N=3,payoff = putpayoff(100),dx=0.2,D=0.03,Nj=3).amer_discount()

# =============================================================================
# def LU_solve(A,z):
#     """
#     LU Decomposition
#     """
#     L,U = np.identity(A.shape[0]), np.zeros(A.shape)
#     U[0][0] = A[0][0]
#     for j in range(0,A.shape[0]-1):
#         try:
#             U[j][j+1] = A[j][j+1]
#             L[j+1][j] = A[j+1][j]/U[j][j]
#             U[j+1][j+1] = A[j+1][j+1]-L[j+1][j]*A[j][j+1]
#         except IndexError:
#             continue
#     """
#     Forward Substitution
#     """
#     y = np.zeros((L.shape[0],z.shape[1]))
#     for p  in range(y.shape[1]):
#         y[0][p] = z[0][p]
#         try:
#             for  i in range(1,L.shape[0]):
#                 y[i][p] = z[i][p]-sum(L[i][j]*y[j][p] for j in range(0,i))
#         except:
#             y[i][p] = np.NaN
#     """
#     Backward Substitution
#     """
#     x = np.zeros((U.shape[0],y.shape[1]))
#     for q in range(x.shape[1]):
#         x[-1][q] = y[-1][q]/U[-1][-1]
#         try:
#             for i in range(-1,-U.shape[0],-1):
#                 x[i-1][q] = (y[i-1][q] - sum(U[i-1][j]*x[j][0] for j in range(-1,i-1,-1)))/U[i-1][i-1]
#         except ZeroDivisionError:
#             x[i-1][q] = np.NaN
#     
#     return x
# 
# 
# 
# 
# class implicit_fd_grid(tree):
#     def __init__(self,T,S,r,sigma,N,payoff,dx,D,Nj):
#         tree.__init__(self,T,S,r,sigma,N,payoff,D)
#         self.Nj = Nj
#         self.dx = dx
#     def build_tree(self):
#         self.delta_t = self.T/self.N
#         self.nu = self.r-self.D-0.5*self.sigma**2
#         self.p_u = -0.5*self.delta_t*((self.sigma/self.dx)**2+self.nu/self.dx)
#         self.p_m = 1+self.delta_t*(self.sigma/self.dx)**2+self.r*self.delta_t
#         self.p_d = -0.5*self.delta_t*((self.sigma/self.dx)**2-self.nu/self.dx)
#         
#         self.St = self.S*np.exp(np.asarray([self.Nj*self.dx-i*self.dx for i in range(2*self.Nj+1)]))
#         self.C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
#     def euro_discount(self):
#         self.build_tree()
#         self.A = np.zeros((2*self.Nj+1,2*self.Nj+1))
#         self.A[0][0] = 1
#         self.A[0][1] = -1
#         self.A[-1][-1] = 1
#         self.A[-1][-2] = -1
#         for row  in range(1,len(self.A)-1):
#             self.A[row][row-1] = self.p_u
#             self.A[row][row] = self.p_m
#             self.A[row][row+1] = self.p_d
#         while (self.N>0):
#             if self.payoff.getidentity() == "callpayoff":
#                 lambda_u = np.asarray([self.St[0] - self.St[1]])
#                 lambda_l = np.asarray([0])
#             else:
#                 lambda_u = np.asarray([0])
#                 lambda_l = np.asarray([-(self.St[-1] - self.St[-2])])
#             self.C = np.concatenate(([lambda_u],self.C[1:-1],[lambda_l]))
#             self.dis_C = LU_solve(self.A,self.C)
#             self.C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])# apply the condition on node
#             self.N -= 1
#         return self.C[self.Nj]
#     def amer_discount(self):
#         self.build_tree()
#         #build A
#         A = np.zeros((2*self.Nj+1,2*self.Nj+1))
#         A[0][0] = 1
#         A[0][1] = -1
#         A[-1][-1] = 1
#         A[-1][-2] = -1
#         for row  in range(1,len(A)-1):
#             A[row][row-1] = self.p_u
#             A[row][row] = self.p_m
#             A[row][row+1] = self.p_d
#         while (self.N>0):
#             #build C
#             if self.payoff.getidentity() == "callpayoff":
#                 lambda_u = np.asarray([self.St[0] - self.St[1]])
#                 lambda_l = np.asarray([0])
#             else:
#                 lambda_u = np.asarray([0])
#                 lambda_l = np.asarray([-(self.St[-1] - self.St[-2])])
#             self.C = np.concatenate(([lambda_u],self.C[1:-1],[lambda_l]))            
#             #solve equation
#             self.dis_C = LU_solve(A,self.C)
#             self.dis_C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])
#             self.exc_C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
#             self.C = np.where(self.dis_C < self.exc_C, self.exc_C, self.dis_C) # do the comparation
#             self.N -= 1
#         return self.C[self.Nj]
# 
# =============================================================================
def solve(p_u,p_m,p_d,C_ip1,Nj):
    pmp = np.zeros((2*Nj-1,1))
    pp = np.zeros((2*Nj-1,1))
    lambda_l = C_ip1[-1]
    lambda_u = C_ip1[0]
    pmp[-1] = p_m+p_d
    pp[-1] = C_ip1[-2]+p_d*lambda_l
    C_i = np.zeros((2*Nj+1,1))
    for i in range(2*Nj-2):
        pmp[2*Nj-3-i] = p_m-p_u/pmp[2*Nj-3-i+1]*p_d
        pp[2*Nj-3-i] = C_ip1[2*Nj-2-i]-pp[2*Nj-3-i+1]/pmp[2*Nj-3-i+1]*p_d
    C_i[0] = lambda_u+(pp[0]-lambda_u*p_u)/(pmp[0]+p_u)
    for i in range(2*Nj-1):
        C_i[i+1] = (pp[i]-p_u*C_i[i])/pmp[i]
    C_i[-1] = C_i[-2]-lambda_l
    return C_i, pmp, pp


class implicit_fd_grid(tree):
    def __init__(self,T,S,r,sigma,N,payoff,dx,D,Nj):
        tree.__init__(self,T,S,r,sigma,N,payoff,D)
        self.Nj = Nj
        self.dx = dx
    def build_tree(self):
        self.delta_t = self.T/self.N
        self.nu = self.r-self.D-0.5*self.sigma**2
        self.p_u = -0.5*self.delta_t*((self.sigma/self.dx)**2+self.nu/self.dx)
        self.p_m = 1+self.delta_t*(self.sigma/self.dx)**2+self.r*self.delta_t
        self.p_d = -0.5*self.delta_t*((self.sigma/self.dx)**2-self.nu/self.dx)
        
        self.St = self.S*np.exp(np.asarray([self.Nj*self.dx-i*self.dx for i in range(2*self.Nj+1)]))
        self.C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
    def euro_discount(self):
        self.build_tree()
        N = self.N
        while (N>0):
            if self.payoff.getidentity() == "callpayoff":
                self.lambda_u = np.asarray([self.St[0] - self.St[1]])
                self.lambda_l = np.asarray([0])
            else:
                self.lambda_u = np.asarray([0])
                self.lambda_l = np.asarray([(self.St[-1] - self.St[-2])])
            self.C_ip1 = np.concatenate(([self.lambda_u],self.C[1:-1],[self.lambda_l]))
            self.dis_C, self.pmp, self.pp= solve(self.p_u,self.p_m,self.p_d,self.C_ip1,self.Nj)
            self.C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])# apply the condition on node
            N -= 1
        return self.C[self.Nj]
    def amer_discount(self):
        self.build_tree()
        N = self.N
        while (N>0):
            #build C
            if self.payoff.getidentity() == "callpayoff":
                self.lambda_u = np.asarray([self.St[0] - self.St[1]])
                self.lambda_l = np.asarray([0])
            else:
                self.lambda_u = np.asarray([0])
                self.lambda_l = np.asarray([(self.St[-1] - self.St[-2])])
            self.C = np.concatenate(([self.lambda_u],self.C[1:-1],[self.lambda_l]))            
            #solve equation
            self.C_ip1 = np.concatenate(([self.lambda_u],self.C[1:-1],[self.lambda_l]))
            self.dis_C, self.pmp, self.pp= solve(self.p_u,self.p_m,self.p_d,self.C_ip1,self.Nj)
            self.dis_C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])
            self.exc_C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
            self.C = np.where(self.dis_C < self.exc_C, self.exc_C, self.dis_C) # do the comparation
            N -= 1
        return self.C[self.Nj]

implicit_fd_grid(T=1,S=100,r=0.06,sigma=0.2,N=3,payoff = putpayoff(100),dx=0.2,D=0.03,Nj=3).amer_discount()
grid = implicit_fd_grid(T=1,S=100,r=0.06,sigma=0.2,N=3,payoff = putpayoff(100),dx=0.2,D=0.03,Nj=3)
grid.euro_discount()

class cn_fd_grid(tree):
    def __init__(self,T,S,r,sigma,N,payoff,dx,D,Nj):
        tree.__init__(self,T,S,r,sigma,N,payoff,D)
        self.Nj = Nj
        self.dx = dx
    def build_tree(self):
        self.delta_t = self.T/self.N
        self.nu = self.r-self.D-0.5*self.sigma**2
        self.p_u = -0.25*self.delta_t*((self.sigma/self.dx)**2+self.nu/self.dx)
        self.p_m = 1+self.delta_t/2*(self.sigma/self.dx)**2+self.r*self.delta_t/2
        self.p_d = -0.25*self.delta_t*((self.sigma/self.dx)**2-self.nu/self.dx)     
        self.St = self.S*np.exp(np.asarray([self.Nj*self.dx-i*self.dx for i in range(2*self.Nj+1)]))
        self.C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
    def euro_discount(self):
        self.build_tree()
        N = self.N
        while (N>0):
            # boundary conditions
            if self.payoff.getidentity() == "callpayoff":
                lambda_u = np.asarray([self.St[0] - self.St[1]])
                lambda_l = np.asarray([0])
            else:
                lambda_u = np.asarray([0])
                lambda_l = np.asarray([(self.St[-1] - self.St[-2])])
            # build C
            self.C_ip1 = -self.p_u*self.C[:-2]-(self.p_m-2)*self.C[1:-1]-self.p_d*self.C[2:]
            self.C_ip1 = np.concatenate(([lambda_u],self.C_ip1,[lambda_l]))
            self.dis_C, self.pmp, self.pp= solve(self.p_u,self.p_m,self.p_d,self.C_ip1,self.Nj)
            self.C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])# apply the condition on node
            N -= 1
        return self.C[self.Nj]
    def amer_discount(self):
        self.build_tree()
        N = self.N
        while (N>0):
            #build C
            if self.payoff.getidentity() == "callpayoff":
                lambda_u = np.asarray([self.St[0] - self.St[1]])
                lambda_l = np.asarray([0])
            else:
                lambda_u = np.asarray([0])
                lambda_l = np.asarray([(self.St[-1] - self.St[-2])])
            # build C
            self.C_ip1 = -self.p_u*self.C[:-2]-(self.p_m-2)*self.C[1:-1]-self.p_d*self.C[2:]
            self.C_ip1 = np.concatenate(([lambda_u],self.C_ip1,[lambda_l]))
            #solve equation
            self.dis_C, self.pmp, self.pp= solve(self.p_u,self.p_m,self.p_d,self.C_ip1,self.Nj)
            self.dis_C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])
            self.exc_C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
            self.C = np.where(self.dis_C < self.exc_C, self.exc_C, self.dis_C) # do the comparation
            N -= 1
        return self.C[self.Nj]

grid = cn_fd_grid(T=1,S=100,r=0.06,sigma=0.2,N=3,payoff = putpayoff(100),dx=0.2,D=0.03,Nj=3)
grid.amer_discount()

e = 0.001
S0 = 100
K = 100
t = 1
sig = 0.25
r = 0.06
div = 0.03
d_t = e/(1+3*sig**2)
d_x = sig*m.sqrt(3*d_t)
n = m.ceil((3*0.25**2+1)/e)
nj = m.ceil(3*m.sqrt(n/3)-0.5)
excall = explicit_fd_grid(T=1,S=100,r=0.06,sigma=0.25,N=n,payoff = callpayoff(100),dx=d_x,D=0.03,Nj=nj).euro_discount()
exput = explicit_fd_grid(T=1,S=100,r=0.06,sigma=0.25,N=n,payoff = putpayoff(100),dx=d_x,D=0.03,Nj=nj).euro_discount()
imcall = implicit_fd_grid(T=1,S=100,r=0.06,sigma=0.25,N=n,payoff = callpayoff(100),dx=d_x,D=0.03,Nj=nj).euro_discount()
imput = implicit_fd_grid(T=1,S=100,r=0.06,sigma=0.25,N=n,payoff = putpayoff(100),dx=d_x,D=0.03,Nj=nj).euro_discount()

# =============================================================================
# n2 = m.ceil(T/((-3*0.25**2+m.sqrt(9*0.25**4+e))/0.5))
# nj2 = m.ceil(3*m.sqrt(n/3)-0.5)
# =============================================================================
cncall = cn_fd_grid(T=1,S=100,r=0.06,sigma=0.25,N=n,payoff = callpayoff(100),dx=d_x,D=0.03,Nj=nj).euro_discount()
cnput = cn_fd_grid(T=1,S=100,r=0.06,sigma=0.25,N=n,payoff = putpayoff(100),dx=d_x,D=0.03,Nj=nj).euro_discount()
price_result3 = pd.DataFrame([[excall,imcall,cncall],
                              [exput,imput,cnput]],index = ["Call","Put"],columns = ["Explicit FD","Implicit FD","Crank-Nicolson"])
print(price_result3)


bscall = BS_Formula(type_opt="call", r=0.06, vol=0.25, K=100, S=100, T=1, q=0.03)
bsput = BS_Formula(type_opt="put", r=0.06, vol=0.25, K=100, S=100, T=1, q=0.03)
grid1 = explicit_fd_grid(T=t,S=100,r=0.06,sigma=sig,N=n,payoff = callpayoff(100),dx=d_x,D=0.03,Nj=nj)
grid2 = explicit_fd_grid(T=t,S=100,r=0.06,sigma=sig,N=n,payoff = putpayoff(100),dx=d_x,D=0.03,Nj=nj)
grid3 = implicit_fd_grid(T=t,S=100,r=0.06,sigma=sig,N=n,payoff = callpayoff(100),dx=d_x,D=0.03,Nj=nj)
grid4 = implicit_fd_grid(T=t,S=100,r=0.06,sigma=sig,N=n,payoff = putpayoff(100),dx=d_x,D=0.03,Nj=nj)
grid5 = cn_fd_grid(T=t,S=100,r=0.06,sigma=sig,N=n,payoff = callpayoff(100),dx=d_x,D=0.03,Nj=nj)
grid6 = cn_fd_grid(T=t,S=100,r=0.06,sigma=sig,N=n,payoff = putpayoff(100),dx=d_x,D=0.03,Nj=nj)

def get_step(b_step,sig, t,nsd,error,grid,type_opt):
    if type_opt =="call":
        bs = BS_Formula(type_opt="call", r=0.06, vol=sig, K=100, S=100, T=t, q=0.03)
    else:
        bs = BS_Formula(type_opt="put", r=0.06, vol=sig, K=100, S=100, T=t, q=0.03)    
    n = b_step
    nj = int(np.ceil((np.sqrt(n)*nsd/np.sqrt(3)-1)/2))
    d_x = nsd*sig*m.sqrt(t)/(2*nj+1)
    grid.N = n
    grid.Nj = nj
    grid.dx = d_x
    grid.T = t
    while (abs(grid.euro_discount()[0]-bs)>error):
        n = n+300
        nj = int((np.sqrt(n)*nsd/np.sqrt(3)-1)/2)
        d_x = nsd*sig*m.sqrt(t)/(2*nj+1)
        grid.N = n
        grid.Nj = nj
        grid.dx = d_x
        #print(abs(grid.euro_discount()[0]-bs))
    return grid.euro_discount(),n,nj,d_x,t/n
result1 = get_step(b_step=10,sig=0.25, t=1,nsd=6,error=0.001,grid = grid1, type_opt="call")
result2 = get_step(b_step=10,sig=0.25, t=1,nsd=6,error=0.001,grid = grid2, type_opt="put")
result3 = get_step(b_step=10,sig=0.25, t=1,nsd=6,error=0.001,grid = grid3, type_opt="call")
result4 = get_step(b_step=10,sig=0.25, t=1,nsd=6,error=0.001,grid = grid4, type_opt="put")
result5 = get_step(b_step=10,sig=0.25, t=1,nsd=6,error=0.001,grid = grid5, type_opt="call")
result6 = get_step(b_step=10,sig=0.25, t=1,nsd=6,error=0.001,grid = grid6, type_opt="put")
step_result = pd.DataFrame([[result1,result3,result5],
                            [result2,result4,result6]],index = ["Call","Put"],columns = ["Explicit FD","Implicit FD","Crank-Nicolson"])
# =============================================================================
# type_opt = "call"
# if type_opt =="call":
#     bs = BS_Formula(type_opt="call", r=0.06, vol=sig, K=100, S=100, T=t, q=0.03)
# else:
#     bs = BS_Formula(type_opt="put", r=0.06, vol=sig, K=100, S=100, T=t, q=0.03)    
# n = 10
# nj = int(np.ceil((np.sqrt(n)*6/np.sqrt(3)-1)/2))
# d_x = 6*sig*m.sqrt(t)
# grid1.N = n
# grid1.Nj = nj
# grid1.dx = d_x
# while (abs(grid1.euro_discount()[0]-bs)>0.001):
#     n = n+1
#     nj = int((np.sqrt(n)*6/np.sqrt(3)-1)/2)
#     d_x = 6*sig*m.sqrt(t)
#     grid1.N = n
#     grid1.Nj = nj
#     grid1.dx = d_x
# 
# =============================================================================

class ran_fd_grid(tree):
    def __init__(self,T,S,r,sigma,N,payoff,dx,D,Nj):
        tree.__init__(self,T,S,r,sigma,N,payoff,D)
        self.Nj = Nj
        self.dx = dx
    def build_tree(self):
        self.delta_t = self.T/self.N
        self.nu = self.r-self.D-0.5*self.sigma**2
        self.p_u = -0.25*self.delta_t*((self.sigma/self.dx)**2+self.nu/self.dx)
        self.p_m = 1+self.delta_t/2*(self.sigma/self.dx)**2+self.r*self.delta_t/2
        self.p_d = -0.25*self.delta_t*((self.sigma/self.dx)**2-self.nu/self.dx)     
        self.St = self.S*np.exp(np.asarray([self.Nj*self.dx-i*self.dx for i in range(2*self.Nj+1)]))
        self.C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
    def euro_discount(self):
        self.build_tree()
        self.add_grid = implicit_fd_grid(T = self.delta_t ,S=self.S,r=self.r,sigma=self.sigma,N=4,payoff=self.payoff,dx=self.dx,D=self.D,Nj=self.Nj)
        self.add_grid.euro_discount()
        self.C = self.add_grid.C
        N = self.N-1
        while (N>0):
            # boundary conditions
            if self.payoff.getidentity() == "callpayoff":
                lambda_u = np.asarray([self.St[0] - self.St[1]])
                lambda_l = np.asarray([0])
            else:
                lambda_u = np.asarray([0])
                lambda_l = np.asarray([(self.St[-1] - self.St[-2])])
            # build C
            self.C_ip1 = -self.p_u*self.C[:-2]-(self.p_m-2)*self.C[1:-1]-self.p_d*self.C[2:]
            self.C_ip1 = np.concatenate(([lambda_u],self.C_ip1,[lambda_l]))
            self.dis_C, self.pmp, self.pp= solve(self.p_u,self.p_m,self.p_d,self.C_ip1,self.Nj)
            self.C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])# apply the condition on node
            N -= 1
        return self.C[self.Nj]
    def amer_discount(self):
        self.build_tree()
        self.add_grid = implicit_fd_grid(T = self.delta_t ,S=self.S,r=self.r,sigma=self.sigma,N=4,payoff=self.payoff,dx=self.dx,D=self.D,Nj=self.Nj)
        self.add_grid.amer_discount()
        self.C = self.add_grid.C
        N = self.N-1
        while (N>0):
            #build C
            if self.payoff.getidentity() == "callpayoff":
                lambda_u = np.asarray([self.St[0] - self.St[1]])
                lambda_l = np.asarray([0])
            else:
                lambda_u = np.asarray([0])
                lambda_l = np.asarray([(self.St[-1] - self.St[-2])])
            # build C
            self.C_ip1 = -self.p_u*self.C[:-2]-(self.p_m-2)*self.C[1:-1]-self.p_d*self.C[2:]
            self.C_ip1 = np.concatenate(([lambda_u],self.C_ip1,[lambda_l]))
            #solve equation
            self.dis_C, self.pmp, self.pp= solve(self.p_u,self.p_m,self.p_d,self.C_ip1,self.Nj)
            self.dis_C = np.asarray([self.payoff.getnodeprice(self.St[i],self.dis_C[i]) for i in range(len(self.St))])
            self.exc_C = np.asarray([self.payoff.getpayoff(p) for p in self.St])
            self.C = np.where(self.dis_C < self.exc_C, self.exc_C, self.dis_C) # do the comparation
            N -= 1
        return self.C[self.Nj]

grid3 = ran_fd_grid(T=1,S=100,r=0.06,sigma=0.2,N=3,payoff = putpayoff(100),dx=0.2,D=0.03,Nj=3)
grid3.amer_discount()














