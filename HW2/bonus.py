# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:56:31 2019

@author: Qiao Guanzhuo
"""
import pandas as pd
from pandas_datareader import data as pdrd
import pandas_datareader.yahoo.options as pdro
import math as m
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
import time
import os

def phi(x):
    if x<1 and x>-1:
        return 1-abs(x)
    else:
        return 0
def phin(x,n):
    return n**(1/3)*phi(x*n**(1/3))

def mutation(x,y,h,M,alpha,nu,mu,f,sig):
    dt = h/M
    count = M
    yp = y
    xp = x
    while (count>0):
        xp = xp+dt*(mu-sig(yp)**2/2)+np.sqrt(dt)*sig(yp)*np.random.normal(0,1,1,)[0]
        yp = yp+dt*alpha*(nu-yp)+np.sqrt(dt)*f(yp)*np.random.normal(0,1,1,)[0]
        count = count-1
    return xp,yp

def selection(Xp,Yp,func,xr,n):
    C = sum([func(xp-xr,n) for xp in Xp])
    prob = np.asarray([0]+[func(xp-xr,n)/C for xp in Xp])
    cumprob = prob.cumsum()
    rand = np.random.uniform(0,1,n)
    sample = []
    for num in rand:
        for l in range(len(cumprob)-1):
            if num> cumprob[l] and num<= cumprob[l+1]:
                index = l
        sample.append(Yp[index])
    return sample,prob[1:]

def vol_dis(X,y0,h,M,alpha,nu,mu,f,sig,n):
    #step1
    count = n
    Xp =[]
    Yp = []
    while (count>0):
        xp,yp = mutation(X[0],y0,h,M,alpha,nu,mu,f,sig)
        Xp.append(xp)
        Yp.append(yp)
        count = count-1
    sample,sample_prob = selection(Xp,Yp,phin,X[1],n)
    # step2
    for index in range(2,len(X)):
        Xp =[]
        Yp = []
        for y in sample:
            xp,yp = mutation(X[index-1],y,h,M,alpha,nu,mu,f,sig)
            Xp.append(xp)
            Yp.append(yp)
        sample,sample_prob = selection(Xp,Yp,phin,X[index],n)
    return Yp,sample_prob

def bigphi(x):
    return x
def sigfunc(x):
    return x

os.chdir('D:\\Grad 2\\621\\assignment\\HM1\\data') 
data = pd.read_csv('combined equity data.csv')
X = data["Close_amzn"]
X = np.log(X/X.shift(1))[1:]
X = list(X)
Y,P = vol_dis(X,0.25,0.1,40,0.2,0.2,0.5,bigphi,sigfunc,100)
plt.bar(Y, P)
def successor(x,L,Yi,dt,sig,p,r):
    vol = sig(Yi)*np.sqrt(dt)
    j = -L
    while (j*vol<x):
        j = j+1
    x1 = (j+1)*vol+(r-sig(Yi)**2/2)*dt
    x2 = j*vol+(r-sig(Yi)**2/2)*dt
    x3 = (j-1)*vol+(r-sig(Yi)**2/2)*dt
    x4 = (j-2)*vol+(r-sig(Yi)**2/2)*dt
    
    if x2-x<=x-x3:
        q = (x-x2)/vol
        p1 = (1+q+q**2)/2-p
        p2 = 3*p-q**2
        p3 = (1-q+q**2)/2-3*p
        p4 = p
    else:
        q = (x-x3)/vol
        p1 = p
        p2 = (1-q+q**2)/2-3*p
        p3 = 3*p-q**2
        p4 = (1+q+q**2)/2-p
    return [[x1,x2,x3,x4],[p1,p2,p3,p4]]
successor(x=X[-1],L=100,Yi=Y[0],dt=0.01,sig=sigfunc,p=0.1,r=0.06)
class multi_tree():
    def __init__(self,T,S,r,Yi,N,L,x0,K):
        self.T = T
        self.S = S
        self.r = r
        self.Yi = Yi
        self.N = N
        self.x0 = x0
        self.K = K
        self.L = L
    def build_tree(self):
        self.delta_t = self.T/self.N
        self.x_new = [self.x0]
        self.prob = []
        for i in range(self.N):
            self.x = self.x_new
            self.x_new = []
            for ele in self.x:
                p  = np.random.uniform(1/12,1/6)
                info = successor(ele,self.L,self.Yi[i],self.delta_t,sigfunc,p,self.r)
                for xx in  info[0]:
                    self.x_new.append(xx)
                self.prob.append(info[1])
        self.x_new = np.asarray(self.x_new)
        self.St = self.S*np.exp(self.x_new)
        self.disc = np.exp(-self.r*self.delta_t)
        self.C = np.where(self.St>=self.K,self.St-self.K,0)
    def euro_discount(self):
        self.build_tree()
        for mm in range(self.N,0,-1):
            self.C = self.C.reshape(4**(mm-1),4)
            self.pp = self.prob[]
            self.C = self.disc*(self.pp[0]*self.C[:-3]+self.pp[1]*self.C[1:-2]+self.pp[2]*self.C[2:-1]+self.pp[3]*self.C[3:])# compute discounted value of product
        return self.C[0]
    def amer_discount(self):
        pass 

tr = multi_tree(T=1,S=100,r=0.06,Yi=[0.5]*200,N =200,L = 10,x0 = 0.1,K=100)
tr.euro_discount()

class multi_tree():
    def __init__(self,T,S,r,Yi,N,L,x0,K):
        self.T = T
        self.S = S
        self.r = r
        self.Yi = Yi
        self.N = N
        self.x0 = x0
        self.K = K
        self.L = L
    def build_tree(self):
        self.delta_t = self.T/self.N
        self.x = self.x0
        self.last_vol = self.Yi[-1]*np.sqrt(self.delta_t)
        self.prob = []
        for i in range(self.N):
            self.nu = self.r-0.5*self.Yi[i]**2
            p  = np.random.uniform(1/12,1/6)
            info = successor(self.x,self.L,self.Yi[i],self.delta_t,sigfunc,p,self.r)
            self.x = info[0][0]
            self.prob.append(info[1])
        self.St = self.S*np.exp(np.asarray([self.x-i*self.last_vol for i in range(3*self.N+1)]))
        self.disc = np.exp(-self.r*self.delta_t)
        self.C = np.where(self.St>=self.K,self.St-self.K,0)
    def euro_discount(self):
        self.build_tree()
        for mm in range(self.N):
            self.pp = self.prob[self.N-1-mm]
            self.C = self.disc*(self.pp[0]*self.C[:-3]+self.pp[1]*self.C[1:-2]\
                                +self.pp[2]*self.C[2:-1]+self.pp[3]*self.C[3:])
        return self.C[0]
    def amer_discount(self):
        pass 

tr = multi_tree(T=1,S=100,r=0.06,Yi=[0.5]*200,N =200,L = 10,x0 = 0.1,K=100)











