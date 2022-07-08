#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 20:14:52 2022

@author: gc
"""

import numpy as np
import matplotlib.pyplot as plt

#%% noyau polynomial
c=1
def kern_pol(x,y):
    return (c+x.T@y)**3

#%% noyau gaussien

c=1
def kern_gauss(x,y):
    return np.exp(-(1/c**2)*np.linalg.norm(x-y)**2)

#%% noyau sigmoïde
c=0
gamma=0.2
def kern_sigmo(x,y):

     return np.tanh((c+gamma*x.T @ y)[0][0])

#%%

#%% noyau "rationnal quadratic kernel"
c=10

def kern_ratio_quad(x,y):
      return 1-np.linalg.norm(x-y)**2/(np.linalg.norm(x-y)**2+c)
 
#%%noyau multiquadratique
c=10

def kern_mquad(x,y):
      return np.sqrt(np.linalg.norm(x-y)**2+c**2)
    
#%% noyau inverse multiquadratique
c=1

def kern_inv_mquad(x,y):
      return 1/np.sqrt(np.linalg.norm(x-y)**2+c**2)
#%% chi noyau

def kern_chi(x,y):
    S=0
    for i in range(0,len(x)): 
        S=S+2*x[i]*y[i]/(x[i]+y[i])
    return S
 

#%% chi histogram intersection kernel

def kern_chi_hist(x,y):
    S=0
    t=np.zeros((2,1))

    for i in range(0,len(x)): 
        for j in range(0,len(y)): 
            S=S+min(x[i],y[j])
    return S
#%% noyau de Cauchy

c=5

def kern_cauchy(x,y):
      return 1/(1+(np.linalg.norm(x-y)**2)/c**2)

#%% log kernel (pour dimension paire)


c=2

def kern_log(x,y):
      return -np.log(np.linalg.norm(x-y)**2+1)


#%% spline kernel

def kern_spline(x,y):
    S=1
    for i in range(0,len(x)): 
        S=S*(1*x[i]*y[i]+x[i]*y[i]*min(x[i],y[i])-((x[i]+y[i])/2)*min(x[i],y[i])**2+(1/3)*min(x[i],y[i])**3)
 
    return S

