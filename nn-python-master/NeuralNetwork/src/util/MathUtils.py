'''
Created on 5 de dez de 2017

@author: marcelovca90
'''
import numpy as np
import random as rnd

class MathUtils:
    
    def __init__(self):
        pass
    
def step(u):
    return 1 if u >= 0 else 0

def sign(u):
    return np.sign(u)

def ramp(u, a = 1):
    if (-a <= u and u <= a):
        return u;
    return a if u > a else -a

def logistic(u, beta = 1):
    return 1 / ( 1 + np.exp(-beta * u))

def tanh(u):
    return np.tanh(u)

def tanh_d(u):
    return 1 - tanh(u)**2

def gauss(u, c = 1, sd = 0.5):
    return np.exp(-((u - c)**2) / (2 * sd**2))

def linear(u):
    return u

def eqm(w, x, d):
    eqm = 0
    for i in range(0,len(x)):
        v = np.dot(np.transpose(w), x[i])
        eqm = eqm + pow(d[i] - v, 2)
    eqm = eqm / len(x)
    return eqm
