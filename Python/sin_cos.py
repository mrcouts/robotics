# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from math import floor
from numpy import sign
from txt2py import *

A = txt2py("sin_cos.txt")

n = len(A)

x_np = np.array([A[i][0] for i in range(n)])
sin_np = np.array([A[i][1] for i in range(n)])
cos_np = np.array([A[i][2] for i in range(n)])

x0 = x_np[0]
xf = x_np[-1]
dx = x_np[1] - x_np[0]

def Sin(x):
    SIGN = sign(x)
    if x < 0:
        x = -x
    if(x > 2*pi):
        x = x - 2*pi*(x//(2*pi))
    indice = x/dx
    i1 = int(floor(indice))
    i2 = i1+1
    Sin1 = sin_np[i1]
    Sin2 = sin_np[i2]
    return SIGN*(Sin1 + (indice-i1)*(Sin2 - Sin1))
    
def Cos(x):
    if x < 0:
        x = -x
    if(x > 2*pi):
        x = x - 2*pi*(x//(2*pi))
    indice = x/dx
    i1 = int(floor(indice))
    i2 = i1+1
    Cos1 = cos_np[i1]
    Cos2 = cos_np[i2]
    return Cos1 + (indice-i1)*(Cos2 - Cos1)