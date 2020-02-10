# -*- coding: utf-8 -*-
from math import cos
from math import sin
import numpy as np

Eye = lambda N:np.matrix( np.eye(N) )
Zeros = lambda Ni,Nj:np.matrix(np.zeros((Ni,Nj)))

def ZerosOne(i,n):
    mat = Zeros(n,1)
    mat[i] = 1.0
    return mat

def Qh_Qo_(lista,n):
    Qh_ = np.hstack([ ZerosOne(i,n) for i in lista ])
    lista2 = []
    flag = True
    for i in range(n):
        for j in lista:
            if i==j:
                flag = False
        if flag==True:
            lista2.append(i)
        flag = True
    Qo_ = np.hstack([ ZerosOne(i,n) for i in lista2 ])
    return [Qh_, Qo_]

S_ = lambda v_: np.matrix([
                          [0       , -v_[2,0],  v_[1,0]],
                          [ v_[2,0],  0      , -v_[0,0]],
                          [-v_[1,0], v_[0,0] ,  0     ]
                          ])

S2_ = lambda v_: S_(v_)*S_(v_)
    
vec2h_ = lambda v_: np.vstack((v_, Eye(1)))

DH = lambda a,alpha,d,theta: np.matrix([
                                       [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                                       [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                                       [0         ,  sin(alpha)           ,  cos(alpha)           , d           ],
                                       [0         ,  0                    ,  0                    , 1           ]
                                       ])

EulerAngles_ = lambda a,b,c: np.matrix([
                                       [cos(a)*cos(c) - sin(a)*cos(b)*sin(c), -cos(a)*sin(c) - sin(a)*cos(b)*cos(c),  sin(a)*sin(b)],
                                       [sin(a)*cos(c) + cos(a)*cos(b)*sin(c),  cos(a)*cos(b)*cos(c) - sin(a)*sin(c), -cos(a)*sin(b)],
                                       [sin(b)*sin(c)                       ,  sin(b)*cos(c)                       ,  cos(b)       ]
                                       ])    
                                     
fDH_RR = lambda q_, l_, lg_: np.matrix([
                                       [l_[0], 0, 0, q_[0,0], -l_[0] + lg_[0], 0,  0, True],
                                       [l_[1], 0, 0, q_[1,0], -l_[1] + lg_[1], 0,  0, True]
                                       ])