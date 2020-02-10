# -*- coding: utf-8 -*-
from math import cos
from math import sin
import numpy as np
from numpy.linalg import norm

Eye = lambda N:np.matrix( np.eye(N) )
Zeros = lambda Ni,Nj:np.matrix(np.zeros((Ni,Nj)))

""" Matrizes de zeros e uns """
def ZerosOne_(i,n):
    mat = Zeros(n,1)
    mat[i] = 1.0
    return mat

def Qh_Qo__(lista,n):
    Qh_ = np.hstack([ ZerosOne_(i,n) for i in lista ])
    lista2 = []
    flag = True
    for i in range(n):
        for j in lista:
            if i==j:
                flag = False
        if flag==True:
            lista2.append(i)
        flag = True
    Qo_ = np.hstack([ ZerosOne_(i,n) for i in lista2 ])
    return [Qh_, Qo_]
    
""" Matrizes de produto vetorial """
S_ = lambda v_: np.matrix([
                          [0       , -v_[2,0],  v_[1,0]],
                          [ v_[2,0],  0      , -v_[0,0]],
                          [-v_[1,0], v_[0,0] ,  0     ]
                          ])

S2_ = lambda v_: S_(v_)*S_(v_)

""" Quaternions """
qqx_ = lambda theta: np.matrix([[sin(theta/2)],[0],[0],[cos(theta/2)]])
qqy_ = lambda theta: np.matrix([[0],[sin(theta/2)],[0],[cos(theta/2)]])
qqz_ = lambda theta: np.matrix([[0],[0],[sin(theta/2)],[cos(theta/2)]])

qq2R_ = lambda qq_: np.matrix([
                              [1 - 2*qq_[1,0]**2 - 2*qq_[2,0]**2        , 2*(qq_[0,0]*qq_[1,0] - qq_[2,0]*qq_[3,0]), 2*(qq_[0,0]*qq_[2,0] + qq_[1,0]*qq_[3,0])],
                              [2*(qq_[0,0]*qq_[1,0] + qq_[2,0]*qq_[3,0]), 1 - 2*qq_[0,0]**2 - 2*qq_[2,0]**2        , 2*(qq_[1,0]*qq_[2,0] - qq_[0,0]*qq_[3,0])],
                              [2*(qq_[0,0]*qq_[2,0] - qq_[1,0]*qq_[3,0]), 2*(qq_[0,0]*qq_[3,0] + qq_[1,0]*qq_[2,0]), 1 - 2*qq_[0,0]**2 - 2*qq_[1,0]**2        ]
                              ])
                              
QqI_ = lambda qq_: np.matrix([
                             [ qq_[3,0], -qq_[2,0],  qq_[1,0], qq_[0,0]],
                             [ qq_[2,0],  qq_[3,0], -qq_[0,0], qq_[1,0]],
                             [-qq_[1,0],  qq_[0,0],  qq_[3,0], qq_[2,0]],
                             [-qq_[0,0], -qq_[1,0], -qq_[2,0], qq_[3,0]]
                             ])
                             
QqII_ = lambda qq_: np.matrix([
                             [ qq_[3,0],  qq_[2,0], -qq_[1,0], qq_[0,0]],
                             [-qq_[2,0],  qq_[3,0],  qq_[0,0], qq_[1,0]],
                             [ qq_[1,0], -qq_[0,0],  qq_[3,0], qq_[2,0]],
                             [-qq_[0,0], -qq_[1,0], -qq_[2,0], qq_[3,0]]
                             ])
                             
Cq_ = lambda qq_: 0.5*np.matrix([
                                [ qq_[3,0],  qq_[2,0], -qq_[1,0]],
                                [-qq_[2,0],  qq_[3,0],  qq_[0,0]],
                                [ qq_[1,0], -qq_[0,0],  qq_[3,0]],
                                [-qq_[0,0], -qq_[1,0], -qq_[2,0]]
                                ])         
                             
def quat_prod(qq1_, qq2_):
    qq3_ = QqI_(qq1_)*qq2_
    return qq3_/norm(qq3_)
                              
""" Denavit-Hartenberg """
                              

DH2Hr_ = lambda a,alpha,d,theta: np.matrix([
                                       [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                                       [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                                       [0         ,  sin(alpha)           ,  cos(alpha)           , d           ],
                                       [0         ,  0                    ,  0                    , 1           ]
                                       ])
                                       
DH2Rl_ = lambda a,alpha,d,theta: np.matrix([
                                       [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha)],
                                       [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha)],
                                       [0         ,  sin(alpha)           ,  cos(alpha)           ]
                                       ])
                                       
DH2ol_ = lambda a,alpha,d,theta: np.matrix([
                                       [a*cos(theta)],
                                       [a*sin(theta)],
                                       [d           ]
                                       ])
                                       
DH2rql_ = lambda a,alpha,d,theta: np.matrix([
                                       [cos(theta/2)*sin(alpha/2)],
                                       [sin(theta/2)*sin(alpha/2)],
                                       [sin(theta/2)*cos(alpha/2)],
                                       [cos(theta/2)*cos(alpha/2)]
                                       ])
                                       
DH2rl_ol__ = lambda a,alpha,d,theta: [DH2rql_(a,alpha,d,theta), DH2ol_(a,alpha,d,theta)]

""" Outros """
vec2h_ = lambda v_: np.vstack((v_, Eye(1)))

EulerAngles2R_ = lambda a,b,c: np.matrix([
                                       [cos(a)*cos(c) - sin(a)*cos(b)*sin(c), -cos(a)*sin(c) - sin(a)*cos(b)*cos(c),  sin(a)*sin(b)],
                                       [sin(a)*cos(c) + cos(a)*cos(b)*sin(c),  cos(a)*cos(b)*cos(c) - sin(a)*sin(c), -cos(a)*sin(b)],
                                       [sin(b)*sin(c)                       ,  sin(b)*cos(c)                       ,  cos(b)       ]
                                       ])