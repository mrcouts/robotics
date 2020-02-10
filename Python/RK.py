#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import solve
from numpy.linalg import norm
from Utilities import *
from Serial import *
from Paralelo import *

class RK(object):
    def __init__(self, method='RK6'):
        if method == 'Heun':
            self.a = [[1.0]]
            self.b = [0.5, 0.5]
        elif method == 'MiddlePoint':
            self.a = [[0.5]]
            self.b = [0.0, 1.0]
        elif method == 'RK3':
            self.a = [[0.5],
                      [-1.0, 2.0]]
            self.b = [1.0/6, 2.0/3, 1.0/6]
        elif method=='RK4':
            self.a=[[0.5],
                    [0,0.5],
                    [0,0,1.0]]
            self.b=[1.0/6, 1.0/3, 1.0/3, 1.0/6]
        elif method=='RK438':
            self.a=[[1.0/3],
                    [-1.0/3,1.0],
                    [1.0,-1.0,1.0]]
            self.b=[1.0/8, 3.0/8, 3.0/8, 1.0/8]
        elif method=='RK4Master':
            self.a=[[0.4],
                    [0.29697761 ,0.15875964],
                    [0.21810040,-3.05096516,3.83286476]]
            self.b=[0.17476028, -0.55148066, 1.20553560, 0.17118478]
        elif method=='RK6':
            self.a=[[0.5],
                    [2.0/9, 4.0/9],
                    [7.0/36, 2.0/9, -1.0/12],
                    [-35.0/144, -55.0/36, 35.0/48, 15.0/8],
                    [-1.0/360, -11.0/36, -1.0/8, 0.5, 1.0/10],
                    [-41.0/260, 22.0/13, 43.0/156, -118.0/39, 32.0/195, 80.0/39]]
            self.b=[13.0/200, 0, 11.0/40, 11.0/40, 4.0/25, 4.0/25, 13.0/200]
        elif method=='RK7':
            b8 = 77.0/1440
            self.a=[[1.0/6],
                    [0.0, 1.0/3],
                    [1.0/8, 0.0, 3.0/8],
                    [148.0/1331, 0.0, 150.0/1331, -56.0/1331],
                    [-404.0/243, 0.0, -170.0/27, 4024.0/1701, 10648.0/1701],
                    [2466.0/2401, 0.0, 1242.0/343, -19176.0/16807, -51909.0/16807, 1053.0/2401],
                    [1.0/(576*b8), 0.0, 0.0, 1.0/(105*b8), -1331.0/(279552*b8), -9.0/(1024*b8), 343.0/(149760*b8)],
                    [-71.0/32 - 270.0*b8/11, 0.0, -195.0/22, 32.0/7, 29403.0/3584, -729.0/512, 1029.0/1408, 270.0*b8/11]]
            self.b=[77.0/1440 - b8, 0, 0, 32.0/105, 1771561.0/6289920, 243.0/2560, 16807.0/74880, b8, 11.0/270]
        else:
            raise ValueError('Este metodo nao esta disponivel')
        self.c=[sum(x) for x in self.a]
        self.n=len(self.b)
     
    def Apply(self,h,tf,y0_,f_):
        nt=int(tf/h)
        t_ = [i*h for i in xrange(nt+1)]
        y__= [y0_ for i in xrange(nt+1)]
        k__= [y0_ for i in xrange(self.n)]         
        
        for i in xrange(nt):
            k__[0] = f_(t_[i],y__[i])
            for j in xrange(1, self.n):
                k__[j] = f_(t_[i] + self.c[j-1]*h, y__[i] + h*sum([self.a[j-1][k]*k__[k] for k in range(j)])) #dot(self.a[j-1], k__[0:j]))
            y__[i+1] = y__[i] + h*sum([self.b[k]*k__[k] for k in range(self.n) ]) #dot(self.b, k__)
            #print(i)
        return y__, t_
                
    def Apply2(self,h,tf,y0_,f_):
        nt=int(tf/h)
        t_ = [i*h for i in xrange(nt+1)]
        y__= [y0_ for i in xrange(nt+1)]
        k__= [y0_ for i in xrange(self.n)]
        u__= [None for i in xrange(nt+1)]         
        
        for i in xrange(nt):
            k__[0], u__[i] = f_(t_[i], y__[i])
            for j in xrange(1, self.n):
                k__[j] = f_(t_[i] + self.c[j-1]*h, y__[i] + h*sum([self.a[j-1][k]*k__[k] for k in range(j)]))
            y__[i+1] = y__[i] + h*sum([self.b[k]*k__[k] for k in range(self.n) ])
            #print(i)
        u__[nt] = f_(t_[nt], y__[nt])[1]
        return y__, u__, t_
    
    def ApplyParallelInvDyn(self,h,tf,y0_, ParallelRobot, lamb=100):
        ParallelRobot.lamb = lamb
        f_ = ParallelRobot.fdqo_
        nt=int(tf/h)
        t_ = [i*h for i in xrange(nt+1)]
        y__= [y0_ for i in xrange(nt+1)]
        k__= [y0_ for i in xrange(self.n)]

        ParallelRobot.t_ = t_
        ParallelRobot.qbarnorm_ = [0 for i in xrange(nt+1)]
        ParallelRobot.q__= [Zeros(ParallelRobot.nq,1) for i in xrange(nt+1)]
        ParallelRobot.dq__= [Zeros(ParallelRobot.nq,1) for i in xrange(nt+1)]
        ParallelRobot.d2q__= [Zeros(ParallelRobot.nq,1) for i in xrange(nt+1)]
        ParallelRobot.u__= [Zeros(ParallelRobot.na,1) for i in xrange(nt+1)]
        
        for i in xrange(nt):
            k__[0] = f_(t_[i],y__[i])
            ParallelRobot.q__[i] = ParallelRobot.q_
            ParallelRobot.dq__[i] = ParallelRobot.C_*ParallelRobot.dqhr_(t_[i])
            ParallelRobot.doit_dq_(ParallelRobot.dq__[i])
            ParallelRobot.d2q__[i] = ParallelRobot.C_*ParallelRobot.d2qhr_(t_[i]) + ParallelRobot.c_
            ParallelRobot.u__[i] = solve(ParallelRobot.ZT_, ParallelRobot.Mh_*ParallelRobot.d2qhr_(t_[i]) + ParallelRobot.vh_ + ParallelRobot.gh_)
            ParallelRobot.qbarnorm_[i] = norm(ParallelRobot.qbar_, np.inf)
            for j in xrange(1, self.n):
                k__[j] = f_(t_[i] + self.c[j-1]*h, y__[i] + h*sum([self.a[j-1][k]*k__[k] for k in range(j)])) #dot(self.a[j-1], k__[0:j]))
            y__[i+1] = y__[i] + h*sum([self.b[k]*k__[k] for k in range(self.n) ]) #dot(self.b, k__)
            #print(i)
        f_(t_[nt],y__[nt])
        ParallelRobot.q__[nt] = ParallelRobot.q_
        ParallelRobot.dq__[nt] = ParallelRobot.C_*ParallelRobot.dqhr_(t_[nt])
        ParallelRobot.doit_dq_(ParallelRobot.dq__[nt])
        ParallelRobot.d2q__[nt] = ParallelRobot.C_*ParallelRobot.d2qhr_(t_[nt]) + ParallelRobot.c_
        ParallelRobot.u__[nt] = solve(ParallelRobot.ZT_, ParallelRobot.Mh_*ParallelRobot.d2qhr_(t_[nt]) + ParallelRobot.vh_ + ParallelRobot.gh_)
        ParallelRobot.qbarnorm_[nt] = norm(ParallelRobot.qbar_, np.inf)
        return y__, t_