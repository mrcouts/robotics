#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from math import cos
from math import sin
from math import sqrt
from math import pi
import numpy as np
from scipy.linalg import block_diag
from numpy.linalg import solve
from numpy.linalg import norm
import matplotlib.pyplot as plt

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
    
class Serial:
    def __init__(self, dof, l_, lg_, m_, I__, gamma_, fDH):
        self.dof = dof
        self.l_ = l_
        self.lg_ = lg_
        self.m_ = m_
        self.Ir__ = I__
        self.gamma_ = Zeros(3,1)
        self.gamma_ = gamma_
        self.fDH = fDH
        
        self.q_ = Zeros(dof,1)
        self.dq_ = Zeros(dof,1)
        self.Hr__  = [ Zeros(4,4) for i in range(dof) ]
        self.H__  = [ Zeros(4,4) for i in range(dof) ]
        self.gr__  = [ Zeros(3,1) for i in range(dof) ]
        self.g__  = [ Zeros(3,1) for i in range(dof) ]
        self.z__  = [ Zeros(3,1) for i in range(dof) ]
        self.o__  = [ Zeros(3,1) for i in range(dof) ]
        self.jw__  = [ Zeros(3,1) for i in range(dof) ]
        self.jv__  = [ Zeros(3,1) for i in range(dof) ]
        self.Jw__  = [ Zeros(3,dof) for i in range(dof) ]
        self.Jv__  = [ Zeros(3,dof) for i in range(dof) ]
        self.wrel__ = [ Zeros(3,1) for i in range(dof) ]
        self.vrel__ = [ Zeros(3,1) for i in range(dof) ]
        self.w__ = [ Zeros(3,1) for i in range(dof) ]
        self.alphatil__ = [ Zeros(3,1) for i in range(dof) ]
        self.atil__ = [ Zeros(3,1) for i in range(dof) ]
        
        self.x_ = Zeros(3,1)
        self.Jw_ = Zeros(3,dof)
        self.Jv_ = Zeros(3,dof)
        self.alphatil_ = Zeros(3,1)
        self.atil_ = Zeros(3,1)
        
        self.I__ = [ Zeros(3,3) for i in range(dof) ]
        self.M_ = Zeros(dof,dof)
        self.v_ = Zeros(dof,1)
        self.g_ = Zeros(dof,1)
        
    def doit_q_(self, q_):
        self.q_ = q_
        fDH_mat = fDH_RR(self.q_, self.l_, self.lg_)
        
        for i in range(self.dof):
            self.gr__[i] = (fDH_mat[i,4:7]).T
            self.Hr__[i] = DH(*((fDH_mat[i,:4]).tolist())[0])
        
        self.H__[0] = self.Hr__[0]        
        for i in range(1,self.dof):
            self.H__[i] = self.H__[i-1]*self.Hr__[i]
            
        for i in range(self.dof):
            self.g__[i] = (self.H__[i]*vec2h_(self.gr__[i]))[:3,0]
            self.z__[i] = self.H__[i][:3,2]
            self.o__[i] = self.H__[i][:3,3]
            self.I__[i] = self.H__[i][:3,:3]*self.Ir__[i]*self.H__[i][:3,:3].T
            
        self.x_ = self.o__[dof-1]
            
        o0_ = Zeros(3,1)
        z0_  = ZerosOne(2,3)            
        self.jw__[0] = Zeros(3,1) if fDH_mat[0,7]==False else z0_
        self.jv__[0] = z0_ if fDH_mat[0,7]==False else S_(z0_)*(self.g__[0] - o0_)
        self.Jw__[0] = self.jw__[0]*ZerosOne(0,self.dof).T
        self.Jv__[0] = self.jv__[0]*ZerosOne(0,self.dof).T
        for i in range(1,self.dof):
            self.jw__[i] = Zeros(3,1) if fDH_mat[i,7]==False else self.z__[i-1]
            self.jv__[i] = self.z__[i-1] if fDH_mat[i,7]==False else S_(self.z__[i-1])*(self.g__[i] - self.o__[i-1])
            self.Jw__[i] = self.Jw__[i-1] + self.jw__[i]*ZerosOne(i,self.dof).T
            self.Jv__[i] = self.Jv__[i-1] - S_(self.g__[i] - self.g__[i-1])*self.Jw__[i-1]  + self.jv__[i]*ZerosOne(i,self.dof).T
            
        self.Jw_ = self.Jw__[dof-1]
        self.Jv_ = self.Jv__[dof-1] - S_(self.x_ - self.g__[dof-1])*self.Jw__[dof-1]
        
        self.M_ = Zeros(dof,dof)
        self.g_ = Zeros(dof,1)            
        for i in range(self.dof):
            self.M_ +=  self.m_[i]*self.Jv__[i].T*self.Jv__[i] + self.Jw__[i].T*self.I__[i]*self.Jw__[i]
            self.g_ += -self.m_[i]*self.Jv__[i].T*self.gamma_
            
    def doit_q_dq_(self, q_, dq_):
        self.doit_q_(q_)
        self.dq_ = dq_
        
        for i in range(self.dof):        
            self.wrel__[i] = self.dq_[i,0]*self.jw__[i]
            self.vrel__[i] = self.dq_[i,0]*self.jv__[i]
        
        self.w__[0] = self.wrel__[0]
        self.alphatil__[0] = Zeros(3,1)
        self.atil__[0] = S_(self.wrel__[0])*self.vrel__[0]
        for i in range(1,self.dof):
            self.w__[i] = self.w__[i-1] + self.wrel__[i]
            self.alphatil__[i] = self.alphatil__[i-1] + S_(self.w__[i-1])*self.wrel__[i]
            self.atil__[i] = self.atil__[i-1] + (S_(self.alphatil__[i-1]) + S2_(self.w__[i-1]))*(self.g__[i] - self.g__[i-1]) + (S_(self.wrel__[i]) + 2*S_(self.w__[i-1]))*self.vrel__[i]
            
        self.alphatil_ = self.alphatil__[dof-1]
        self.atil_ = self.atil__[dof-1] + (S_(self.alphatil__[dof-1]) + S2_(self.w__[dof-1]))*(self.x_ - self.g__[dof-1])
        
        self.v_ = Zeros(dof,1)
        for i in range(self.dof):
            self.v_ += self.m_[i]*self.Jv__[i].T*self.atil__[i] + self.Jw__[i].T*(self.I__[i]*self.alphatil__[i] + S_(self.w__[i])*self.I__[i]*self.w__[i] )
            
class EfetuadorTranslacional:
    def __init__(self, m, gamma_, a, b, c, Espacial = True):
        self.transdof = 3 if Espacial == True else 2
        self.dof = self.transdof
        self.q_ = Zeros(self.dof,1)
        self.dq_ = Zeros(self.dof,1)
        self.R_ = EulerAngles_(a,b,c)
        self.M_ = m*Eye(self.dof)
        self.v_ = Zeros(self.dof,1)
        self.g_ = -m*gamma_[0:self.dof,0]
        
        
    def doit_q_(self, q_):
        self.q_ = q_
        
    def doit_q_dq_(self, q_, dq_):
        self.doit_q_(q_)
        self.dq_ = dq_
        
class Paralelo:
    def __init__(self, dof, Serial_, EndEffector, angulos__, origens__, xb__, lista_ind, lista_actuator):
        self.dof = dof
        self.Serial_ = Serial_
        self.EndEffector = EndEffector
        self.angulos__ = angulos__
        self.origens__ = origens__
        self.xb__ = xb__
        self.n = len(Serial_)
        self.R__ = [ Zeros(3,3) for i in range(self.n) ]
        self.nq = EndEffector.dof + sum([Serial_[i].dof for i in range(self.n)])
        self.q_ = Zeros(self.nq,1)
        self.dq_ = Zeros(self.nq,1)
        self.x_ = Zeros(EndEffector.transdof*self.n,1)
        self.qbar_ = Zeros(EndEffector.transdof*self.n,1)
        self.Jv_ = Zeros(EndEffector.transdof*self.n,self.nq - EndEffector.dof)
        self.atil_ = Zeros(EndEffector.transdof*self.n,1)
        self.A_ = Zeros(EndEffector.transdof*self.n, self.nq)
        self.b_ = Zeros(EndEffector.transdof*self.n, 1)
        self.Qh_ = Qh_Qo_(lista_ind, self.nq)[0]
        self.Qo_ = Qh_Qo_(lista_ind, self.nq)[1]
        self.U_ = Qh_Qo_(lista_actuator, self.nq)[0]
        self.Ah_ = Zeros(EndEffector.transdof*self.n, dof)
        self.Ao_ = Zeros(EndEffector.transdof*self.n, self.nq - dof)
        self.C_ = Zeros(self.nq, dof)
        self.c_ = Zeros(self.nq, 1)
        self.M_ = Zeros(self.nq,self.nq)
        self.v_ = Zeros(self.nq,1)
        self.g_ = Zeros(self.nq,1)
        self.Mh_ = Zeros(dof,dof)
        self.vh_ = Zeros(dof,1)
        self.gh_ = Zeros(dof,1)
        self.na = np.size(self.U_,1)
        self.ZT_ = Zeros(self.na, dof)
        
        self.qhr = lambda t: Zeros(dof,1)
        self.dqhr = lambda t: Zeros(dof,1)
        self.d2qhr = lambda t: Zeros(dof,1)
        self.lamb = 100.0
        
        self.t_ = []
        self.q__ = []
        self.dq__ = []
        self.d2q__ = []
        self.u__ = []
        self.qbarnorm_ = []
        
        for i in range(self.n):
            self.R__[i] = EulerAngles_(*(angulos__[i].T.tolist()[0]))[0:EndEffector.transdof,:]
            
        self.D_ = np.vstack([ Eye(EndEffector.transdof) for i in range(self.n) ])
        self.E_ = np.matrix(block_diag(*self.R__))
        self.d_ = np.vstack([ (origens__[i] - EndEffector.R_*xb__[i])[0:EndEffector.transdof,:] for i in range(dof) ])
        
    def doit_q_(self, q_):
        self.q_ = q_
        soma1 = 0
        soma2 = soma1 + EndEffector.dof
        self.EndEffector.doit_q_(q_[soma1:soma2,0])
        for i in range(self.n):
            soma1 = soma2
            soma2 += self.Serial_[i].dof
            self.Serial_[i].doit_q_(q_[soma1:soma2,0])
            
        self.x_ = np.vstack([ self.Serial_[i].x_ for i in range(dof) ])
        self.qbar_ = self.D_*self.EndEffector.q_- self.d_ - self.E_*self.x_
        self.Jv_ = np.matrix(block_diag(*[self.Serial_[i].Jv_ for i in range(self.n)]))
        self.A_ = np.hstack([self.D_, -self.E_*self.Jv_])
        self.Ah_ = self.A_*self.Qh_
        self.Ao_ = self.A_*self.Qo_
        self.C_ = self.Qh_ - self.Qo_*solve(self.Ao_,self.Ah_)
        self.ZT_ = self.C_.T*self.U_
        self.M_ = np.matrix(block_diag(*([self.EndEffector.M_] + [self.Serial_[i].M_ for i in range(self.n)])))
        self.g_ = np.vstack([self.EndEffector.g_] + [ self.Serial_[i].g_ for i in range(dof) ])
        self.Mh_ = self.C_.T*self.M_*self.C_
        self.gh_ = self.C_.T*self.g_
        
    def doit_dq_(self, dq_):
        self.dq_ = dq_
        q_ = self.q_
        soma1 = 0
        soma2 = soma1 + EndEffector.dof
        self.EndEffector.doit_q_dq_(q_[soma1:soma2,0], q_[soma1:soma2,0])
        for i in range(self.n):
            soma1 = soma2
            soma2 += self.Serial_[i].dof
            self.Serial_[i].doit_q_dq_(q_[soma1:soma2,0], dq_[soma1:soma2,0])
        self.atil_ = np.vstack([ self.Serial_[i].atil_ for i in range(dof) ])
        self.b_ = -self.E_*self.atil_
        self.c_ = - self.Qo_*solve(self.Ao_,self.b_)
        self.v_ = np.vstack([self.EndEffector.v_] + [ self.Serial_[i].v_ for i in range(dof) ])
        self.vh_ = self.C_.T*(self.M_*self.c_ + self.v_)
        
    def doit_q_dq_(self, q_, dq_):
        self.doit_q_(q_)
        self.doit_dq_(dq_)
        
    def set_ref(self, qhr_, dqhr_, d2qhr_):
        self.qhr_ = qhr_
        self.dqhr_ = dqhr_
        self.d2qhr_ = d2qhr_
        
    def fdqo_(self, t, qo_):
        self.doit_q_( self.Qh_*self.qhr_(t) + self.Qo_*qo_ )
        return solve(-self.Ao_,(self.Ah_*self.dqhr_(t) + self.lamb*self.qbar_ ))
        
class GNR:
    def __init__(self, ParallelRobot, qo0_, qh_, nsteps, method='RK7'):
        self.ParallelRobot = ParallelRobot
        self.qo0_ = qo0_
        self.qh_ = qh_
        self.nsteps = nsteps
        self.rk = RK(method)
        self.ParallelRobot.doit_q_(self.ParallelRobot.Qh_*qh_ + self.ParallelRobot.Qo_*qo0_)
        self.qbar_ = self.ParallelRobot.qbar_

    def f_(self, t, qo_):
        self.ParallelRobot.doit_q_(self.ParallelRobot.Qh_*self.qh_ + self.ParallelRobot.Qo_*qo_)
        return solve(-self.ParallelRobot.Ao_, self.qbar_)
        
    def set_qh_(self,qh_):
        self.qh_ = qh_
        self.ParallelRobot.doit_q_(self.ParallelRobot.Qh_*qh_ + self.ParallelRobot.Qo_*self.qo0_)
        self.qbar_ = self.ParallelRobot.qbar_
        
    def set_qo0_(self,qo0_):
        self.qo0_ = qo0_
        self.ParallelRobot.doit_q_(self.ParallelRobot.Qh_*self.qh_ + self.ParallelRobot.Qo_*self.qo0_)
        self.qbar_ = self.ParallelRobot.qbar_

    def set_qh_qo0_(self, qh_, qo0_):
        self.qo0_ = qo0_
        self.qh_ = qh_
        self.ParallelRobot.doit_q_(self.ParallelRobot.Qh_*self.qh_ + self.ParallelRobot.Qo_*self.qo0_)
        self.qbar_ = self.ParallelRobot.qbar_            
        
    def doit(self):
        y__, t_ = self.rk.Apply(1.0/self.nsteps, 1.0, self.qo0_, self.f_)
        self.ParallelRobot.doit_q_(self.ParallelRobot.Qh_*self.qh_ + self.ParallelRobot.Qo_*y__[-1])
        
        
        
fDH_RR = lambda q_, l_, lg_: np.matrix([
                                       [l_[0], 0, 0, q_[0,0], -l_[0] + lg_[0], 0,  0, True],
                                       [l_[1], 0, 0, q_[1,0], -l_[1] + lg_[1], 0,  0, True]
                                       ])
 
dof = 2      
l1 = 0.12
l2 = 0.16
lg1 = 0.06
lg21 = 0.078
lg22 = 0.058
m1 = 0.068
m21 = 0.124
m22 = 0.097
Jz11 = 107.307e-6 + 146.869e-6
Jz21 = 438.0e-6
Jz12 = 107.307e-6 + 188.738e-6
Jz22 = 301.679e-6

l_ = [l1,l2]
lg1_ = [lg1, lg21]
lg2_ = [lg1, lg22]
m1_ = [m1, m21]
m2_ = [m1, m22]

I11_ = Zeros(3,3)
I11_[2,2] = Jz11
I21_ = Zeros(3,3)
I21_[2,2] = Jz21
I1__ = [I11_, I21_]

I12_ = Zeros(3,3)
I12_[2,2] = Jz12
I22_ = Zeros(3,3)
I22_[2,2] = Jz22
I2__ = [I12_, I22_]

gamma_ = Zeros(3,1)
gamma_[1] = -9.8

q_ = Zeros(dof,1)
q_[0,0] = pi/4
q_[1,0] = pi/2

dq_ = Zeros(dof,1)
dq_[0,0] = 5.0
dq_[1,0] = 10.0

#print DH(*((fDH_RR(q_, l_, lg_)[0,:4]).tolist())[0])
#print (fDH_RR(q_, l_, lg_)[0,4:7]).T
        
RR = Serial(dof, l_, lg1_, m1_, I1__, gamma_, fDH_RR)
RR1 = Serial(dof, l_, lg1_, m1_, I1__, gamma_, fDH_RR)
RR2 = Serial(dof, l_, lg2_, m2_, I2__, gamma_, fDH_RR)
RR_ = [RR1, RR2]
EndEffector = EfetuadorTranslacional(0, gamma_, 0, 0, 0, False)

angulos1_ = Zeros(3,1)
angulos2_ = Zeros(3,1)
angulos2_[0,0] = pi
angulos2_[1,0] = pi
angulos__ = [angulos1_, angulos2_]

l0 = 0.05
origem1_ = Zeros(3,1)
origem1_[0,0] = l0
origem2_ = -origem1_
origens__ = [origem1_, origem2_]

xb__ = [Zeros(3,1) for i in range(dof)]

Clara = Paralelo(dof, RR_, EndEffector, angulos__, origens__, xb__, [0,1], [2,4])
claraq_ = np.matrix([[0.0,  0.108,  0.5375391526183005, 2.308800210309811,  0.5375391526183005, 2.308800210309811]]).T


#RR.doit_q_dq_(q_, dq_)
#print RR.M_
#print RR.v_
#print RR.g_

#print m1*lg1**2 + Jz1 + m2*(l1**2 + lg2**2) + Jz2 + 2*m2*l1*lg2*cos(q_[1,0]) - RR.M_[0,0]
#print m2*lg2**2 + Jz2 + m2*l1*lg2*cos(q_[1,0]) - RR.M_[0,1]
#print m2*lg2**2 + Jz2 - RR.M_[1,1]
#print 9.8*((m1*lg1+ m2*l1)*cos(q_[0,0]) + m2*lg2*cos(q_[0,0]+q_[1,0])) - RR.g_[0,0]
#print 9.8*(m2*lg2*cos(q_[0,0]+q_[1,0])) - RR.g_[1,0]
#print -m2*l1*lg2*sin(q_[1,0])*(2*dq_[0,0]*dq_[1,0] + dq_[1,0]**2) - RR.v_[0,0]
#print m2*l1*lg2*sin(q_[1,0])*(dq_[0,0]**2) - RR.v_[1,0]

w = 2*pi
r = 0.05
qhr_ = lambda t: np.matrix([[ -r*sin(w*t), 0.158 -r*cos(w*t)]]).T
dqhr_ = lambda t: np.matrix([[ -w*r*cos(w*t),  w*r*sin(w*t)]]).T
d2qhr_ = lambda t: np.matrix([[ w*w*r*sin(w*t), w*w*r*cos(w*t)]]).T

Clara.set_ref(qhr_, dqhr_, d2qhr_)

Clara.doit_q_(claraq_)
dqo_ = Clara.fdqo_(0, Clara.Qo_.T*claraq_)
claradq_ = Clara.C_*np.matrix([[1.0, 2.0]]).T
Clara.doit_q_dq_(claraq_, claradq_)

rk = RK('RK7')
y__, t_ = rk.ApplyParallelInvDyn(0.001, 0.010, Clara.Qo_.T*claraq_, Clara, 10)

#print Clara.Mh_
#print Clara.vh_
#print Clara.gh_
#print dqo_

t_np = np.array([t_[i] for i in range(len(t_))])
qbarnorm_np = np.array([Clara.qbarnorm_[i] for i in range(len(t_))])

u1_np = np.array([Clara.u__[i][0,0] for i in range(len(t_))])
u2_np = np.array([Clara.u__[i][1,0] for i in range(len(t_))])

x_np = np.array([Clara.q__[i][0,0] for i in range(len(t_))])
y_np = np.array([Clara.q__[i][1,0] for i in range(len(t_))])
theta11_np = np.array([Clara.q__[i][2,0] for i in range(len(t_))])
theta12_np = np.array([Clara.q__[i][3,0] for i in range(len(t_))])
theta21_np = np.array([Clara.q__[i][4,0] for i in range(len(t_))])
theta22_np = np.array([Clara.q__[i][5,0] for i in range(len(t_))])


dx_np = np.array([Clara.dq__[i][0,0] for i in range(len(t_))])
dy_np = np.array([Clara.dq__[i][1,0] for i in range(len(t_))])
dtheta11_np = np.array([Clara.dq__[i][2,0] for i in range(len(t_))])
dtheta12_np = np.array([Clara.dq__[i][3,0] for i in range(len(t_))])
dtheta21_np = np.array([Clara.dq__[i][4,0] for i in range(len(t_))])
dtheta22_np = np.array([Clara.dq__[i][5,0] for i in range(len(t_))])

d2x_np = np.array([Clara.d2q__[i][0,0] for i in range(len(t_))])
d2y_np = np.array([Clara.d2q__[i][1,0] for i in range(len(t_))])
d2theta11_np = np.array([Clara.d2q__[i][2,0] for i in range(len(t_))])
d2theta12_np = np.array([Clara.d2q__[i][3,0] for i in range(len(t_))])
d2theta21_np = np.array([Clara.d2q__[i][4,0] for i in range(len(t_))])
d2theta22_np = np.array([Clara.d2q__[i][5,0] for i in range(len(t_))])

plt.figure()
plt.plot(t_np, x_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$x[m]$')
plt.title('Lalala')
plt.savefig('x.png')

plt.figure()
plt.plot(t_np, y_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$y[m]$')
plt.title('Lalala')
plt.savefig('y.png')

plt.figure()
plt.plot(t_np, theta11_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$theta11[rad]$')
plt.title('Lalala')
plt.savefig('theta11.png')

plt.figure()
plt.plot(t_np, theta12_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$theta12[rad]$')
plt.title('Lalala')
plt.savefig('theta12.png')

plt.figure()
plt.plot(t_np, theta21_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$theta21[rad]$')
plt.title('Lalala')
plt.savefig('theta21.png')

plt.figure()
plt.plot(t_np, theta22_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$theta22[rad]$')
plt.title('Lalala')
plt.savefig('theta22.png')

plt.figure()
plt.plot(t_np, dx_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$dx[m/s]$')
plt.title('Lalala')
plt.savefig('dx.png')

plt.figure()
plt.plot(t_np, dy_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$dy[m/s]$')
plt.title('Lalala')
plt.savefig('dy.png')

plt.figure()
plt.plot(t_np, dtheta11_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$dtheta11[rad/s]$')
plt.title('Lalala')
plt.savefig('dtheta11.png')

plt.figure()
plt.plot(t_np, dtheta12_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$dtheta12[rad/s]$')
plt.title('Lalala')
plt.savefig('dtheta12.png')

plt.figure()
plt.plot(t_np, dtheta21_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$dtheta21[rad/s]$')
plt.title('Lalala')
plt.savefig('dtheta21.png')

plt.figure()
plt.plot(t_np, dtheta22_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$dtheta22[rad/s]$')
plt.title('Lalala')
plt.savefig('dtheta22.png')

plt.figure()
plt.plot(t_np, d2x_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$d2x[m/s2]$')
plt.title('Lalala')
plt.savefig('d2x.png')

plt.figure()
plt.plot(t_np, d2y_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$d2y[m/s2]$')
plt.title('Lalala')
plt.savefig('d2y.png')

plt.figure()
plt.plot(t_np, d2theta11_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$d2theta11[rad/s2]$')
plt.title('Lalala')
plt.savefig('d2theta11.png')

plt.figure()
plt.plot(t_np, d2theta12_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$d2theta12[rad/s2]$')
plt.title('Lalala')
plt.savefig('d2theta12.png')

plt.figure()
plt.plot(t_np, d2theta21_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$d2theta21[rad/s2]$')
plt.title('Lalala')
plt.savefig('d2theta21.png')

plt.figure()
plt.plot(t_np, d2theta22_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$d2theta22[rad/s2]$')
plt.title('Lalala')
plt.savefig('d2theta22.png')

plt.figure()
plt.plot(t_np, u1_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$u1[Nm]$')
plt.title('Lalala')
plt.savefig('u1.png')

plt.figure()
plt.plot(t_np, u2_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
plt.ylabel(r'$u2[Nm]$')
plt.title('Lalala')
plt.savefig('u2.png')

plt.figure()
plt.plot(t_np, qbarnorm_np, 'r', linewidth=2)
plt.xlabel(r'$t[s]$')
#plt.ylabel(r'$x[m]$')
plt.title('Lalala')
plt.savefig('qbarnorm.png')

gnr = GNR(Clara, 
          np.matrix([[0.5375391526183005, 2.308800210309811,  0.5375391526183005, 2.308800210309811]]).T, 
          np.matrix([[0.17,0.16]]).T, 
          5)
print norm(gnr.qbar_, np.inf)
gnr.doit()
print norm(Clara.qbar_, np.inf)