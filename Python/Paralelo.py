#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import block_diag
from numpy.linalg import solve
from Utilities import *
from Serial import *

class EfetuadorTranslacional:
    def __init__(self, m, gamma_, a, b, c, Espacial = True):
        self.transdof = 3 if Espacial == True else 2
        self.dof = self.transdof
        self.q_ = Zeros(self.dof,1)
        self.dq_ = Zeros(self.dof,1)
        self.R_ = EulerAngles2R_(a,b,c)
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
        self.Qh_ = Qh_Qo__(lista_ind, self.nq)[0]
        self.Qo_ = Qh_Qo__(lista_ind, self.nq)[1]
        self.U_ = Qh_Qo__(lista_actuator, self.nq)[0]
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
            self.R__[i] = EulerAngles2R_(*(angulos__[i].T.tolist()[0]))[0:EndEffector.transdof,:]
            
        self.D_ = np.vstack([ Eye(EndEffector.transdof) for i in range(self.n) ])
        self.E_ = np.matrix(block_diag(*self.R__))
        self.d_ = np.vstack([ (origens__[i] - EndEffector.R_*xb__[i])[0:EndEffector.transdof,:] for i in range(dof) ])
        
    def doit_q_(self, q_):
        self.q_ = q_
        soma1 = 0
        soma2 = soma1 + self.EndEffector.dof
        self.EndEffector.doit_q_(q_[soma1:soma2,0])
        for i in range(self.n):
            soma1 = soma2
            soma2 += self.Serial_[i].dof
            self.Serial_[i].doit_q_(q_[soma1:soma2,0])
            
        self.x_ = np.vstack([ self.Serial_[i].x_ for i in range(self.dof) ])
        self.qbar_ = self.D_*self.EndEffector.q_- self.d_ - self.E_*self.x_
        self.Jv_ = np.matrix(block_diag(*[self.Serial_[i].Jv_ for i in range(self.n)]))
        self.A_ = np.hstack([self.D_, -self.E_*self.Jv_])
        self.Ah_ = self.A_*self.Qh_
        self.Ao_ = self.A_*self.Qo_
        self.C_ = self.Qh_ - self.Qo_*solve(self.Ao_,self.Ah_)
        self.ZT_ = self.C_.T*self.U_
        self.M_ = np.matrix(block_diag(*([self.EndEffector.M_] + [self.Serial_[i].M_ for i in range(self.n)])))
        self.g_ = np.vstack([self.EndEffector.g_] + [ self.Serial_[i].g_ for i in range(self.dof) ])
        self.Mh_ = self.C_.T*self.M_*self.C_
        self.gh_ = self.C_.T*self.g_
        
    def doit_dq_(self, dq_):
        self.dq_ = dq_
        q_ = self.q_
        soma1 = 0
        soma2 = soma1 + self.EndEffector.dof
        self.EndEffector.doit_q_dq_(q_[soma1:soma2,0], q_[soma1:soma2,0])
        for i in range(self.n):
            soma1 = soma2
            soma2 += self.Serial_[i].dof
            self.Serial_[i].doit_q_dq_(q_[soma1:soma2,0], dq_[soma1:soma2,0])
        self.atil_ = np.vstack([ self.Serial_[i].atil_ for i in range(self.dof) ])
        self.b_ = -self.E_*self.atil_
        self.c_ = - self.Qo_*solve(self.Ao_,self.b_)
        self.v_ = np.vstack([self.EndEffector.v_] + [ self.Serial_[i].v_ for i in range(self.dof) ])
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