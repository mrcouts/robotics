# -*- coding: utf-8 -*-
from Utilities import *

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
            
        self.x_ = self.o__[self.dof-1]
            
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
            
        self.Jw_ = self.Jw__[self.dof-1]
        self.Jv_ = self.Jv__[self.dof-1] - S_(self.x_ - self.g__[self.dof-1])*self.Jw__[self.dof-1]
        
        self.M_ = Zeros(self.dof,self.dof)
        self.g_ = Zeros(self.dof,1)            
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
            
        self.alphatil_ = self.alphatil__[self.dof-1]
        self.atil_ = self.atil__[self.dof-1] + (S_(self.alphatil__[self.dof-1]) + S2_(self.w__[self.dof-1]))*(self.x_ - self.g__[self.dof-1])
        
        self.v_ = Zeros(self.dof,1)
        for i in range(self.dof):
            self.v_ += self.m_[i]*self.Jv__[i].T*self.atil__[i] + self.Jw__[i].T*(self.I__[i]*self.alphatil__[i] + S_(self.w__[i])*self.I__[i]*self.w__[i] )