#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from numpy.linalg import solve
from Utilities import *
from Serial import *
from Paralelo import *
from RK import *

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