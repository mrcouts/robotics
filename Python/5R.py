#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from math import cos
from math import sin
from math import pi
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from Utilities import *
from Serial import *
from Paralelo import *
from RK import *
from GNR import *

""" Arquitetura das cadeias seriais """
fDH_RR = lambda q_, l_, lg_: np.matrix([
                                       [l_[0], 0, 0, q_[0,0], -l_[0] + lg_[0], 0,  0, True],
                                       [l_[1], 0, 0, q_[1,0], -l_[1] + lg_[1], 0,  0, True]
                                       ])

""" Parametros das cadeias """
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

""" Gravidade """
gamma_ = Zeros(3,1)
gamma_[1] = -9.8

""" Cadeias Seriais """        
RR1 = Serial(dof, l_, lg1_, m1_, I1__, gamma_, fDH_RR)
RR2 = Serial(dof, l_, lg2_, m2_, I2__, gamma_, fDH_RR)
RR_ = [RR1, RR2]

""" Efetuador """
EndEffector = EfetuadorTranslacional(0, gamma_, 0, 0, 0, False)

""" Parametros da arquitetura do mecanismo Paralelo """
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

""" Mecanismo Paralelo """
Clara = Paralelo(dof, RR_, EndEffector, angulos__, origens__, xb__, [0,1], [2,4])

""" Condicao inicial """
claraq_ = np.matrix([[0.0,  0.108,  0.5375391526183005, 2.308800210309811,  0.5375391526183005, 2.308800210309811]]).T

""" Definicao da trajetoria """
w = 2*pi
r = 0.05
qhr_ = lambda t: np.matrix([[ -r*sin(w*t), 0.158 -r*cos(w*t)]]).T
dqhr_ = lambda t: np.matrix([[ -w*r*cos(w*t),  w*r*sin(w*t)]]).T
d2qhr_ = lambda t: np.matrix([[ w*w*r*sin(w*t), w*w*r*cos(w*t)]]).T
Clara.set_ref(qhr_, dqhr_, d2qhr_)

""" Obtencao das condicoes iniciais de velocidades """
Clara.doit_q_(claraq_)
dqo_ = Clara.fdqo_(0, Clara.Qo_.T*claraq_)
claradq_ = Clara.C_*np.matrix([[1.0, 2.0]]).T
Clara.doit_q_dq_(claraq_, claradq_)

""" Simulacao Dinamica Inversa """
rk = RK('RK3')
y__, t_ = rk.ApplyParallelInvDyn(0.001, 2.000, Clara.Qo_.T*claraq_, Clara, 10)

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