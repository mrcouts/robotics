# -*- coding: utf-8 -*-
from math import cos
from math import sin
from math import pi

x0 = 0.0
xf = 2*pi
n = 10

SIN = [sin(x0 + i*(xf-x0)/(n-1)) for i in range(n)]
COS = [cos(x0 + i*(xf-x0)/(n-1)) for i in range(n)]

f1 = open("sin.txt", "w")
f2 = open("cos.txt", "w")

for i in range(n-1):
    f1.write("%f \n" % SIN[i])
    f2.write("%f \n" % COS[i])
    
f1.write("%f" % SIN[n-1])
f2.write("%f" % COS[n-1])

f1.close()
f2.close()