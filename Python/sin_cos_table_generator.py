# -*- coding: utf-8 -*-
from math import cos
from math import sin
from math import pi

x0 = 0.0
xf = 2*pi
n = 10000000

X = [x0 + i*(xf-x0)/(n-1) for i in range(n)] 
SIN = [sin(x) for x in X]
COS = [cos(x) for x in X]

f = open("sin_cos.txt", "w")

for i in range(n):
    f.write("%.50f; " % X[i])
    f.write("%.50f; " % SIN[i])
    f.write("%.50f;\n" % COS[i])

f.close()