# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:41:52 2023

@author: Nebula
"""

import numpy as np
import matplotlib.pyplot as plt

N = int(input("size of matrix: "))
w = float(input("value of w0: "))

h = 2 * np.pi / N

diagonal = np.zeros(N)
A = np.diag(diagonal)

j = 0 
n = 0
for j in range(N-1):
    for n in range(N-1):
        if j!=n:
            p = ((j*h)-(n*h))/2
            q = 1 / np.tan(p)
            r = j + n
            s = -1 ** r
            d = (s * q)/2
            A[j][n] = d
    
print(A)

#checking for skew symmetric matrix
#negative transpose = original
b = A.transpose(1,0)
print(A+b)

#Matrix D2 = A^2
C = np.power(A,2)
print(C)

#Identity matrix
a = w**2
diag2 = np.array([a for i in range(N)])
D = np.diag(diag2)
print(D)
#A^2 + I
res = C + D

#function f
k=0
F = np.array([3 * np.cos(k*h) for k in range(N)]).reshape(N,1)
print(F)

#solve Au=f
x = np.linalg.solve(res, F)
#u
print(x)

#plotting
t = np.linspace(0, 2*np.pi, N)
print (t)
#plot x vs t
plt.plot(t, x)
plt.xlabel('t')
plt.ylabel('x')
plt.title('Plot of x vs t')
plt.show()