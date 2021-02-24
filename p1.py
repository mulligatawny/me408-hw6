# 1/5
import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby
from scipy import integrate

N = np.array([4, 8, 16])
def compute_int(N, func=1):
    t = np.arange(0, N+1)*np.pi/N
    x = np.cos(t)
    if func==1:
        f = lambda x: x**6
    elif func==2:
        f = lambda x: x*np.exp(-(x**2)/2) # part (a)
    else:
        f = lambda x: np.piecewise(x, [x < 0, x >= 0], \
        [lambda x: -2*x-1,lambda x: 2*x-1]) # part (b)
    # true integral
    Ie = np.zeros_like(x)
    for i in range(len(x)):
        Ie[i] = integrate.quadrature(f, -1, x[i], maxiter=1000)[0]
    # chebyshev transform
    Fk = cheby.cheby(x, f)
    k = np.arange(0, N+1)
    # compute coefficients (dn)
    d = np.zeros(N+1)
    for i in range(1, N+1):
        if i==1:
            d[1] = 1/(2*i)*(2*Fk[i-1] - Fk[i+1])
        elif i==N:
            d[N] = 1/(2*i)*(2*Fk[i-1])
        else:
            d[i] = 1/(2*i)*(1*Fk[i-1] - Fk[i+1])
    d[0] = sum(d[1::2]) - sum(d[::2])
    # inverse transform
    dc = cheby.icheby(t, d)
    return x, Ie, dc

for i in range(len(N)):
    x, Ie, dc = compute_int(N[i], 2)
    plt.plot(x,dc, label='N = {}'.format(N[i]))
plt.plot(x, Ie,'D',label='exact')
plt.xlabel('$x$')
plt.ylabel('Integral')
plt.legend()
plt.show()
