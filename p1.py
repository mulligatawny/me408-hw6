# 1/5
import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby
from scipy import integrate

np.set_printoptions(precision=3)

N = np.array([4, 8, 16])
def compute_int(N):
    t = np.arange(0, N+1)*np.pi/N
    x = np.cos(t)
#    f = lambda x: x**6
#    f = lambda x: x*np.exp(-(x**2)/2) # part (a)
    f = lambda x: np.piecewise(x, [x < 0, x >= 0], \
    [lambda x: -2*x-1,lambda x: 2*x-1]) # part (b)
    # true integral
    Ie = integrate.quadrature(f, -1, 1, maxiter=100)[0]
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
    print(dc)
    return Ie, dc

for i in range(len(N)):
    Ie, dc = compute_int(N[i])
    print('For N = {}'.format(N[i]), 'the computed value is',dc[0])
    print('The error is', (dc[0]-Ie)**2)
    plt.plot(np.arange(N[i]+1)/N[i],dc)
print('The true value is:', Ie)
plt.show()
