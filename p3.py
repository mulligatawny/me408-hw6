# 3/5
import numpy as np
from scipy.special import j0, j1, jn_zeros
import matplotlib.pyplot as plt

N = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
def plot_besself(N):
    x = np.linspace(0, 1, 1000)
    c = np.zeros(N)
    Sn = np.zeros_like(x, dtype='float')
    for i in range(N):
        z = jn_zeros(0,N)           # zeros of zeroth Bessel function
        c[i] = 2/(z[i]*j1(z[i]))    # Bessel coefficient
        Sn = Sn + c[i]*j0(z[i]*x)   # Bessel series
    # compute error
    exact = np.ones_like(x, dtype='float')
    e0 = abs(exact[0] - Sn[0])
    e1 = np.max(abs(exact[int(0.4*len(x)):int(0.6*len(x))] \
         -Sn[int(0.4*len(x)):int(0.6*len(x))]))  
    plt.plot(x, Sn, label='N = {}'.format(N))
    return e0, e1

e0 = np.zeros_like(N, dtype='float')
e1 = np.zeros_like(N, dtype='float')
for i in range(len(N)):
    e0[i], e1[i] = plot_besself(N[i])
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Approximation with Bessel\'s series')
plt.legend()
plt.show()
# plot errors
plt.loglog(N, e0, 'D', label='x=0 error')
plt.loglog(N, 1/N, label='slope-1')
plt.loglog(N, e1, '^', label='x=0.4-0.6 error')
plt.loglog(N, 1/N**(0.5), label='slope-0.5')
plt.legend()
plt.grid(which='both')
print(e1)
plt.show()
