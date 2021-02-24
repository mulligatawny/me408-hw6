# 3/5
import numpy as np
from scipy.special import j0, j1, jn_zeros
import matplotlib.pyplot as plt

N = np.array([10, 20, 40])
def plot_besself(N):
    x = np.linspace(0, 1, 100000)
    c = np.zeros(N)
    Sn = np.zeros_like(x, dtype='float')
    for i in range(N):
        z = jn_zeros(0,N)           # zeros of zeroth Bessel function
        c[i] = 2/(z[i]*j1(z[i]))    # Bessel coefficient
        Sn = Sn + c[i]*j0(z[i]*x)   # Bessel series

    plt.plot(x, Sn, label='N = {}'.format(N))
for i in range(len(N)):
    plot_besself(N[i])
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Approximation with Bessel\'s series')
plt.legend()
plt.show()
