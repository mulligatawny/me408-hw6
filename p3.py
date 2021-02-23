# 3/5
import numpy as np
from scipy.special import j0, j1, jn_zeros
import matplotlib.pyplot as plt

N = 40
x = np.linspace(0, 1, N)
c = np.zeros(N)
for i in range(N):
    z = jn_zeros(0,N)
    c[i] = 2/(z[i]*j1(z[i]))
f = c*j0(z*x)

plt.plot(x, f)
plt.ylim([-1,1])
plt.show()
