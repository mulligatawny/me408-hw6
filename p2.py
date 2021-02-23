# 2/5
import numpy as np
import matplotlib.pyplot as plt
import itertools
from transforms import cheby
from sklearn.metrics import mean_squared_error

N = np.array([10, 20, 40, 80, 160, 320, 640])
M = np.array([10, 20, 40, 80])

def plot_derivative(N, M):
    t = np.arange(0, N+1)*np.pi/N
    x = np.cos(t)
    f = lambda x: np.sin(M*np.pi*x)
    dfdx = lambda x: M*np.pi*np.cos(M*np.pi*x)

    # compute chebyshev transform
    Fk = cheby.cheby(x, f)
    k = np.arange(0, N+1)
    # assemble bi-diagonal matrix
    A = np.zeros((N+1, N+1))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:,1:], -1)
    A[0,:] = 0
    A[1,0] = 2
    nA = A[1:,:-1]
    # assemble RHS
    b = np.zeros(N+1)
    b = 2*k*Fk
    bn = b[1:]
    # solve bi-diagonal system
    phi = np.linalg.solve(nA, bn)
    # set last coefficient to 0
    phi = np.append(phi, 0.0)
    # inverse transform
    fp = cheby.icheby(t, phi)
    # error
    e = mean_squared_error(fp, dfdx(x))
    return e

e = np.zeros((len(N), len(M)))
for j in range(len(M)):
    for i in range(len(N)):
        e[i,j] = plot_derivative(N[i], M[j])

marker = itertools.cycle(('D', '*', '^', 'o')) 
for j in range(len(M)):
    plt.loglog(N/M[j], e[:,j], marker=next(marker),label='M = {}'.format(M[j]))
plt.xlabel('$N/M$')
plt.ylim([1e-18,1e6])
plt.vlines(np.pi, 1e-18, 1e6,linewidth=2,color='k')
plt.grid(which='both')
plt.ylabel('Error')
plt.legend()
plt.show()
