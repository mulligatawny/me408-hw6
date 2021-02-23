import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby
from sklearn.metrics import mean_squared_error

N = np.array([16, 640])
#N = np.array([10, 20, 40, 80, 160, 320, 640])

def plot_derivative(N, func, method):
    M = 10
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
    # exact solution
    xe = np.linspace(-1, 1, 640)
    ex = dfdx(xe)
    # interpolate onto 640 nodes
    fpi = np.interp(xe, x, fp)
    # error
    e = mean_squared_error(fpi, ex)
    plt.plot(x, fp, '-o', label='N = {}'.format(N))
    return e
e = np.zeros_like(N, dtype='float')
for i in range(len(N)):
    e[i] = plot_derivative(N[i], 2, 'chebyshev')

#x = np.linspace(-1, 1, 128)
#plt.plot(x, dfdx(x), '-', label='exact')
plt.xlabel('$x$')
plt.ylabel('$df$/$dx$')
plt.legend()
plt.show()
print(e)
