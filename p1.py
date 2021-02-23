import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby

N = 8
t = np.arange(0, N+1)*np.pi/N
x = np.cos(t)
f = lambda x: x**6
If = np.trapz(f(x))

Fk = cheby.cheby(x, f)
k = np.arange(0, N+1)

d = np.zeros(N+1)
d[-1] = 0
d[-2] = 0
for i in range(N,0,-1):
    if i == N+1:
        d[i] = 1/(2*i)*2*Fk[i-1]
    elif i == N:
        d[i] = 1/(2*i)*Fk[i-1]
    elif i == 2:
        d[i] = 1/(2*i)*2*Fk[i-1]
    else:
        d[i] = 1/(2*i)*(Fk[i-1] - Fk[i+1])

dp = d
s = dp[1::2]
#print(dp)
#print(dp[::2])
#print(dp[1::2]) # odd indices
dp[0] = sum(dp[1::2]) - sum(dp[::2])

dc = cheby.icheby(t, dp)
print(If)
print(dc[-1])
plt.plot(x, dc,'-o')
#plt.show()
