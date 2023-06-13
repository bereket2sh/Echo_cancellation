

import numpy as np
from scipy.linalg import hankel

def blms(x, d, N=4, L=4, mu = 0.1):
  nIters = min(len(x),len(d))//L
  u = np.zeros(L+N-1)
  w = np.zeros(N)
  e = np.zeros(nIters*L)
  for n in range(nIters):
    u[:-L] = u[L:]
    u[-L:] = x[n*L:(n+1)*L]
    d_n = d[n*L:(n+1)*L]
    A = hankel(u[:L],u[-N:])
    e_n = d_n - np.dot(A,w)
    w = w + mu*np.dot(A.T,e_n)/L
    e[n*L:(n+1)*L] = e_n
  return e