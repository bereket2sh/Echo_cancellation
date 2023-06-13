

import numpy as np

def rls(x, d, N = 4, lmbd = 0.999, delta = 0.01):
  nIters = min(len(x),len(d)) - N
  lmbd_inv = 1/lmbd
  u = np.zeros(N)
  w = np.zeros(N)
  P = np.eye(N)*delta
  e = np.zeros(nIters)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    e_n = d[n] - np.dot(u, w)
    r = np.dot(P, u)
    g = r / (lmbd + np.dot(u, r))
    w = w + e_n * g
    P = lmbd_inv*(P - np.outer(g, np.dot(u, P)))
    e[n] = e_n
  return e