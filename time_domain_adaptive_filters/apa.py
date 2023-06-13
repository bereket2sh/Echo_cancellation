

import numpy as np

def apa(x, d, N = 4, P = 4, mu = 0.1):
  nIters = min(len(x),len(d)) - N
  u = np.zeros(N)
  A = np.zeros((N,P))
  D = np.zeros(P)
  w = np.zeros(N)
  e = np.zeros(nIters)
  alpha = np.eye(P)*1e-2
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    A[:,1:] = A[:,:-1]
    A[:,0] = u
    D[1:] = D[:-1]
    D[0] = d[n] 
    e_n = D - np.dot(A.T, w)
    delta = np.dot(np.linalg.inv(np.dot(A.T,A)+alpha),e_n)
    w = w + mu * np.dot(A ,delta)
    e[n] = e_n[0]
  return e