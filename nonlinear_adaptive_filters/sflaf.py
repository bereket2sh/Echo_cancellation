

import numpy as np

def sflaf(x, d, M=128, P=5, mu_L=0.2, mu_FL=0.5):
  nIters = min(len(x),len(d)) - M
  Q = P*2
  sk = np.arange(0,Q*M,2)
  ck = np.arange(1,Q*M,2)
  pk = np.tile(np.arange(P),M)
  u = np.zeros(M)
  w_L = np.zeros(M)
  w_FL = np.zeros(Q*M)
  e = np.zeros(nIters)    
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    g = np.repeat(u,Q)
    g[sk] = np.sin(pk*np.pi*g[sk])
    g[ck] = np.cos(pk*np.pi*g[ck])
    y_L = np.dot(w_L, u.T)
    y_FL = np.dot(w_FL,g.T)
    y = y_L + y_FL
    e_n = d[n] - y
    w_L = w_L + mu_L * e_n * u / (np.dot(u,u)+1e-3)
    w_FL = w_FL + mu_FL * e_n * g / (np.dot(g,g)+1e-3)
    e[n] = e_n
  return e