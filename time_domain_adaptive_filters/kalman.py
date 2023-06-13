

import numpy as np


def kalman(x, d, N = 64, sgm2v=1e-4):
  nIters = min(len(x),len(d)) - N
  u = np.zeros(N)
  w = np.zeros(N)
  Q = np.eye(N)*sgm2v
  P = np.eye(N)*sgm2v
  I = np.eye(N)
  e = np.zeros(nIters)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    e_n =  d[n] - np.dot(u, w)
    R = e_n**2+1e-10
    Pn = P + Q
    r = np.dot(Pn,u)
    K = r / (np.dot(u, r) + R + 1e-10)
    w = w + np.dot(K, e_n)
    P = np.dot(I - np.outer(K, u), Pn)
    e[n] = e_n

  return e