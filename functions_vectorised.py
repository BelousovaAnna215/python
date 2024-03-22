import numpy as np
from typing import Tuple


def sum_non_neg_diag(X):
    diag = np.diag(X)
    d2 = np.array(diag[diag >= 0])
    if len(d2) > 0 :
        return d2.sum()
    else:
        return -1


def are_multisets_equal(x, y):
    return np.array_equal(np.sort(x), np.sort(y))


def max_prod_mod_3(x):
  if (len(x) <= 1):
    return -1
  else:
    y = x.copy()
    y = np.append(y, [x[0]])
    y = np.delete(y, [0])
    m = x*y
    m[(m%3 != 0)] = 0
    if m.max() == 0:
      return -1
    return m.max()

def convert_image(image, weights):
    return np.sum(image * weights, axis = 2)


def rle_scalar(x, y):
  newx = np.repeat(x[..., 0], x[..., 1])
  newy = np.repeat(y[..., 0], y[..., 1])
  return int(np.dot(newy, newx)) if newx.shape[0] == newy.shape[0] else -1
  

def cosine_distance(X, Y):
    m = np.sqrt((X*X).sum(axis=1).reshape(-1, 1) * (Y*Y).sum(axis=1))
    k = np.ma.array(np.dot(X,Y.T))
    return (k/m).filled(1)
