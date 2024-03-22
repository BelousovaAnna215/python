from typing import List


def sum_non_neg_diag(X):
    sum = 0
    fl = False
    for a in range (min(len(X), len(X[0]))):
        if X[a][a] >= 0:
          sum += X[a][a]
          fl = True
    if fl:
      return sum
    else :
      return -1
      
    
def are_multisets_equal(x, y):
    if len(x) != len(y):
        return False
    x = list(x)
    y = list(y)
    x.sort()
    y.sort()
    for i in range(len(x)):
        if x[i] != y[i]:
            return False
    return True


def max_prod_mod_3(x) :
    m = 0
    for a in range (len(x)-1) :
      if x[a]*x[a+1] % 3 == 0 :
        m = max(m,x[a]*x[a+1])
    if m :
      return m
    else :
      return -1


def convert_image(image, weights):
    X = []
    for i in image:
        Y = []
        for j in i:
            S = 0
            for k in range(len(j)):
                S += j[k]*weights[k]
            Y.append(S)
        X.append(Y)
    return X


def rle_scalar(x, y):
    a = []
    b = []
    for i in range (len(x)):
      for j in range (x[i][1]):
        a.append(x[i][0])
    for i in range (len(y)):
      for j in range (y[i][1]):
        b.append(y[i][0])
    sum = 0
    if len(a) == len(b):
      for i in range (len(a)):
        sum += a[i]*b[i]
      return sum
    else:
      return -1
      

def cosine_distance(X, Y):
  res = []
  for i in X:
    Z = []
    for j in Y:
      S = 0
      SA = 0
      SB = 0
      for k in range(len(j)):
        S += i[k] * j[k]
        SA += i[k] ** 2
        SB += j[k] ** 2
      if SA*SB == 0 :
        Z.append(1)  
      else :
        Z.append(S/(( SA*SB ) ** 0.5))
    res.append(Z)
  return res
