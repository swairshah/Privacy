import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_regression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

m = 600
n = 500
N = 10

X, y = make_regression(n_samples = m, 
                       n_features= n, 
                       n_targets = N,
                       random_state=0)


tr_size = int(m*0.8)
Xtr = X[:tr_size,:]
Xtst = X[tr_size:,:]
ytr = y[:tr_size,:]
ytst = y[tr_size:,:]


lm = LinearRegression()
lm.fit(Xtr, ytr)

y_hat = lm.predict(Xtst)
print("LinearModel error:", np.linalg.norm(ytst - y_hat))


