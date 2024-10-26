

import numpy as np 
import pandas as pd


df_wine = pd.read_csv('data_wine_drinkers.csv', index_col=False)


df_wine = df_wine[['V1', 'V2', 'V3', 'V4']]

df_std= (df_wine - df_wine.mean())/df_wine.std()

from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_standardized = scaler.fit_transform(df_wine)

pca = PCA()
pca.fit(data_standardized)

U, S, v = np.linalg.svd(df_std)


def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)
  

pd.DataFrame(varimax(v[:, :2]))
  
