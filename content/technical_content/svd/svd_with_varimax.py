

import numpy as np 
import pandas as pd
import os

from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer.rotator import Rotator

os.chdir('content/technical_content/svd/')
df_wine = pd.read_csv('data_wine_drinkers_short.csv', index_col=False)

X = scale(df_wine)

# apply PCA
pca = decomposition.PCA(n_components=2)
X = pca.fit_transform(X)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

df_center = (df_wine - df_wine.mean())
df_std = df_center/df_wine.std(ddof=0)

# 1) Analysis on centered data set
np.cov(df_wine)
U_cent, S_cent, vt_cent = np.linalg.svd(df_center)
V_cent = vt_cent.T
eig_cent = (S_cent**2)/9

(V_cent * eig_cent).dot(V_cent.T)

# Apply varimax rotation
rotator = Rotator()
rotated_loadings_cent = rotator.fit_transform(V_cent[: ,:2])


# 2) Analysis on standardized data set
df_corr = df_wine.corr()

U_std, E_std, vt_std = np.linalg.svd(df_std)

V_std = vt_std.T

# convert eigenvalues to singualr values 
np.linalg.eigvals(df_corr)
sing_values = np.sqrt(E_std * (10 - 1))

# why doesn't this give the correlation matrix?
V_std.dot(S_std_diag).dot(V_std.T)



loadings = V_std.dot(S_std_diag)

loadings.dot(loadings.T)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer.rotator import Rotator

# PCA on mean-centered data
pca = PCA(n_components=2)
output = pca.fit(df_std)
loadings_std = pca.components_.T  
pca.explained_variance_


# Apply varimax rotation
rotator = Rotator()
rotated_loadings_pca = rotator.fit_transform(loadings_std)

rotated_loadings_pca * np.sqrt(pca.explained_variance_)



# PCA on standardized data (loadings = correlations)
pca = PCA(n_components=2)
output = pca.fit(df_std)
loadings_std = pca.components_.T  




# Step 0. Compute centered variables X and covariance matrix S.
cov_wine = np.dot(df_center.T, df_center)/9

# Step 1.1. Decompose data X or matrix S to get eigenvalues and right eigenvectors.
          # You may use svd or eigen decomposition (see https://stats.stackexchange.com/q/79043/3277)
eig_values = np.linalg.eigvals(cov_wine)
U, S_std, vt = np.linalg.svd(df_center)
V = vt.T

Step 1.2. Decide on the number M of first PCs you want to retain.
          You may decide it now or later on - no difference, because in PCA values of components do not depend on M.
          Let's M=2. So, leave only 2 first eigenvalues and 2 first eigenvector columns.

Step 2. Compute loadings A. May skip if you don't need to interpret PCs anyhow.
Loadings are eigenvectors normalized to respective eigenvalues: A value = V value * sqrt(L value)
Loadings are the covariances between variables and components.

Loadings A
              PC1           PC2           
SLength    .32535081     .11487892
SWidth     .35699193    -.11925773
PLength    .04694612     .09416050
PWidth     .03090888     .02515873

U, S_std, vt = np.linalg.svd(df_center)
V_std = vt.T  # Transpose to get V matrix
rotated_loadings_svd_unweighted = pd.DataFrame(rotator.fit_transform(V_std[:, :2])*-1)



U, S_raw, vt = np.linalg.svd(df_wine)

V_raw = vt.T  # Transpose to get V matrix
rotated_loadings_svd_unweighted = rotator.fit_transform(V[:, :2])*-1


cov_matrix = np.cov(df_std, rowvar=False)
np.linalg.eigvals(cov_matrix)


# Scale V by the singular values to match PCA scaling
scaled_V = V[:, :2] * S[:2]  # Only keep the first 2 components
scaled_U = U[:, :2] * S[:2]  # Only keep the first 2 components


rotated_loadings_svd = rotator.fit_transform(scaled_V)
rotated_scores_svd = rotator.fit_transform(scaled_U)

from sklearn.preprocessing import normalize
normalized_rotated_svd_loadings = normalize(rotated_loadings_svd, axis=0)
normalized_rotated_svd_scores = normalize(rotated_scores_svd, axis=0)

