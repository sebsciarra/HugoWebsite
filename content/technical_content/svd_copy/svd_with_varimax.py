

import numpy as np 
import pandas as pd
import os

from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer.rotator import Rotator


# orthogonal projections give the value of a vector in another basis 

# os.chdir('content/technical_content/svd/')

df_wine = pd.read_csv('data_wine_drinkers_short.csv', index_col=False)

# apply PCA
df_center = (df_wine - df_wine.mean())
df_std = df_center/df_wine.std(ddof=1)
df_corr = df_wine.corr()


# needed for right eigenvectors 
U_cent, S_cent, vt_cent = np.linalg.svd(df_std)
V_cent = vt_cent.T



V_cent.dot(np.linalg.inv(V_cent.T.dot(V_cent))).dot(V_cent.T)
# scores

# rectangular_matrix = np.zeros((10, 4))
# np.fill_diagonal(rectangular_matrix, (S_cent**2)/9)

pca = decomposition.PCA(n_components=2)
pca_scores_og = pca.fit_transform(df_std)
pca_score_rep = pd.DataFrame(U_cent[ : ,:2].dot(np.diag(S_cent[:2]))) 


pca_score_std = (pca_score_rep/pd.DataFrame(pca_score_rep).std())

df_std.T.dot(pca_score_std)/9

pd.DataFrame(U_cent[ : ,:2].dot(np.sqrt(np.diag(S_cent[:2]))))
U_cent[ : ,:2]*3
# confirm sd
eig_values = np.diag((S_cent[0:2]**2)/9)

loadings_pca = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_pca1 = V_cent[: ,:2].dot(np.sqrt(np.diag(S_cent[0:2]**2)/9))
loadings_pca2 = df_std.T.dot(U_cent[: ,:2])/np.sqrt(10-1)

pca_scores = df_std.dot(V_cent[ : , :2])


loadings_pca = pca.components_.T * np.sqrt(pca.explained_variance_)
pca_scores_rep = df_std.dot(loadings_pca)

pca_scores1 = pd.DataFrame(U_cent[ : ,:2].dot(np.diag(S_cent[:2])))
pca_scores_std = df_std.dot(V_cent[ : , :2]).dot(np.sqrt(np.diag((S_cent[0:2]**2)/9)))
pca_scores_std.std()
pca_scores.var() 

# scale PC scores such that variances becomes new SD
new_std = (S_cent[0:2]**2)/9
old_std = np.sqrt(new_std)
scaling_factors = new_std/old_std

standarzing_factors = 1/new_std

# scale PC scores such that SD=1 and mean=0
Q = np.linalg.inv(loadings_pca.T.dot(loadings_pca))

Q_var = np.linalg.inv((loadings_pca.T.dot(loadings_pca)**2))

(df_std.dot(V_cent[ : , :2]).dot(np.sqrt(np.diag((S_cent[0:2]**2)/9))).dot(Q)).std()




# standardized unrotated scores
#            PC1        PC2     rot
#  0.06420693  1.0554468 varimax
# -0.23247171  1.1378799 varimax
#  1.07555164  1.1295544 varimax
#  0.80807295  0.7756882 varimax
# -2.00047241  0.5404843 varimax
# -0.58404876 -0.8007176 varimax
# -1.03529633 -0.8829445 varimax
#  0.73638778 -1.2262495 varimax
#  1.08039130 -0.8974282 varimax
#  0.08767861 -0.8317139 varimax


# another way to compute loadings 
pca_score_rep = U_cent[ : ,:2].dot(np.diag(S_cent[0:2]**2)/9) # standardized scores
pca_scores_stand = (pca_score_rep - pca_score_rep.mean(axis=0)) / pca_score_rep.std(axis=0, ddof=1)
loadings_pca2 = df_std.T.dot(pca_scores_stand)/9
loadings_pca3 = V_cent[ : ,:2].dot(np.diag(S_cent[0:2]))/np.sqrt(9)


np.corrcoef(df_std.T, pca_scores1.T)[:df_std.shape[1], pca_scores1.shape[1]:]



# Apply varimax rotation on  variable loadings 
rotator = Rotator(method="varimax")
quatimax = Rotator(method="oblimax")

rotated_loadings = rotator.fit_transform(loadings_pca)
quatirmax_loadings = quatimax.fit_transform(loadings_pca)

# Step 4: Compute the rotated component scores
eigenvalues = pca.explained_variance_  # Eigenvalues (Variance explained by each PC)

Q = np.linalg.inv(rotated_loadings.T.dot(rotated_loadings))
rotated_stand_scores2 = df_std.dot(V_cent[ : ,:2].dot(np.sqrt(np.diag((S_cent[0:2]**2)/9)))).dot(rotator.rotation_).dot(Q)

(df_std.dot(V_cent[ : ,:2]).var())
df_std.dot(V_cent[ : ,:2].dot(np.sqrt(np.diag((S_cent[0:2]**2)/9)))).std()

# provides rotated scores 
df_std.dot(rotated_loadings).dot(Q)

rotated_stand_scores2.dot(np.sqrt(np.diag((S_cent[0:2]**2)/9))).var()

variance_rotated_scores = np.var(rotated_stand_scores2, axis=0, ddof=1)


from factor_analyzer import FactorAnalyzer

fa = FactorAnalyzer(rotation='varimax', n_factors=2, method='principal')
fa.fit(df_std)
factor_scores = fa.transform(df_std)


# Step 4: Compute the rotated component scores
rotator = Rotator(method="varimax")
rotated_loadings = rotator.fit_transform()

Q = np.linalg.inv(rotated_loadings.T @ rotated_loadings)
rotated_stand_scores = df_std @ rotated_loadings @  Q
variance_rotated_scores = np.var(rotated_stand_scores, axis=0, ddof=1)


og_scores = df_std.dot(V_cent[ : , :2]).dot(np.sqrt(np.diag(eigenvalues)))
np.sqrt(np.var(og_scores, axis=0, ddof=1))


variance_rotated_scores
eigenvalues.sum()

tolerance = 1e-6  # Define a small tolerance
assert np.isclose(variance_rotated_scores.sum(), eigenvalues.sum(), atol=tolerance), "Variance mismatch exceeds tolerance!"

stand_rotated_loadings_rec = df_std.T.dot(rotated_stand_scores)/9

# solve for singular value matrix A=U\SigmaV^T --> \Sigma = U^TXV

# compute PCA scores on varimax-rotated right eigenvectors
rotated_scores = df_std.dot(rotated_right_eig)
loadings3 = df_std.T.dot(rotated_left_eig[: ,:2])/np.sqrt(10-1)

# NOTE: these loadings are not orthogonal
rotated_loadings = rotator.fit_transform(loadings_pca)


# R loadings
# Loadings:
#      RC1   RC2  
# [1,] 0.182 0.980
# [2,]       0.994
# [3,] 0.995      
# [4,] 0.990 0.121


# Factor scores 
#             PC1         PC2
# 1  -0.684045038  0.80633302
# 2  -0.955255479  0.66053067
# 3  -0.005265157  1.55970406
# 4   0.046532621  1.11915539
# 5  -1.818118279 -0.99421276
# 6   0.132505689 -0.98219338
# 7  -0.136335525 -1.35382496
# 8   1.380173552 -0.37560047
# 9   1.400948787  0.09982591
# 10  0.638858829 -0.53971748

# apply factor analysis 
from factor_analyzer import FactorAnalyzer

fa = FactorAnalyzer(rotation="varimax", n_factors=2, method='principal')
fa.fit(df_std)
factor_scores = fa.transform(df_std)

fa.get_eigenvalues()

# Compute PCA scores (with correction)
factor_scores_manual = df_std.dot(rotated_loadings).dot(np.linalg.inv(rotated_loadings.T.dot(rotated_loadings)))

fa.loadings_.T.dot(fa.loadings_)
factor_scores_manual.T.dot(factor_scores_manual)
factor_scores.T.dot(factor_scores)

