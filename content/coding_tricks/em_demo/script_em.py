def nCr(n,r):
    """
    A naive implementation for calculating the combination number C_n^r.

    Args:
        n: int, the total number
        r: int, the number of selected

    Returns:
        C_n^r, the combination number
    """
    f = math.factorial
    return f(n) / f(r) / f(n - r)


def binomial(x, n, p):
    """
    The bionomial distribution C_n^x p^x (1-p)^(n-x)

    Args:
        n: int, the total number
        x: int, the number of selected
        p: float, the probability

    Returns:
        The bionomial probability C_n^x p^x (1-p)^(n-x)
    """
    return nCr(n, x) * (p**x) * (1 - p)**(n - x)


def generate_binomial_mixture(num_components, size, p, mix):
  """
  Generate data according to a mixture of binomials.
  Parameters:
      - num_components (int): Number of mixture components
      - size (int or tuple of ints): Size of the generated data
      - p (list or array): List or array of probabilities of success for each binomial distribution
      - mix (list or array): List or array of mixing coefficients for each component
  Returns:
      - data (ndarray): Generated data according to the mixture of binomials
  """
  assert num_components == len(p), "Number of components must match the length of the probability list."
  assert num_components == len(mix), "Number of components must match the length of the mixing coefficient list."
  assert sum(mix) == 1, "Mixture probabilities sum to 1"
  
  # Generate data for each binomial distribution 
  binomial_mixture_data = np.random.binomial(n=1, p=p, size=(size, num_components))
  
  # Compute dot product between mixture values and binomial mixture data. If dot product >= 0.5, value = 1, else, value = 0. 
  binomial_mixture_data = (np.dot(binomial_mixture_data, mix) >= 0.5).astype(int)
  
  return binomial_mixture_data



  

import math
import numpy as np
import pandas as pd
from scipy.stats import binom
  

n = 10       # number of tosses per trial
X = [5, 9, 8, 4, 7]  # observation  
lam = 0.5            # prior
p1 = 0.6             # parameter: pA
p2 = 0.5             # parameter: pB
n_trials = len(X)    # number of trials
n_iters = 10   # number of EM iterations

p = [0.6, 0.5]

#unnormalized responsibilities for each data point for each mixture
unnormalized_responsibilities = [lam * binom.pmf(x, n=10, p= np.array(p)) for x in X]

#normalized probabilites (i.e., responsibilities)
normalized_responsibilities = [rp / np.sum(rp) for rp in unnormalized_responsibilities]


column_names = ['resp_mixture_{}'.format(mix+1) for mix in range(len(normalized_responsibilities[0]))]

df_responsibilities = pd.DataFrame(np.vstack(normalized_responsibilities), 
                                  columns = column_names)

#insert data column as the first one
df_responsibilities.insert(0, 'data', data)  

for i in range(n_iters):
    print(f'==========EM Iter: {i + 1}==========')

    # E-step
    q = np.zeros([n_trials, 2])
    for trial in range(n_trials):
        x = X[trial]
        q[trial, 0] = lam * binom.pmf(x, n, p1)
        q[trial, 1] = (1 - lam) * binom.pmf(x, n, p2)
        q[trial, :] /= np.sum(q[trial, :])

   print('E-step: q(z) = ')
   print(q)

    # M-step
    p1 = sum((np.array(X) / n) * q[:, 0]) / sum(q[:, 0])
    p2 = sum((np.array(X) / n) * q[:, 1]) / sum(q[:, 1])
    #lam = sum(q[:, 0]) / n_trials

    path.append([p1, p2])

    print('M-step: theta = ', p1, p2)

  
    
    
