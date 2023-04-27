---
title: "The Expectation-Maximization Algorithm: A Method for Generating Mixture Models" 
draft: false
summary: 'Upcoming post' 
date: ""
article_type: technical
output:
  bookdown::html_document2:
     keep_md: true
always_allow_html: true
header-includes: 
tags: []
---   






# Introduction to Mixture Models

Consider a situation where a researcher entrusts a colleague to flip coins and record the results of 10 flips. Flips that result in heads are recorded as `1` and flips that result in tails are recorded as `0`. Importantly, before each flip, the colleague picks one of two coins (according to another probability mass function) but does not tell the researcher which coin was flipped. When the colleague records the results of the 10 coin flips, they provide the researcher with the following data: $\mathbf{x} = \[1, 1, 1, 1, 0, 0, 0, 0, 0, 0\]$. With these data, the researcher wants to estimate the following three parameters: 

1) The probability of picking each of the two coins (i.e., a probability mass function), $\mu_1, \mu_2$. Given that both these probability values sum to one, only the probability of picking one coin must be estimated, $\mu_1$, with $\mu_2 = 1 - \mu_1$.
2) The first coin's probability of heads, $p_1$. 
3) The second coin's probability of heads, $p_2$. 

# The Difficulty of Directly Estimating Mixture Models With Maximum Likelihood Estimation 

One way to estimate the parameters ($mu_1$, $p_1$, $p_2$) is to use maximum likelihood estimation (for a review, see my post on [maximum likelihood estimation](https://sebastiansciarra.com/technical_content/mle/)). In maximum likelihood estimation, we solve for the parameter values $\boldsymbol{\theta} = \mu_1, p_1, p_2$ that maximize the probability of observing the data, $P(\mathbf{x}|\boldsymbol{\theta})$. Because the values being maximized are not technically probabilities and are instead likelihoods,$L(\boldsymbol{\theta}|\mathbf{x})$, maximum likelihood estimation solves for the parameter values with the highest likelihood, as shown below in Equation \ref{eq:mle}:

$$ 
\begin{align}
\boldsymbol{\theta_{MLE}} &= \underset{\boldsymbol{\theta}}{\arg\max}  L(\theta|\mathbf{x}).
\label{eq:mle}
\end{align}
$$
In the current example, the likelihood values can be calculated by using Equation \ref{eq:incomplete-data} shown below: 

$$ 
\begin{align}
L(\boldsymbol{\theta}|\mathbf{x}) &= \prod_{n=1}^{10} \sum_{k=1}^{2} \mu_k B(x_n|p_k),
\label{eq:incomplete-data}
\end{align}
$$

where $B(x_n|p_k)$ is the binomial probability of the $n^{th}$ data point given the the $k^{th}$ coin, and this probability is weighted by the corresponding probability of selecting the $k^{th}$ coin. Importantly, because the researcher does not know which of the two coins produces the result of any flip, then any flip could be the result of flipping the first or second coin. To model this uncertainty, the calculation of the likelihood for each coin flip result, $x_n$, computes the sum of weighted binomial probabilities, $\sum^{2}_{k=1} \mu_k B(x_n|p_k)$. The lack of information surrounding the identity of the coin that produces each flip result also explained why Equation \ref{eq:incomplete-data} aboe is often called the *incomplete-data likelihood*.
To prevent *underflow* (the rounding of small numbers to zero in computers), the log-likelihood is taken, resulting in the incomplete-data log-likelihood shown below in Equation \ref{eq:log-incomplete-data}: 

$$ 
\begin{align}
\log L(\boldsymbol{\theta}|\mathbf{x}) &= \sum_{n=1}^{10} \log\Big(\sum_{k=1}^{2} \mu_k B(x_n|p_k) \Big).
\label{eq:log-incomplete-data}
\end{align}
$$

To find maximum likelihood estimates for the parameters, partial derivatives are computed with respect to each parameter ($\mu_1, p_1, p_2$) and then set to equal zero. In computing the partial derivatives of the incomplete-data log-likelihood (Equation \ref{eq:log-incomplete-data} with respect to the parameters, it is important to note that the existence of the summation symbol within the logarithm will oftentimes yield a complex and lengthy derivative because the chain rule has to be applied for each data point. Although computing the partial derivatives does not yield overly complex equations in the current example, solutions (often called closed-form solutions) for the parameters cannot be obtained. As an example, I will show how maximum likelihood estimation would be implemented for estimating the probability of selecting coin 1, $\mu_1$. To compute the likelihood, we can expand the binomial term in the incomplete-data log-likelihood function above (Equation \ref{eq:log-incomplete-data}) to produce 

$$ 
\begin{align}
\log L(\boldsymbol{\theta}|\mathbf{x}) &= \sum_{n = 1}^{10} \log \Big(\sum_{k=1}^{2} \mu_k  {n \choose h}p_k^h (1-p_k)^{n-h} \Big).
\label{eq:log-incomplete-data-expanded}
\end{align}
$$
To allow the partial derivative to be computed, I will apply the binomial calculation on each flip. Thus, $n = 1$ and $h = \{0, 1\}$, which means that ${n \choose h} = 1$. In expanding Equation \ref{eq:log-incomplete-data-expanded} over the summation sign within the logarithm, Equation \ref{eq:log-incomplete-data-binom-expanded} is obtained 

$$ 
\begin{align}
\log L(\boldsymbol{\theta}|\mathbf{x}) &= \sum^{10}_{n = 1} \log \Big( \mu_1 p_1^h (1-p_k)^{n-h} + (1-\mu_1) p_2^h (1-p_2)^{n-h}  \Big).
\label{eq:log-incomplete-data-binom-expanded}
\end{align}
$$
Because $h = \{0, 1\}$ for any given $n$ flip, the term inside the logarithm will only ever take on one of the two following forms: 

$$
\begin{spreadlines}{0.5em}
\begin{align}
    \log L(\boldsymbol{\theta}|\mathbf{x}_i) =
    \begin{cases}
      \text{If } \mathbf{x}_i = h, & \log(\mu_1 p_1 + (1 - \mu_1)p_2), \\\\ \\\\
      \text{If } \mathbf{x}_i = t, & \log(\mu_1(1 - p_1) + (1 - \mu_1)(1 - p_2))
    \end{cases}
    \label{eq:log-terms}
\end{align},
\end{spreadlines}
$$
where $h$ indicates a coin flip that results in 'heads' and `t` indicates a coin flip that results in 'tails' (i.e., $h = 0$). To expand Equation \ref{eq:log-incomplete-data-binom-expanded} over the summation sign outside the logarithm, we can apply Equation \ref{eq:log-terms} and obtain a simplified expression of the incomplete-data log-likelihood shown below in  Equation \ref{eq:incomplete-simplified}: 

$$
\begin{align}
\log L(\boldsymbol{\theta}|\mathbf{x}) &= H\log(\mu_1 p_1 + (1 - \mu_1)p_2) + T\log(\mu_1(1 - p_1) + (1 - \mu_1)(1 - p_2)), 
 \label{eq:incomplete-simplified}
\end{align}
$$
where $H$ indicates the total number of heads and $T$ indicates the total number of tails. In the current data set, $\mathbf{x} = \[1, 1, 1, 1, 0, 0, 0, 0, 0, 0\]$, four heads and six tails are obtained. Although we can compute the partial derivative of Equation \ref{eq:incomplete-simplified} with respect to $\mu_1$ and obtain a closed-form solution, Equation \ref{eq:unsolvable-mu1} shows that it is inadmissible because it always yields negative values for $\mu_1$. 


$$ 
\begin{spreadlines}{0.5em}
\begin{align}
\log L(\boldsymbol{\theta}|\mathbf{x}) &= 4\log(\mu_1 p_1 + (1 - \mu_1)p_2) + 6\log(\mu_1(1 - p_1) + (1 - \mu_1)(1 - p_2))  \nonumber \\\\
\frac{\partial L(\boldsymbol{\theta}|\mathbf{x})}{\partial \mu_1} &= \frac{\partial}{\partial \mu_1} \Big( 4\log(\mu_1 p_1 + (1 - \mu_1)p_2) + 6\log(\mu_1(1 - p_1) + (1 - \mu_1)(1 - p_2))\Big) \nonumber \\\\
&= \frac{4(p_1 - p_2)}{\mu_1 p_1 + (1 - \mu_1)p_2} + \frac{6(p_2 - p_1)}{\mu_1(1 - p_1) + (1 - \mu_1)(1 - p_2)} \nonumber \\\\
\text{Set } \frac{\partial L(\boldsymbol{\theta}|\mathbf{x})}{\partial \mu_1}& = 0  \nonumber \\\\
0 &= \frac{4(p_1 - p_2)}{\mu_1 p_1 + (1 - \mu_1)p_2} + \frac{6(p_2 - p_1)}{\mu_1(1 - p_1) + (1 - \mu_1)(1 - p_2)} \nonumber \\\\
&= \frac{4(p_1 - p_2)}{\mu_1 p_1 - \mu_1p_2 + p_2} + \frac{6(p_2 - p_1)}{\mu_1 - \mu_1p_1 + 1 - \mu_1 + \mu_1p_2 - p_2} \nonumber \\\\
&= \frac{4(p_1 - p_2)}{\mu_1(p_1 - p_2) + p_2} + \frac{6(p_2 - p_1)}{ \mu_1(p_2 - p_1) + 1 - p_2} \nonumber \\\\
&= \frac{4}{\mu_1 + p_2} + \frac{6}{ \mu_1 + 1 - p_2} \nonumber \\\\
&= 4(\mu_1 + 1 - p_2) + 6(\mu_1 + p_2) \nonumber \\\\
&= 10\mu_1 + 2p_2 + 4 \nonumber \\\\
\mu_1 = \frac{-p_2 - 2}{5} 
\label{eq:unsolvable-mu1}
\end{align}
\end{spreadlines}
$$
Therefore, although the summation symbol within the logarithm does not result in an overly complex partial derivative in the current example, maximum likelihood estimation results in inadmissible estimates for parameter values, and so is not a viable method for modelling mixture distributions.[^1]

[^1]: It should also be noted that maximum likelihood estimation can result in singularities with Gaussian mixture models. In short, if the estimate for a $k^{th}$ mixture's mean, $\mu_k$, happens to exactly match the value of an individual $n$ value, $x_n$, then the mixture in question can become 'stuck' on this data point with all the other data points being modelled by the other mixtures. With the mixture fixed on the one data point, the variance of the mixture will decrease to zero and the log-likelihood will increase to infinity. 



# Indirectly Estimating Mixture Models With the Expectation-Maximization (EM) Algorithm

Unlike maximum likelihood estimation, the expectation-maximization (EM) algorithm provides viable parameter estimates for modelling mixture distributions. The EM algorithm works because it indirectly maximizes the incomplete-data log-likelihood; that is, it does not directly operate on the incomplete-data log-likelihood. To act as an indirect estimation method, the EM algorithm begins by modelling the uncertainty of the coin's identity on each flip as a *latent variable*: A variable that is assumed to exist but has not been directly measured, whether by choice or because direct measurement is impossible. To model the coin's identity on each $n$ flip as a latent variable, one-hot (or 1-of-*K*) encoding is used. In one-hot encoding, the levels of a categorical variable can be represented numerically with a binary vector that sums to one and a length equal to the number of levels in the categorical variable. In the current example, the categorical variable is the coin's identity, and the two levels (i.e., coin 1, coin 2) can be represented in each $n$ flip by $\mathbf{z_n}$, as shown below in Equation \ref{eq:one-hot} below: 

$$
\begin{align}
    \mathbf{z_n} =
    \begin{cases}
      \text{If coin 1 }, & \[1, 0\] \\\\ \\\\
      \text{If coin 2 }, & \[0, 1\]
    \end{cases}
    \label{eq:one-hot}
\end{align}.
$$
By modelling the coin's status as a latent variable with one-hot encoding, the incomplete-date likelihood (Equation \ref{eq:incomplete-data} can be modified to produce the *complete-data likelihood* shown below in Equation \ref{eq:complete-data}:

$$ 
\begin{align}
L(\boldsymbol{\theta}|\mathbf{x}, \mathbf{z}) &= \prod_{n=1}^{10} \prod_{k=1}^{2} \mu_k B(x_n|p_k)^{z_{nk}},
  \label{eq:complete-data}
\end{align}
$$
where $k$ is used to index each value of $z_n$. Note that, because the likelihood of each $x_n$ data point is raised to an exponent value of either 0 or 1, we now take the product over the *K* classes so that likelihood computation is not affected by individual likelihood values of data points that do not belong to the mixture in question. Relatedly, Equation \ref{eq:complete-data} is called the complete-data likelihood because it can only be computed if we know the mixture membership of each $x_n$ data point; that is, we must have the complete data. As before, the logarithm of the complete-data likelihood is taken to prevent underflow, resulting in the complete-data log-likelihood shown below in Equation \ref{eq:log-complete-data}: 

$$ 
\begin{align}
\log L(\boldsymbol{\theta}|\mathbf{x}, \mathbf{z}) &= \sum_{n=1}^{10} \sum_{k=1}^{2} z_{nk}\Big(\log(\mu_k) + \log\big(B(x_n|p_k)\big)\Big).
  \label{eq:log-complete-data}
\end{align}
$$
In looking at the complete-data log-likelihood, two points must be made: one good and one bad. Beginning with the good, one desirable outcome of the complete-data log-likelihood is that the summation over *K* is now not inside a logarithm, which means that the partial derivatives will be much less complex and yield admissible solutions. Ending with the bad, the complete-data log-likelihood cannot be computed because the latent variables, $z_n$, are unknown; recall that the researcher does not know which coin produced the result of any flip. 

Although the complete-data log-likelihood cannot be directly computed, the EM algorithm finds a clever way to circumvent this problem in the E step. 

## Expectation (E) Step: Using Expectations to Estimate the Distribution of the Latent Variable

To understand how the EM algorithm manages to work with the complete-data log-likelihood without being able to compute it, we must first understand the equivalence between the complete- and incomplete-data log-likelihood. That is, the complete- and incomplete-data log-likelihood are the same function, which is shown below in Equation \ref{eq:complete-incomplete}: 


$$
\begin{spreadlines}{0.5em}
\begin{align}
\log L(\boldsymbol{\theta}|\mathbf{x}) &=  \log \Big( \sum_{\mathbf{z}}L(\boldsymbol{\theta}|\mathbf{x}, \mathbf{z} \Big) \nonumber \\\\
&=\log \Big(\sum_{\mathbf{z}}  \prod_{n=1}^{10} \prod_{k=1}^{2} \mu_k B(x_n|p_k)^{z_{nk}} \Big) \nonumber \\\\
&= \log \Big(\mu_1 B(x_1|p_1)^1 \cdot \mu_2 B(x_1|p_2)^0 \cdot ... \cdot \mu_1 B(x_{10}|p_1)^1 \cdot \mu_2 B(x_{10}|p_2)^0 + \nonumber \\\\
 &\qquad\quad\text{ }\text{ } \mu_1 B(x_1|p_1)^0 \cdot \mu_2 B(x_1|p_2)^1 \cdot ... \cdot \mu_1 B(x_{10}|p_1)^0 \cdot \mu_2 B(x_{10}|p_2)^1 \Big) \nonumber \\\\ 
 &= \log \Big(\prod_{n=1}^{10} \sum_{k=1}^{2} \mu_k B(x_n|p_k)\Big)  \nonumber \\\\
 &= \sum_{n=1}^{10} \log \Big(\sum_{k=1}^2 \mu_k B(x_n|p_k) \Big).
 \label{eq:complete-incomplete}
\end{align}
\end{spreadlines}
$$

Thus, in summing over all $z$ possible latent variable representations for each $n$ data point, the incomplete-data log-likelihood becomes the complete-data log-likelihood. As an aside, in computing the sum over $z$, we are marginalizing over $z$, which explains why the incomplete-data log-likelihood is often called the *marginal log-likelihood* and why the complete-data log-likelihood is often called the *joint log-likelihood*.

In showing the equivalence between the incomplete- and complete-data log likelihood, the E step applies two clever tricks to work with the complete-data log-likelihood. First, it applies what appears to be an inconsequential algebraic manipulation, whereby the complete-data log-likelihood is multiplied and divided by some distribution on the latent variable $q(\mathbf{z})$ as shown below in Equation \ref{eq:variation-method}:

$$
\begin{spreadlines}{0.5em}
\begin{align}
\log L(\boldsymbol{\theta}|\mathbf{x}) &= \log \sum_{\mathbf{z}} L(\boldsymbol{\theta}|\mathbf{x},\mathbf{z}) \nonumber \\\\
&=\log \sum_{\mathbf{z}} q(\mathbf{z})\frac{L(\boldsymbol{\theta}|\mathbf{x},\mathbf{z})}{q(\mathbf{z})}.
\label{eq:variation-method}
\end{align}
\end{spreadlines}
$$

Second, given that the logarithm is a concave function, Jensen's inequality can be applied to convert Equation \ref{eq:variation-method} to an inequality. Briefly, Jensen's inequality states that, for concave functions, the function of the expected value, $f(\mathbb{E}\[x\])$), is greater than or equal to the expected value of the function, $\mathbb{E}\[f(x)\]$ (for a proof, see [Appendix A](#jensen)), and is shown below in Equation \ref{eq:jensen-concave}:

$$
\begin{align}
f(\mathbb{E}\[x\]) \ge \mathbb{E}\[f(x)\].
\label{eq:jensen-concave}
\end{align}
$$
In other words, Equation \ref{eq:jensen-concave} above shows that $\mathbb{E}\[f(x)\]$ is a lower bound on $f(\mathbb{E}\[x\])$. Applying Jensen's inequality to Equation \ref{eq:variation-method}, a lower bound for the incomplete-data log-likelihood is obtained, $\mathcal{L}(q, \boldsymbol{\theta})$, resulting in the inequality shown below in Equation \ref{eq:variation-inequality}: 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\log L(\boldsymbol{\theta}|\mathbf{x}) &\ge \sum_{\mathbf{z}} q(\mathbf{z})\log \Bigg(\frac{L(\boldsymbol{\theta}|\mathbf{x},\mathbf{z})}{q(\mathbf{z})}\Bigg) \nonumber \\\\
&\ge \mathbb{E}_{q(\mathbf{z})}\log \Bigg(\frac{L(\boldsymbol{\theta}|\mathbf{x},\mathbf{z})}{q(\mathbf{z})}\Bigg) = \mathcal{L}(q, \boldsymbol{\theta}),  
\label{eq:variation-inequality}
\end{align}
\end{spreadlines}
$$

where the summation over $z$ of $q(\mathbf{z})$ is the expected value of $q(\mathbf{z})$, and can be represented as $\mathbb{E}_{q(\mathbf{z})}$. As a note, the lower bound,  $\mathcal{L}(q, \boldsymbol{\theta})$, is often called the *evidence lower bound* because it is a lower bound on the marginal log-likelihood, which is often called the evidence in Bayesian inference. 

Although we still do not have a way for computing $q(\mathbf{z})$, a closer inspection of the above inequality provides a way forward. In the E step, it is in our best interest to obtain the most accurate approximation of the distribution $q(\mathbf{z})$, because, as will become clear in the M step, doing so will result in the greatest improvements in the estimates for the parameters, $\boldsymbol{\theta} = \[\mu_1, p_1, p_2\]$. To obtain the best estimate of $q(\mathbf{z})$, and thus maximize the potential for improvement in the parameter estimates in the M step, we need to maximize the lower bound with respect to $q(\mathbf{z})$ and transform the inequality of Equation \ref{eq:variation-inequality} into an equality. To do so, we can compute $q(\mathbf{z})$ such that the logarithm, $\log \Big(\frac{L(\boldsymbol{\theta}|\mathbf{x}, \mathbf{y})}{q(\mathbf{z})}\Big)$, returns constant values, and this can be accomplished if the probability values computed for the latent variable $\mathbf{z}$ are proportional to the numerator, $q(\mathbf{z}) \propto L(\boldsymbol{\theta}|\mathbf{x},\mathbf{y})$. Bayes' theorem provides one way for us to compute $q(\mathbf{z})$ to maximize the lower bound such that 

$$
\begin{spreadlines}{0.5em}
\begin{align}
q(\mathbf{z})= P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) &= \frac{P(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta})P(\mathbf{z}|\boldsymbol{\theta})}{ P(\mathbf{x}|\boldsymbol{\theta})} 
\label{eq:bayes} \\\\
&= \frac{P(\mathbf{z}, \mathbf{x}|\boldsymbol{\theta})}{\sum_{z\prime}P(\mathbf{z\prime}, \mathbf{x}|\boldsymbol{\theta})} \nonumber \\\\
&= \frac{L(\boldsymbol{\theta}|\mathbf{z}, \mathbf{x})}{\sum_{z\prime}L(\boldsymbol{\theta}|\mathbf{z\prime}, \mathbf{x})}
\label{eq:posterior}
\end{align}
\end{spreadlines}
$$

where I have used likelihood notation in Equation \ref{eq:posterior} to highlight the equivalence with the calculation of probabilities and likelihoods. It is important to note that, because latent variable memberships exist for each $n$ data point for each $k$ mixture in the complete-data log-likelihood (Equation \ref{eq:log-complete-data}), Equation \ref{eq:posterior} above is computed for each $n$ data point such that 


$$
\begin{align}
P(z_{nk} |x_n, \boldsymbol{\theta} = \[\mu_k, p_k\]) &= \gamma(z_{nk}) = \frac{\mu_k B(x_n|p_k)}{\sum_k^2 \mu_k B(x_n|p_k)}.
\label{eq:ind-posterior}
\end{align}
$$

Note that the responsibilities are often represented as $\gamma(z_{nk})$, which is simply the scalar form of  $\mathbb{E}_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})}$ that I use throughout this post.


Because these values represent the (posterior) probability of membership to each $k$ mixture, they are often called *responsibilities*. Therefore, by setting $q(\mathbf{z})= P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$, we can compute $q(\mathbf{z})$ and also obtain a lower bound, $\mathcal{L}(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}), \boldsymbol{\theta})$ that is equal to the incomplete-data log-likelihood. Using Equation \ref{eq:bayes}, we can rewrite the inequality of Equation \ref{eq:variation-inequality} as an equality in Equation \ref{eq:post-variation-inequality} below:

$$
\begin{spreadlines}{0.5em}
\begin{align}
\log L(\boldsymbol{\theta}|\mathbf{x}) &=   \mathcal{L}\Big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}),\boldsymbol{\theta}\Big) \label{eq:equality} \\\\
&= \mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})}\log \Bigg(\frac{L(\boldsymbol{\theta}|\mathbf{x},\mathbf{z})}{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})}\Bigg)
\label{eq:post-variation-inequality} \\\\
\end{align}
\end{spreadlines}
$$

To show that the lower bound is equal to the incomplete-data log-likelihood when $q(\mathbf{z})= P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$ (see Equation \ref{eq:equality}), I provide the Python code block below. In order to better understand the code, I provide the following expansions of the incomplete-data log-likelihood and the lower bound, Equations \ref{eq:log-incomplete-data} and \ref{eq:log-incomplete-data}. Importantly, and as I will discuss later on in this post, the first term in Equation \ref{eq:lower-bound-exp} is the expected complete-data log-likelihood, and the second term is the entropy of the responsibilities. Recall that the researcher's data set is $\mathbf{x} = \[1, 1, 1, 1, 0, 0, 0, 0, 0, 0\]$.

$$
\begin{spreadlines}{0.5em}
\begin{align}
\log L(\boldsymbol{\theta}|\mathbf{x}) &= \sum_{n=1}^{10} \log \Big(\sum_{k=1}^2 \mu_k B(x_n|\mu_k)\Big) 
\tag{\ref{eq:log-incomplete-data} revisited} \\\\
\mathcal{L}\big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}), \boldsymbol{\theta})\big) &=  \underbrace{\mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})}\log (L(\boldsymbol{\theta}|\mathbf{x},\mathbf{z}))}\_{\text{Expected complete-data log-likelihood}} \phantom{e x} \underbrace{-\mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})} \log({P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})})}\_{\text{Entropy}} \label{eq:lower-bound}\\\\
&= \sum_{n=1}^{10} \sum_{k=1}^2\gamma(z_{nk})\big(\log(\mu_k) + x_n\log(\mu_k) + (1 - x_n)\log(1 - \mu_k)\big) - \gamma(z_{nk})\log\big(\gamma(z_{nk})\big)
\label{eq:lower-bound-exp}
\end{align}
\end{spreadlines}
$$



```r {language=python}
import numpy as np
import pandas as pd
from scipy.stats import binom

def e_step(data, mu, p):
  """
  Compute expectations (i.e., responsibilities) for each data point's membership to each mixture
  Parameters:
      - data: data set 
      - mu: Probability of each component 
      - p: Probabilities of success for each binomial distribution
  Returns:
      - pandas dataframe
  """
    
  assert len(mu) == len(p), "Number of estimates in mu is equal to the number of sucsess probabilities"
  assert sum(mu) == 1, "Sum of mu should be equal to 1"
  
  #unnormalized responsibilities for each data point for each mixture
  unnormalized_responsibilities = [mu * binom.pmf(x, n=1, p= np.array(p)) for x in data]
  
  #normalized probabilites (i.e., responsibilities)
  normalized_responsibilities = [rp / np.sum(rp) for rp in unnormalized_responsibilities]
  
  column_names = ['resp_mixture_{}'.format(mix+1) for mix in range(len(normalized_responsibilities[0]))]

  df_responsibilities = pd.DataFrame(np.vstack(normalized_responsibilities), 
                                    columns = column_names)
  
  #insert data column as the first one
  df_responsibilities.insert(0, 'data', data)                

  return(df_responsibilities)


#incomplete/complete-data log-likelihood
def compute_incomplete_log_like(data, mu, p):
  
  #probability of each data point coming from each distribution
  mixture_sums = [np.sum(mu * binom.pmf(flip_result, n=1, p= np.array(p))) for flip_result in binom_mixture_data]
  
  #log of mixture_sums
  log_mixture_sums = np.log(mixture_sums)
  
  #sums of log of mixture_sums
  incomplete_like = np.sum(log_mixture_sums)

  return(incomplete_like)

#lower bound
def compute_lower_bound(responsibilities, mu, p):
  
  #expected complete-data log-likelihood 
  expected_complete_data_like = responsibilities.apply(compute_expected_complete_like, axis=1).sum()

  ##extract responsibility columns and then compute entropy, x*np.log(x), for each cell value
  resp_colummns = responsibilities.filter(like = 'resp_mixture')
  entropy = -np.sum(resp_colummns.values * np.log(resp_colummns.values))

  return(expected_complete_data_like + entropy)

#expected complete-data log-likelihood
def compute_expected_complete_like(row):
  resp_columns = [col for col in row.index if 'resp_mixture' in col]
  resp_values = [row[col] for col in resp_columns]
  
  return np.sum(
      [resp_values * (np.log(mu) + 
      row['data'] * np.log(p) + #non-zero if flip result is success (i.e., 'heads')
      (1 - row['data']) * np.log(1 - np.array(p)) #non-zero if flip result is failure (i.e., 'tails')
      )]
  )
    
#data given to researcher
binom_mixture_data = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

#initial guesses for E step 
mu = [0.3, 0.7] #mixture probabilities 
p = [0.6, 0.8] #success probabilities

print('Incomplete-data log-likelihood:', np.round(compute_incomplete_log_like(data = binom_mixture_data, mu = mu, p = p), 5))
responsibilities = e_step(data = binom_mixture_data, mu = mu, p = p)
print('Lower bound:', np.round(compute_lower_bound(responsibilities = responsibilities,  mu = mu, p = p), 5))'
```

<pre><code class='python-code'>Incomplete-data log-likelihood =  -9.28686
Lower bound: -9.28686
</code></pre>

Therefore, by introducing a distribution of the latent variable and taking the expectation of the complete-data log-likelihood, a lower bound of the complete-data log-likelihood is obtained. The lower bound is then optimized with respect to the distribution of the latent variable by computing probabilities of mixture membership. 

## Maximization (M) Step: Using Expectations to Obtain New Parameter Estimates

The maximization (M) step uses the responsibilities obtained in the E step to compute new parameter estimates for $\boldsymbol{\theta}$. As in the E step, the M step also indirectly optimizes the incomplete-data log-likelihood by optimizing the lower bound. In the M step, however, instead of optimizing the lower bound with respect to $q(\mathbf{z})$, the lower bound is optimized with respect to the parameters, $\boldsymbol{\theta}$, resulting in new parameter estimates. Because new parameter estimates are obtained in the M step, I will represent them with $\boldsymbol{\theta}^{new}$ and the old estimates with $\boldsymbol{\theta}^{old}$. Thus, in optimizing the lower bound with respect to the parameter values in the M step, we can say the lower bound is optimized with respect to $\boldsymbol{\theta}^{old}$. In optimizing the lower bound with respect to $\boldsymbol{\theta}^{old}$, the incomplete-data log-likelihood is obtained with new parameter values, $\log L(\boldsymbol{\theta}^{new}|\mathbf{x})$, and increases by at least as much as the lower bound increases when optimized with respect to $\boldsymbol{\theta}^{old}$, as shown below in Equation \ref{eq:theta-optimize}:

$$
\begin{align}
\log L(\boldsymbol{\theta}^{new}|\mathbf{x}) - \log L(\boldsymbol{\theta}^{old}|\mathbf{x})  &\ge \mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})} \log \Big(L(\boldsymbol{\theta}^{new}|\mathbf{x},\mathbf{z})\Big)  - \mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})} \log \Big(L(\boldsymbol{\theta}^{old}|\mathbf{x},\mathbf{z})\Big).
\label{eq:theta-optimize}
\end{align}
$$

To understand the logic behind Equation \ref{eq:theta-optimize} above, a brief discussion of entropy, cross-entropy, and the Kullback-Liebler (KL) divergence is necessary and I provide an overview in the following section. 

### A Brief Review of Entropy and the Kullback-Liebler (KL) Divergence

Consider an example where an analyst for a wine merchant records wine preferences among customers over a two-year period. Specifically, the analyst asks customers for their favourite type of wine from red, white, and sparkling. Figure \ref{fig:wine-year} shows the customers' preferences in each year.



<div class="figure">
  <div class="figDivLabel">
    <caption>
      <span class = 'figLabel'>Figure \ref{fig:wine-year}<span> 
    </caption>
  </div>
   <div class="figTitle">
    <span>Favourite Wine Types by Customers in Each of Two Years</span>
  </div>
    <img src="images/wine_plot.png" width="90%" height="90%"> 
  
  <div class="figNote">
  </div>
</div>


To quantify the difference between the probability distributions with a single value, the analyst uses the Kullback-Liebler (KL) divergence shown below in Equation \ref{eq:kl-divergence}


$$
\begin{align}
KL(P\\|Q) &= \sum_{\mathbf{x}} P(\mathbf{x}) \log\Bigg(\frac{P(\mathbf{x})}{Q(\mathbf{x})}\Bigg).
\label{eq:kl-divergence}
\end{align}
$$
To understand the KL divergence, it is helpful to understand each of its three computations (for an excellent explanation, see [KL divergence](https://www.youtube.com/watch?v=q0AkK8aYbLY&t=173s)) that are presented below: 

1) $\frac{P(\mathbf{x})}{Q(\mathbf{x})}$: measures change in each wine type relative to Year 1. 
2) $\log$: gives equal weightings to reciprocals. As an example, consider the change in preferences across the two years for red and white wine. Across the two years, the preference for red wine across increases from 20% to 40%, whereas the preference for white wine decreases from 40% to 20%. Given that these changes are exactly the same, they should contribute the same amount to the total difference between the years. Using logarithm accomplishes this goal; whereas $\frac{2}{4} \neq \frac{4}{2}$, $\log(\frac{0.4}{0.2}) = \log(\frac{0.2}{0.4})$. 
3) $P(\mathbf{x})$: each value of $\mathbf{x}$ is weighed by its current probability (i.e., Year 2). 

Thus, the KL divergence measures the difference between two probability distributions, $P(\mathbf{x})$ and $Q(\mathbf{x})$, by computing the sum of weighted logarithmic ratios. Intuitively, if the two distributions are the same, the KL divergence is zero, and if the distributions are different, the KL divergence is positive. Therefore, the KL divergence is always non-negative, $KL \ge 0$ (for a proof, see [Appendix B](#kl-divergence)). 

To understand why the KL divergence is always non-negative, it is important to understand entropy and cross-entropy (for an excellent explanation, see [entropy & cross-entropy](https://www.youtube.com/watch?v=ErfnhcEV1O8&t=376s)). If we expand the KL divergence expression of Equation \ref{eq:kl-divergence}, we obtain 

$$
\begin{align}
KL(P\\|Q) &= \underbrace{\sum_{\mathbf{x}} P(\mathbf{x}) \log (P(\mathbf{x}))}\_{\text{(Negative) Entropy}}\text{ }\text{ }  \underbrace{-\sum_{\mathbf{x}} P(\mathbf{x}) \log (Q(\mathbf{x}))}\_{\text{Cross-entropy}}.
\label{eq:kl-divergence-exp}
\end{align}
$$

The first term of Equation \ref{eq:kl-divergence-exp} represents *entropy*,[^2] which can be conceptualized as the amount of information or surprise obtained for a given $x$ wine type from the distribution in Year 2, $P(\mathbf{x})$, when encoding it by itself. 

[^2]: The first value in Equation \ref{eq:kl-divergence-exp} is technically the negative entropy. Because entropy (and cross-entropy) compute information/surprise, it makes conceptual sense to represent them as positive values. Unfortunately, the term $\sum_{\mathbf{x}} P(\mathbf{x}) \log(P(\mathbf{x}))$ returns negative values. To reflect the conceptualization that entropy computes information, a negative sign is included to multiply the negative value returned by $\sum_{\mathbf{x}} P(\mathbf{x}) \log(P(\mathbf{x}))$ into a positive value. 

The second term of Equation \ref{eq:kl-divergence-exp} represents *cross-entropy*, which can be conceptualized as the amount of information or surprise obtained for a given $x$ wine type from the distribution in Year 1, $Q(\mathbf{x})$, when encoded by the distribution in Year 2, $P(\mathbf{x})$. Because the distributions in each year are different, it is intuitive to think that cross-entropy is greater than the entropy; that is, it should be more surprising to encode values of one distribution, $P(\mathbf{x})$, with values of another distribution, $Q(\mathbf{x})$, than with values of the same distribution.  The conceptualization that cross-entropy is greater than entropy is formally represented by Gibbs' inequality (for a proof, see [Appendix C](#gibbs)) below: 

$$
\begin{align} 
-\sum_{\mathbf{x}}  P(\mathbf{x}) \log (Q(\mathbf{x})) \ge  -\sum_{\mathbf{x}}  P(\mathbf{x}) \log (P(\mathbf{x})).
\end{align}
$$
Using Gibbs' inequality, it then becomes clear looking at the expression in Equation \ref{eq:kl-divergence-exp} that the KL divergence will always be non-negative because the larger value of the cross-entropy is added to negative entropy, which has a smaller value. 

### Computing New Parameter Estimates Increases The Incomplete Log-Likelihood More Than the Evidence Lower Bound

Returning to the M step, the incomplete-data log-likelihood increases by at least as much as the evidence lower bound increases. The inequality between the increase in the incomplete-data log-likelihood and the evidence lower bound after the optimization of the lower bound with respect to $\boldsymbol{\theta}^{old}$ occurs because the optimization results in a larger-value cross-entropy term, $\mathbb{E}_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})}\log(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{new}))$. Importantly, the cross-entropy term is absorbed by the incomplete-data log-likelihood and not the evidence lower bound. To understand how, after the M step, the increase in the incomplete-data log-likelihood is at least the increase in the evidence lower bound, I will prove the following two points: 

Point 1: After the M step, the evidence lower bound only increases as much as the expected complete-data log-likelihood.
Point 2: After the M step, the incomplete-data log-likelihood increases by as much as the evidence lower bound and the cross-entropy of the new responsibilities, $P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{new})$, with respect to the old responsibilities, $P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})$. 

#### Point 1: The Increase in the Evidence Lower Bound is Equal to the Increase in the Expected Complete-Data Log-Likelihood

In computing new parameter estimates in the M step, $\boldsymbol{\theta}^{new}$, the evidence lower bound increases by the amount that the expected complete-data log-likelihood increases. To show this, I repeat the function for the evidence lower bound below (Equation \ref{eq:lower-bound}) and set $\boldsymbol{\theta} = \boldsymbol{\theta}^{old}$ to keep track of the iteration index.  


$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathcal{L}\big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}), \boldsymbol{\theta}^{old})\big) &=  \underbrace{\mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})}\log (L(\boldsymbol{\theta}^{old}|\mathbf{x},\mathbf{z}))}\_{\text{Expected complete-data log-likelihood}} \phantom{e x} \underbrace{-\mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})} \log({P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})})}\_{\text{Entropy}} 
\tag{\ref{eq:lower-bound}$\phantom{i}$ revisited}
\end{align}
\end{spreadlines}
$$

Below, I show that, when determining the parameters values, $\boldsymbol{\theta}^{old} = \[\mu_k^{old}, p_k^{old}\]$, that maximize the evidence lower bound, the entropy term does not contribute to the derivative (Equation \ref{eq:lower-bound-max}), and so the maximization is equivalent to maximizing the expected data complete-data log-likelihood (Equation \ref{eq:lower-bound-expected}). 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\underset{\boldsymbol{\theta}^{old} = \[\mu_k^{old}, p_k^{old}\]}{\arg \max} \mathcal{L}\big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}), \boldsymbol{\theta}^{old})\big) &=  \underset{\boldsymbol{\theta}^{old} = \[\mu_k^{old}, p_k^{old}\]}{\arg \max}  \Big(\sum_{n=1}^{10} \sum_{k=1}^2\gamma(z_{nk}^{old})\big(\log(\mu_k^{old}) + x_n\log(p_k^{old}) + (1 - x_n)\log(1 - p_k^{old})\big) \phantom{e x} -\underbrace{\gamma(z_{nk}^{old})\log\big(\gamma(z_{nk}^{old}\big)}\_{\text{=0 (i.e., constant)}}\Big) 
\label{eq:lower-bound-max} \\\\
&= \sum_{n=1}^{10} \sum_{k=1}^2\gamma(z_{nk}^{old})\big(\log(\mu_k^{new}) +  x_n\log(p_k^{new}) + (1 - x_n)\log(1 - p_k^{new})  \\\\
\underset{\boldsymbol{\theta}^{old} = \[\mu_k^{old}, p_k^{old}\]}{\arg \max} \mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})}\log (L(\boldsymbol{\theta}^{old}|\mathbf{x},\mathbf{z}))  &= \mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})}\log (L(\boldsymbol{\theta}^{new}|\mathbf{x},\mathbf{z})) 
\label{eq:lower-bound-expected}
\end{align}
\end{spreadlines}
$$

Therefore, optimizing the evidence lower bound with respect to $\boldsymbol{\theta}^{old}$ is equivalent to maximizing the expected complete-data log-likelihood with respect to $\boldsymbol{\theta}^{old}$. Importantly, to compute the value of the lower bound after it has been optimized with respect to $\boldsymbol{\theta}^{old}$, $\mathcal{L}\big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}), \boldsymbol{\theta}^{new})$, the entropy of $P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})$ is included, as shown below in Equation \ref{eq:lower-bound-m}.

$$
\begin{align}
\max \mathcal{L}\big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}), \boldsymbol{\theta}^{old}) &=\mathcal{L}\big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}), \boldsymbol{\theta}^{new}) = \mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})}\log L\big(\boldsymbol{\theta}^{new}|\mathbf{x},\mathbf{z})\big) - \mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})} \log({P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})}) 
\label{eq:lower-bound-m}
\end{align}
$$

To more concisely represent the difference between the new and old evidence lower bounds, I auxiliary function notation in Equation \ref{eq:auxiliary}, where $Q(\boldsymbol{\theta}^{new}|\boldsymbol{\theta}^{old})$ is the new lower bound and $Q(\boldsymbol{\theta}^{old}|\boldsymbol{\theta}^{old})$ is the old lower bound. 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathcal{L}\big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}), \boldsymbol{\theta}^{new}) - \mathcal{L}\big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}), \boldsymbol{\theta}^{old}) &= \mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})}\log L\big(\boldsymbol{\theta}^{new}|\mathbf{x},\mathbf{z})\big) - \mathbb{E}\_{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})}\log L\big(\boldsymbol{\theta}^{old}|\mathbf{x},\mathbf{z})\big) \nonumber \\\\
&= Q(\boldsymbol{\theta}^{new}|\boldsymbol{\theta}^{old}) - Q(\boldsymbol{\theta}^{old}|\boldsymbol{\theta}^{old})
\label{eq:auxiliary}
\end{align}
\end{spreadlines}
$$

#### Point 2: The Incomplete-Data Log-Likelihood Increases by at Least as Much as the Evidence Lower Bound

In computing new parameter estimates in the M step, $\boldsymbol{\theta}^{new}$, the incomplete-data log-likelihood increases by at least as much as the evidence lower bound increases. To show the inequality between the increase in the incomplete-data log-likelihood and the evidence lower bound, I first show below that the incomplete-data log-likelihood can be decomposed as the sum of the lower bound and a KL divergence (see Equation \ref{eq:e-step-kl}. Importantly, I use probability notation in the beginning to highlight that $L(\boldsymbol{\theta}|\mathbf{x}, \mathbf{z}) = P(\mathbf{x}, \mathbf{z}|\boldsymbol{\theta})$, which can be decomposed into $P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) P(\mathbf{x}|\boldsymbol{\theta}) = P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})L(\boldsymbol{\theta}|\mathbf{x})$. I denote $P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$ with probability notation because it is a true probability distribution, but I denote $P(\mathbf{x}|\boldsymbol{\theta})$ with likelihood notation, $L(\boldsymbol{\theta}|\mathbf{x})$, because, when fixing the data and varying the parameters, values become likelihoods (see my previous post on [likelihood and probability](https://sebastiansciarra.com/technical_content/mle/)). As mentioned before, because the M step computes new parameter estimates, $\boldsymbol{\theta}^{new}$, I distinguish them from the current estimates, $\boldsymbol{\theta}^{old}$. 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathcal{L}(q, \boldsymbol{\theta}^{old}) &= \sum_{\mathbf{z}} q(\mathbf{z}) \log \Bigg(\frac{P(\mathbf{x}, \mathbf{z}|\boldsymbol{\theta}^{old})}{q(\mathbf{z})}\Bigg) \nonumber \\\\
&=\sum_{\mathbf{z}} q(\mathbf{z})  \log \Bigg(\frac{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}) P(\mathbf{x}|\boldsymbol{\theta}^{old})}{q(\mathbf{z})}\Bigg)  \nonumber \\\\
&= \underbrace{\sum_{\mathbf{z}} q(\mathbf{z})  \log \Bigg(\frac{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})}{q(\mathbf{z})}\Bigg)}\_{\text{-ve (reverse) KL divergence}} +  {\underbrace{\vphantom{\Bigg(} \sum_{\mathbf{z}} q(\mathbf{z})}\_{=1}} \log \big(P(\mathbf{x}|\boldsymbol{\theta}^{old}) \big)\nonumber \\\\
&= -\underbrace{\mathbb{E}\_{q(\mathbf{z})} \log \Bigg(\frac{q(\mathbf{z})}{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})}\Bigg)}\_{\text{KL divergence}}  + {\underbrace{\vphantom{\Bigg(} \log \big(P(\mathbf{x}|\boldsymbol{\theta}^{old}) \big)}\_{\text{Incomplete-data log-likelihood}}} \nonumber \\\\
\log \big(L(\boldsymbol{\theta}^{old}|\mathbf{x}) \big) &= \mathcal{L}(q, \boldsymbol{\theta}^{old}) +\underbrace{\mathbb{E}\_{q(\mathbf{z})} \log \Bigg(\frac{q(\mathbf{z})}{P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})}\Bigg)}\_{\text{KL divergence}}   \nonumber \\\\
\log \big(L(\boldsymbol{\theta}^{old}|\mathbf{x}) \big) &= \mathcal{L}(q, \boldsymbol{\theta}^{old}) + KL\big(q(\mathbf{z} )\\| P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})\big)
\label{eq:e-step-kl}
\end{align}
\end{spreadlines}
$$

As a brief aside, we can see that, by setting $q(\mathbf{z}) = P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})$ in the E step, the KL divergence goes to zero and the incomplete-data log-likelihood becomes equal to the evidence lower bound (Equation \ref{eq:equality} is repeated below, with $\boldsymbol{\theta} = \boldsymbol{\theta}^{old}$). 

$$
\begin{align}
\log L(\boldsymbol{\theta}^{old}|\mathbf{x}) &=   \mathcal{L}\Big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}),\boldsymbol{\theta}^{old}\Big)
\tag{\ref{eq:equality} revisited}
\end{align}
$$

After computing new parameter estimates in the M step, the value of the incomplete-data log-likelihood increases at these new values $\boldsymbol{\theta}$ such that it it the sum of the old evidence lower bound maximized with respect to $\boldsymbol{\theta}^{new}$ and the KL divergence between the old responsibilities and some new distribution of the responsibilities, $\mathbf{z}^{new}$ (Equation \ref{eq:new-incomplete-q}). From Equation \ref{eq:ind-posterior}, we know that we can use Bayes' theorem to compute the new distribution of the latent variables, and so I set $q(\mathbf{z}^{new} = P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{new})$) in Equation \ref{eq:new-incomplete-bayes}. 


$$
\begin{spreadlines}{0.5em}
\begin{align}
\log L(\boldsymbol{\theta}^{new}|\mathbf{x}) &=   \mathcal{L}\Big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}),\boldsymbol{\theta}^{new}\Big) + KL\big(q(\mathbf{z}^{new})\|P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old})) 
\label{eq:new-incomplete-q} \\\\
&= \mathcal{L}\Big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}),\boldsymbol{\theta}^{new}\Big) + KL\big(P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{new})\|P(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}^{old}))
\label{eq:new-incomplete-bayes}
\end{align}
\end{spreadlines}{0.5em}
$$
### Computation of Parameter Estimates 

# Visualizing the Expectation-Maximization (EM) Algorithm

# Conclusion

# Appendix A: Proof of Jensen's Inequality {#jensen}

# Appendix B: Proof that KL Divergence is Always Non-Negative {#kl-divergence}

# Appendix C: Proof of Gibbs' Inequality{#gibbs}









