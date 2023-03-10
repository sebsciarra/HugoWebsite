---
title: "Probability, Likelihood, and Maximum Likelihood Estimation" 
draft: false
summary: 'Explanation of post' 
date: "2023-03-08"
article_type: simulation
output:
  bookdown::html_document2:
     keep_md: true
always_allow_html: true
header-includes: 
  - \usepackage{amsmath}
bibFile: content/technical_content/MLE_EM_algorithms/biblio.json    
tags: []
---   








# Probability Mass Functions: The Probability of Observing Each Possible Outcome Given Specific Parameter Values

Consider an example where a researcher obtains a coin and believes it to be unbiased, $P(\theta) = P(head) = 0.50$. To test this hypothesis, the researcher intends to flip the coin 10 times and record the result as a `1` for heads and `0` for tails, thus obtaining a vector of 10 observed scores, $\mathbf{y} \in \\{0, 1 \\}^{10}$, where $n = 10$. Before collecting the data to test their hypothesis, the researcher would like to get an idea of the probability of observing any given number of heads given that the coin is unbiased and there are 10 coin flips, $P(\mathbf{y}|\theta, n)$. Thus, the outcome of interest is the number of heads, $h$, where $\\{h|0 \le h \le10\\}$. Because the coin flips have a dichotomous outcome and the result of any given flip is independent of all the other flips, the probability of obtaining any given number of heads will be distributed according to a binomial distribution, $h \sim B(n, h)$. To compute the probability of obtaining any given number of heads, the *binomial function* shown below in Equation \ref{eq:prob-mass-function} can be used:
$$
\begin{align}
P(h|\theta, n) = {n \choose h}(\theta)^{h}(1-\theta)^{(n-h)},
\label{eq:prob-mass-function}
\end{align}
$$
where ${n \choose h}$ gives the total number of ways in which $h$ heads (or successes) can be obtained in a series of $n$ attempts (i.e., coin flips) and $(\theta)^{h}(1-\theta)^{(n-h)}$ gives the probability of obtaining a given number of $h$ heads and $n-h$ tails in a given set of $n$ flips. Thus, the binomial function (Equation \ref{eq:prob-mass-function}) has an underlying intuition: To compute the probability of obtaining a given number of $h$ heads given $n$ flips and a certain $\theta$ probability of success, the probability of obtaining $h$ heads in a given set of $n$ coin flips, $(\theta)^{h}(1-\theta)^{(n-h)}$, is multiplied by the total number of ways in which $h$ heads can be obtained in $n$ coin flips ${n \choose h}$.
<div style="display:none">\(\nextSection\)</div>
As an example, the probability of obtaining four heads ($h=4$) in 10 coin flips ($n = 10$) is calculated below. 

$$
\begin{alignat}{2}
P(h = 4|\theta = 0.50, n = 10) &= {10 \choose 4}(0.50)^{4}(1-0.50)^{(10-4)}   \nonumber \\\\
&= \frac{10!}{4! (10 - 4)!}(0.50)^{4}(1-0.50)^{(10-4)} \nonumber \\\\
&= 210(0.5)^{10}\nonumber \\\\
&= 0.205 \nonumber
\end{alignat}
$$
Thus, there are 210 possible ways of obtaining four heads in a series of 10 coin flips, with each way having a probability of $(0.5)^{10}$ of occurring. Altogether, four heads have a probability of .205 of occurring given a probability of heads of .50 and 10 coin flips. 

In order to calculate the probability of obtaining each possible number of heads in a series of 10 coin flips, the binomial function (Equation \ref{eq:prob-mass-function}) can be computed for each number. The resulting probabilities of obtaining each number of heads can then be plotted to produce a *probability mass function*: A distribution that gives the probability of obtaining each possible value of a discrete random variable[^1] (see Figure \ref{fig:prob-mass-binom}). Importantly, probability mass functions have two conditions: 1) the probability of obtaining each value is non-negative and 2) the sum of all probabilities is zero. The R code block below (lines <a href="#1">1--68</a>) produces a probability mass function for the binomial situation.


[^1]: Discrete variables have a countable number of discrete values. In the current example with ten coin flips ($n = 10$), the number of heads is a discrete variable because the number of heads, $h$, has a countable number of outcomes, $h \in \\{0, 1, 2, ..., n\\}$. 

```r 
#create function that computes probability mass function with following arguments:
  ##num_trials = number of trials (10  [coin flips] in the current example)
  ##prob_success = probability of success (or heads; 0.50 in the current example)
  ##num_successes = number of successes (or heads; [1-10] in the current example)

compute_binom_mass_density <- function(num_trials, prob_success, num_successes){
  
  #computation of binomial term (i.e., number of ways of obtaining a given number of successes)
  num_success_patterns <- (factorial(num_trials)/(factorial(num_successes)*factorial(num_trials-num_successes)))
  
  #computation of the number of possible ways of obtaining a given number of successes (i.e., heads)
  prob_single_pattern <- (prob_success)^num_successes*(1-prob_success)^(num_trials-num_successes)
  
  
  probability <- num_success_patterns*prob_single_pattern
  
  pmf_df <- data.frame('probability' = probability, 
                   'num_successes' = num_successes, 
                   'prob_success' = prob_success, 
                   'num_trials' = num_trials)
  
  return(pmf_df)
}



num_trials <- 10
prob_success <- 0.5
num_successes <- 0:10  #manipulated variable 

prob_distribution <- compute_binom_mass_density(num_trials, prob_success, num_successes)

library (tidyverse) 
library(grDevices) #needed for italic()

#create data set for shaded rectangle that indicates the most likely value 
##index of highest probability 
highest_number_ind <- which.max(prob_distribution$probability) 
##most likely number of successes
most_likely_number <- prob_distribution$num_successes[highest_number_ind] 
##probability value of most likely number of successes
highest_prob <- max(prob_distribution$probability) 

rectangle_data <-data.frame(
  'xmin' = most_likely_number - 0.10, 
  'xmax' = most_likely_number + 0.10,
  'ymin' = 0,
  'ymax' = highest_prob)


#create pmf plot 
pmf_plot <- ggplot(data = prob_distribution, aes(x = num_successes, y = probability)) + 
  geom_bar(stat = 'identity', 
           fill =  ifelse(test = prob_distribution$num_successes == most_likely_number, 
                                no =  "#002241", 
                                yes = "#00182d")) +  #calculate sum of probability for each num_successes
 ## geom_rect(inherit.aes = F, 
 ##           data = rectangle_data, mapping = aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax), 
 ##           fill = 'grey50', color = NA, alpha = 0.2) +
  scale_y_continuous(name = bquote(italic("P(h")*"|"*italic(theta == .(prob_success)*","~n == .(num_trials)*")"))) + 
  scale_x_continuous(name = bquote("Number of Heads (i.e., "*italic("h")~")"), 
                     breaks = seq(from = 0, to = 10, by = 1)) +
  theme_classic(base_family = "Helvetica", base_size = 18) +
  theme(axis.title.y = element_text(face = 'italic'), 
        
        #embolden the most likely number of heads 
        axis.text.x = 
          element_text(face = 
                         ifelse(test = prob_distribution$num_successes == most_likely_number, 
                                no =  "plain", 
                                yes = "bold")), 
        text = element_text(color = "#002241"),
        line = element_line(color = "#002241"), 
        axis.text = element_text(color = "#002241"))

ggsave(filename = 'images/pmf_plot.png', plot = pmf_plot, height = 6, width = 8)
```

<div class="figure">
  <div class="figDivLabel">
    <caption>
      <span class = 'figLabel'>Figure \ref{fig:prob-mass-binom}<span> 
    </caption>
  </div>
   <div class="figTitle">
    Probability Mass Function With an Unbiased Coin (<span class = "theta">&theta;</span> = 0.50) and Ten Coin Flips (n = 10)</span> 
  </div>
    <img src="images/pmf_plot.png" width="70%" height="70%"> 
  
  <div class="figNote">
      <span><em>Note. </em> Number emboldened on the x-axis indicates the number of heads that is most likely to occur with an unbiased coin and 10 coin flips, with the corresponding bar in darker blue  indicating the corresponding probability.</span> 
  </div>
</div>

Figure \ref{fig:prob-mass-binom} shows the probability mass function that results with an unbiased coin ($\theta = 0.50$) and ten coin flips ($n = 10$). In looking across the probability values of obtaining each number of heads (x-axis), 5 heads is the most likely value, as indicated by the emboldened number on the x-axis and the bar above it with a darker blue color. As an aside, the R code below verifies the two conditions of probability mass functions for the current example (for a mathematical proof, see [Appendix A](#proof-pmf)). 

```r 
#Condition 1: All probability values have nonnegative values. 
sum(prob_distribution$probability >= 0) #11 nonnegative values 

#Condition 2: Sum of all probability values is 1. 
sum(prob_distribution$probability)  #1
```

<pre><code class='r-code'>[1] 11
[1] 1
</code></pre>

With a probability mass function that shows the probability of obtaining each possible number of heads, the researcher now has an idea of what outcomes to expect after flipping the coin 10 times. Unfortunately, the probability mass function in Figure \ref{fig:prob-mass-binom} gives no insight into the coin's probability of heads after data have been collected; in computing the probability mass function, the probability of heads ($\theta$) is fixed. Thus, the researcher must use a different type of distribution to estimate the coin's probability of heads. 


# Likelihood Distributions: The Probability of Observing Each Possible Set of Parameter Values Given a Specific Outcome

Continuing with the coin flipping example, the researcher flips the coin 10 times and obtains seven heads. With this data, the researcher wants to determine the probability value of heads that most likely produced the data. In other words, the researcher wants to find the value of $\theta$ that maximizes the possibility of observing the data, $\max_{\theta \in \Theta} P(h = 7, n = 10|\theta)$. Before continuing, it is vital to explain why the researcher is no longer dealing with probabilities and is instead dealing with likelihoods.  


## Likelihoods are not Probabilities







Although probability density functions compute the probability that a set of data values have been observed given a fixed set of parameter valuess, we are seldom interested in this probability. Practitioners and researchers
are more interested in the probability that a certain set of parameter values characterize the larger population (i.e, $p(\theta|y)$). When trying to determine the most likely set of
parameter values, we use likelihood functions. Thus, the likelihood of a set of parameters given the observed data is represented as $L(\theta|y)$. Importantly, likelihoods do not 
represent the probability that a given set of parameter values defines the population (i.e., $P(\theta)$); we sample data and want to infer parameter values at the population level. With
probabilities, we assume knowledge of the population parameter values and calculate the probability of observing any given set of data. Using perhaps more relatable terms, the 
hypothesis is assumed to be true when calculating conditional probabilities and the data are varied. For likelihoods, the data is fixed and the hypothesis is varied. Note that Bayes
theorem can be used to convert likelihoods ($L(\theta|y)$) to probabilities ($P(\theta)$). 

To compute likelihoods, the function used to compute the above probabilities is used. Although confusing, the parameter values (the probability of success in this example [$\theta_1$])
are now being manipulated and not the data (see Figure \@ref(fig:likelihood-dist). Importantly, the sum of all the likelihoods does not sum to 1 ($\int^n_0 f(y|n, \theta) \neq 1$), 
which explains why likelihoods are sometimes called *unnormalized probabilities*. Although it seems unintuitive that the sum of the likelihoods is not 1, remember that likelihoods do not
describe the probability of a set of parameter being true (i.e., $p(\theta)$); if they did, then the integral would sum to 1. Given that we have two parameters, we can also produce 
another likelihood function by changing the values of the number of trials $\theta_1$ (see Figure \@ref(likelihood-dist2)). A joint likelihood function can be produced by varying 
both parameters simultaneously. 


## Resources 

# References


{{< bibliography cited >}}

# Appendix A: Proof That the Binomial Function is a Probability Mass Function  {#proof-pmf}

To prove that the binomial function is a probability mass function, two outcomes must be shown: 1) all probability values are non-negative and 2) the sum of all probabilities is 1. 

For the first condition, the impossibility of negative values occurring in the binomial function becomes obvious when individually considering the binomial coefficient, $n \choose k$, and the binomial factors, $p^k (1-p)^{n-k}$. With respect to the binomial coefficient, $n \choose k$, it is always nonnegative because it is the product of two non-negative numbers; the number of trials, $n$, and the number of successes can never be negative. With respect to the binomial factors, the resulting value is always nonnegative because all the term are nonnegative; in addition to the number of trials and successes ($n, k$, respectively),  the probability of success and failure are also always nonnegative ($p, k \in \[0,1\]$). Therefore, probabilities can be conceptualized as the product of a nonnegative binomial coefficient and a nonnegative binomial factor, and so is alwasys nonnegative. 

For the second condition, the equality stated below in Equation \ref{eq:binomial-sum-one} must be proven: 

\begin{align}
1 = \sum^n_{k=0} {n \choose k} \theta^k(1-\theta)^{n-k}.  
\label{eq:binomial-sum-one}
\end{align}

Importantly, it can be proven that all probabilities sum to one by using the binomial theorem, which states below in Equation \ref{eq:binomial} that 

\begin{align}
(a + b)^n =  \sum^n_{k=0} {n \choose k} a^k(b)^{n-k}. 
\label{eq:binomial}
\end{align}

Given the striking resemblance between the binomial function in Equation \ref{eq:binomial-sum-one} and the binomial theorem in Equation \ref{eq:binomial-sum-one}, it is possible to restate the binomial theorem with respect to the variables in the binomial function. Specifically, we can let $a = p$ and $b = 1-p$, which returns the proof as shown below: 

\begin{spreadlines}{0.5em}
\begin{align*}
(p + 1 -p)^n &= \sum^n_{k=0} {n \choose k} p^k(1-p)^{n-k} \\\\ \nonumber
1 &= \sum^n_{k=0} {n \choose k} p^k(1-p)^{n-k}   \nonumber 
\end{align*}
\end{spreadlines}


For a proof of the binomial theorem, see [Appendix E](#proof-binomial). 


# Appendix B: Proof That Likelihoods are not Probabilities  {#proof-likelihood}

As a reminder, although the same formula is used to compute likelihoods and probabilities, the variables allowed to vary and those set to be fixed differ when computing likelihoods and probabilities. With probabilities, the parameters are fixed ($\theta$, $n$) and the data are varied ($h$; notice how, in Appendix A, probabilities were summed across all possible values of $h$). With likelihoods, however, the data are fixed ($h$) and the parameters are varied ($\theta$, $n$). For the current proof, it is sufficient to only allow $\theta$ to vary. To prove that likelihoods are not probabilities, we have to prove that likelihoods do not satisfy one of the two conditions required by probabilities (i.e., likelihoods can have negative values or likelihoods do not sum to one). Given that likelihoods are calculated with the same function as probabilities and probabilities can never be negative (see [Appendix A](#proof-pmf)), likelihoods likewise can never ne negative. Therefore, to prove that likelihoods are not probabilities, we must prove that likelihoods do not always sum to 1. Thus, the following proposition must be proven: 
 
$$
 \int_0^1 L(\theta|h, n) \phantom{c} d\theta= \sum_{\theta = 0}^1{n \choose h} \theta^h(1 - \theta)^{n-h} \neq 1. 
$$
In summing each likelihood for $\theta \in \[0, 1\]$, an equivalent calculation is to take the integral of the binomial function with respect to theta such that
$$
\begin{spreadlines}{0.5em}
\begin{align}
 \int_0^1 L(\theta|h, n) \phantom{c} d\theta &= \int_0^1 L(\theta|h ,n) \phantom{c} d\theta  
\label{eq:int-sum-likelihood}\\\\
&= {n \choose h} \int_0^1 \theta^h(1-\theta)^{n-h}.
\label{eq:int-sum-likelihood-binomial}
\end{align}
\end{spreadlines}
$$
At this point, it is important to realize that $ \int_0^1 \theta^h(1-\theta)^{n-h}$ can be restated in terms of the beta function, $\mathrm{B}(x, y)$, which is shown below. 
$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathrm{B}(x, y) &= \int_0^1 t^{x-1}(1-t){^{y-1}} \phantom{c} dt 
\label{eq:beta-function} \\\\ 
\text{Let }&t = \theta \nonumber \\\\
\mathrm{B}(x, y) &= \int_0^1 \theta^{x-1}(1-\theta){^{y-1}} \phantom{c} d\theta \nonumber \\\\
\text{Let }&x = h +1 \text{ and } y = n -h +1 \nonumber \\\\
\mathrm{B}(h+1, n-h+1) &= \int_0^1 \theta^{h+1-1}(1-\theta){^{n-h+1-1}} \phantom{c} d\theta \nonumber \\\\
&= \int_0^1 \theta^{h}(1-\theta){^{n-h}} \phantom{c} d\theta
\end{align}
\end{spreadlines}
$$
Therefore, the function in Equation \ref{eq:int-sum-likelihood-binomial} can be restated below in Equation \ref{eq:beta-restate} as 
$$
\begin{align}
 \int_0^1 L(\theta|h, n) \phantom{c} d\theta = {n \choose h} \mathrm{B}(h+1, n-h+1).
\label{eq:beta-restate}
\end{align}
$$
At this point, another proof becomes important because it allows us to express the beta function in terms of another function that will, ultimately, allow us to simplify Equation \ref{eq:beta-restate} and prove that likelihoods do not sum to one and are, therefore, not probabilities.  Specifically, the beta function, $\mathrm{B}(x, y)$ can be stated in terms of the gamma function $\Gamma$ such that 

$$
\begin{align}
\mathrm{B}(x, y) = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}.
\label{eq:beta-gamma-relation}
\end{align}
$$
For a proof of the beta-gamma relation, see [Appendix C](#proof-beta-gamma). Thus, Equation \ref{eq:beta-restate} can be stated in terms of the gamma function such that 

$$
\begin{align}
\int_0^1 L(\theta|h, n) \phantom{c} d\theta  = {n \choose h} \frac{\Gamma(h+1)\Gamma(n-h+1)}{\Gamma(n+2)}.
\label{eq:binomial-gamma}
\end{align}
$$
One nice feature of the gamma function is that it can be stated as a factorial (for a proof, see [Appendix D](#proof-gamma-factorial)) such that 

$$
\begin{align}
\Gamma(x) = (x - 1)!.
\end{align}
$$
Given that the gamma function can be stated as a factorial, Equation \ref{eq:binomial-gamma} can be now be written with factorial terms and simplified to prove that likelihoods do not sum to one. 

$$
\begin{spreadlines}{0.5em}
\begin{align}
 \int_0^1 L(\theta|h, n) \phantom{c} d\theta &= \frac{n!}{h!(n-h)!}\frac{h!(n-h)!}{(n + 1)!} \nonumber \\\\ 
&= \frac{n!}{(n + 1)!} \nonumber \\\\ 
&= \frac{1}{n+1} 
\label{eq:likelihood-proof}  
\end{align} 
\end{spreadlines}
$$

Therefore, binomial likelihoods sum to a multiple of $\frac{1}{1+n}$, where the multiple is the number of integration steps. The R code block below provided an example where the integral can be shown to be a multiple of the value in Equation \ref{eq:likelihood-proof}. In the example, the integral of the likelihood is taken over 100 equally spaced steps. Thus, the sum of likelihoods should be $100\frac{1}{1+n} = 9.09$, and this turns out to be true in the code below. 

```r 
num_trials <- 10 #n
num_successes <- 7 #h
prob_success <- seq(from = 0, to = 1, by = 0.01) #theta; contains 100 values (i.e., there are 100 dtheta values)

likelihood_distribution <- compute_binom_mass_density(num_trials = num_trials, num_successes =  num_successes, prob_success = prob_success)
sum(likelihood_distribution$probability) #compute integral
```
<pre><code class='r-code'>[1] 9.09091
</code></pre>
# Appendix C: Proof of Relation Between Beta and Gamma Functions  {#proof-beta-gamma}

Equation \ref{eq:beta-gamma-proof} below will be proven 

$$
\begin{align}
\mathrm{B}(x, y) = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}.
\label{eq:beta-gamma-proof}
\end{align}
$$
To begin, let's write out the expansions of the gamma function, $\Gamma(x)$, and the numerator of Equation \ref{eq:beta-gamma-proof}, \Gamma(x)\Gamma(y), where 
$$
\begin{spreadlines}{0.5em}
\begin{align}
\Gamma(x) &= \int^\infty_0 t^{x-1}e^{-t} \phantom{c} dt 
\label{eq:gamma-function} \\\\ 
\Gamma(x)\Gamma(y) &= \int^\infty_0 t^{x-1}e^{-t} \phantom{c} dt \int^\infty_0 s^{y-1}e^{-s} \phantom{c} ds.
\label{eq:gamma-numerator}
\end{align}
\end{spreadlines}
$$
Equation \ref{eq:gamma-function} shows the gamma function which will be useful as a reference and Equation \ref{eq:gamma-numerator} shows the expansion of the numerator in Equation \ref{eq:beta-gamma-proof}. To prove Equation \ref{eq:beta-gamma-proof}, we will begin by changing the variables of $s$ and $t$ in Equation \ref{eq:gamma-numerator} by reexpressing them in terms of $u$ and $v$. Importantly,  when changing variables in a double integral, the formula below in Equation \ref{eq:double-integral} must be followed: 

$$
\begin{align}
 \underset{G}{\iint} f(x, y) \text{ }dx \text{ }dy =  \underset{}{\iint}f(g(u, v), h(u, v))\det(\mathbf{J}(u, v)) \text{ }du \text{ }dv, 
\label{eq:double-integral}
\end{align}
$$
where $|\mathbf{J}(u, v)|$ is the Jacobian of $u$ and $v$ (for a great explanation, see [Jacobian](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/jacobian/v/the-jacobian-matrix) and [change of variables](https://www.youtube.com/watch?v=wUF-lyyWpUc)). To apply Equation \ref{eq:double-integral}, we will first determine the expressions of $s$ and $t$ in terms of $u$ and $v$ to obtain $g(u,v)$ and $h(u,v)$, which are, respectively, provided below in Equation \ref{eq:s-rexp} and Equation \ref{eq:t-rexp}. 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\text{Let } u &= s + t, \text{  } v = \frac{t}{s+t} \nonumber \\\\
\text{then }  \text{ }s &= u - t = u - uv = g(u, v)
\label{eq:s-rexp} \\\\
t &= u - s = u - (u - uv) = uv = h(u, v).
\label{eq:t-rexp}
\end{align}
\end{spreadlines}
$$
With the expression for $g(u,v)$ and $h(u,v)$, the determinant of the Jacobian of $u$ and $v$ can now be computed, as shown below and provided in Equation \ref{eq:det-Jac}. 
$$
\begin{spreadlines}{0.5em}
\begin{align}
\det\mathbf{J}(u, v) &= 
\det\begin{bmatrix}
\frac{\partial g}{\partial u} & \frac{\partial g}{\partial v} \\
\frac{\partial h}{\partial u} & \frac{\partial h}{\partial v}
\end{bmatrix} \nonumber \\\\
&= \det\begin{bmatrix}
1-v & -u \\
v & u 
\end{bmatrix} \nonumber \\\\ 
&= (1 - v)u - (-uv) = u - uv + uv = u
\label{eq:det-Jac}
\end{align}
\end{spreadlines}
$$
With the $\det\mathbf{J}(u, v)$ computed, we can no express the new function with the changed variables, as shown below in Equation \ref{eq:gamma-reexp}. 

$$
\begin{align}
\underset{G}{\iint} f(g(u, v), h(u, v))\det(\mathbf{J}(u, v)) \text{ }du \text{ }dv &= \underset{R}{\iint} \Gamma(g(u, v))\Gamma(h(u,v))\det(\mathbf{J}(u, v)) \text{ }du \text{ }dv \nonumber \\\\
&= \underset{R}{\iint}  uv^{x-1}e^{-uv}  (u - uv)^{y-1}e^{-(u - uv)}u d\text{ } u \text{ }dv \nonumber \\\\
&= \underset{R}{\iint}  u^{x-1}v^{x-1} e^{-uv} (u(1 - v)^{y-1})e^{-(u - uv)}u \text{ } du\text{ }dv \nonumber \\\\
&= \underset{R}{\iint}  u^{x-1}u^{y-1}u e^{-uv}e^{-u + uv}  v^{x-1} (1 - v)^{y-1} \text{ } du\text{ }dv \nonumber \\\\
&= \underset{R}{\iint}  u^{x+y-1} e^{-u} v^{x-1} (1 - v)^{y-1} \text{ } du\text{ }dv 
\label{eq:gamma-reexp}
\end{align}
$$
At this point, we need to determine the integration limits of $u$ and $v$ by evaluating them at the limits of $s$ and $t$, which is shown below. 

$$
\begin{spreadlines}{0.5em}
\begin{align}
&\text{Recall } u = s + t, v = \frac{t}{s+t}, \text{ and } s,y \in\[0, \infty\] \nonumber \\\\
&\text{If } s = 0 \Rightarrow u = t, v = 1 \nonumber \\\\
&\phantom{If } s = \infty \Rightarrow u = \infty, v = 0 \nonumber \\\\
&\phantom{If } t = 0 \Rightarrow u = s, v = 0 \nonumber \\\\
&\phantom{If } t = \infty \Rightarrow u = \infty, v = 1 \nonumber
\end{align}
\end{spreadlines}
$$

Therefore, the original integration limits of 0 to $\infty$ of $s$ and $t$ produce integration limits 0 to $\infty$ for $u$ and 0 to 1 for $v$.  Recalling the gamma function (Equation \ref{eq:gamma-function} and the beta function (Equation \ref{eq:beta-function}, the beta function can now be expressed in terms of the gamma function, proving Equation \ref{eq:beta-gamma-proof}. 

\begin{spreadlines}{0.5em}
\begin{align*}
\Gamma(x)\Gamma(y) &= \int_0^1 \int_0^\infty u^{x+y-1} e^{-u} v^{x-1} (1 - v)^{y-1} \,du\,dv \\\\
&=  \int_0^\infty  u^{x+y-1} e^{-u}\text{ } du \int_0^1v^{x-1} (1 - v)^{y-1} \,dv \\\\
&=  \Gamma(x + y)\mathrm{B}(x,y) \\\\
\mathrm{B}(x,y) &= \frac{\Gamma(x)\Gamma(y)}{ \Gamma(x + y)}
\end{align*}
\end{spreadlines}



# Appendix D: Proof of Relation Between Gamma and Factorial Functions  {#proof-gamma-factorial}

# Appendix E: Proof of Binomial Theorem  {#proof-binomial}



