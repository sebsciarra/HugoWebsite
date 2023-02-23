---
title: "Probability, Likelihood, and Maximum Likelihood Estimation" 
draft: false
summary: 'Explanation of post' 
article_type: technical
output:
  bookdown::html_document2:
     keep_md: true
always_allow_html: true
bibFile: content/technical_content/em_algorithm/biblio.json    
tags: []
---   






# Probability Mass Functions: The Probability of Observing Each Possible Outcome Given Specific Parameter Values

Consider an example where a researcher obtains a coin and believes it to be unbiased, $P(\theta) = P(head) = 0.50$. To test this hypothesis, the researcher intends to flip the coin 10 times and record the result as a `1` for heads and `0` for tails, thus obtaining a vector of 10 observed scores, $\mathbf{y} \in \\{0, 1 \\}^{10}$, where $n = 10$. Before collecting the data to test their hypothesis, the researcher would like to get an idea of the probability of observing any given number of heads given that the coin is unbiased and there are 10 coin flips, $P(\mathbf{y}|\theta, n)$. Thus, the outcome of interest is the number of heads, $h$, where $\\{h|0 \le h \le10\\}$. Because the coin flips have a dichotomous outcome and the result of any given flip is independent of all the other flips, the probability of obtaining any given number of heads will be distributed according to a binomial distribution, $h \sim B(n, h)$. To compute the probability of obtaining any given number of heads, the *binomial function* shown below in Equation \ref{eq:prob-mass-function} can be used:

\begin{align}
P(h|\theta, n) = {n \choose h}(\theta)^{h}(1-\theta)^{(n-h)},
\label{eq:prob-mass-function}
\end{align}

where ${n \choose h}$ gives the total number of ways in which $h$ heads (or successes) can be obtained in a series of $n$ attempts (i.e., coin flips) and $(\theta)^{h}(1-\theta)^{(n-h)}$ gives the probability of obtaining a given number of $h$ heads and $n-h$ tails in a given set of $n$ flips. Thus, the binomial function (Equation \ref{eq:prob-mass-function}) has an underlying intuition: To compute the probability of obtaining a given number of $h$ heads given $n$ flips and a certain $\theta$ probability of success, the probability of obtaining $h$ heads in a given set of $n$ coin flips, $(\theta)^{h}(1-\theta)^{(n-h)}$, is multiplied by the total number of ways in which $h$ heads can be obtained in $n$ coin flips ${n \choose h}$.

As an example, the probability of obtaining four heads ($h=4$) in 10 coin flips ($n = 10$) is calculated below. 

$$
\begin{alignat}{2}
P(h = 4|\theta = 0.50, n = 10) &= {10 \choose 4}(0.50)^{4}(1-0.50)^{(10-4)}   \nonumber \\\\
&= \frac{10!}{4! (10 - 4)!}(0.50)^{4}(1-0.50)^{(10-4)} \nonumber \\\\
&= 210(0.5)^{10}\nonumber \\\\
&= 0.205 \nonumber
\end{alignat}
$$
Thus, there are 210 possible ways of obtaining four heads in a series of 10 coin flips, with each way having a probability of $(0.5)^{10}$ of occurring.

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
    <img src="images/pmf_plot.png" width="60%" height="60%"> 
  
  <div class="figNote">
      <span><em>Note. </em> Number emboldened on the x-axis indicates the number of heads that is most likely to occur with an unbiased coin and 10 coin flips, with the corresponding bar in darker blue  indicating the corresponding probability.</span> 
  </div>
</div>

Figure \ref{fig:prob-mass-binom} shows the probability mass function that results with an unbiased coin ($\theta = 0.50$) and ten coin flips ($n = 10$). In looking across the probability values of obtaining each number of heads (x-axis), 5 heads is the most likely value, as indicated by the emboldened number on the x-axis and the bar above it with a darker blue color. As an aside, the R code below verifies the two conditions of probability mass functions for the current example (for a mathematical proof, see [Appendix A](#proofs)). 

```r 
#Condition 1: All probability values have nonnegative values. 
sum(prob_distribution$probability >= 0) #11 nonnegative values 

#Condition 2: Sum of all probability values is 1. 
sum(prob_distribution$probability)  #1
```

<pre><code class='r-code'>[1] 11
[1] 1
</code></pre>

Although the binomial distribution the number of heads that is most likely to occur with an unbiased coin, it gives little insight on the coin's probability of heads ($\theta$) after data have been obtained. If the researcher flips the coin 10 times and obtains 7 heads, the researcher may then want to know the likelihood of each probability value of heads ($\theta$ ) given the 

<pre><code class=''>[1] 0.2460938
</code></pre><pre><code class=''>0.004921845 with absolute error < 5.5e-17
</code></pre><pre><code class=''>[1] 0.004921845
</code></pre>

# Likelihood Distributions: The Probability of Observing Each Possible Set of Parameter Values Given a Specific Outcome
 $\forall  \theta \in \[0, 1\], P(\theta|h = 7, n = 10)$. 
# Maximum Likelihood Estimation: Estimating the Set of Parameter Values That Most Likely Produce a Specific Outcome

# References


{{< bibliography cited >}}

# Appendix A: Proof That the Binomial Function is a Probability Mass Function  {#proofs}

To proove that the binomial function is a probability mass function, two outcomes must be shown: 1) all probability values are non-negative and 2) the sum of all probabilities is 1. 





The incomplete beta function is defined as:

$$B(x;a,b) = \int_0^x t^{a-1}(1-t)^{b-1} dt$$

The binomial function is defined as:

$$f(k;n,p) = {n\choose k}p^k(1-p)^{n-k}$$

where ${n\choose k}$ is the binomial coefficient, which represents the number of ways to choose k items from a set of n items.

To derive the incomplete beta function from the binomial function, we can make use of the following relationship:

$${n \choose k} = \frac{n!}{k!(n-k)!} = \frac{\Gamma(n+1)}{\Gamma(k+1)\Gamma(n-k+1)}$$

where $\Gamma(x)$ is the gamma function. Using this relationship, we can write the binomial function as:

$$f(k;n,p) = \frac{\Gamma(n+1)}{\Gamma(k+1)\Gamma(n-k+1)}p^k(1-p)^{n-k}$$

Now, let's make the substitution $t = x$ and $u = 1-x$. Then, we have:

$$\begin{aligned} B(x;a,b) &= \int_0^x t^{a-1}(1-t)^{b-1} dt \ &= \int_0^1 (xt)^{a-1}[(1-x)u]^{b-1} dx \quad \text{(using the substitution } t=x \text{ and } u=1-x \text{)}\ &= x^{a-1}(1-x)^{b-1} \int_0^1 t^{a-1}u^{b-1} dt \ &= x^{a-1}(1-x)^{b-1} B(a,b) \end{aligned}$$

where the last step follows from the definition of the complete beta function:

$$B(a,b) = \int_0^1 t^{a-1}(1-t)^{b-1} dt$$

Now, let's write the binomial function in terms of the incomplete beta function. Using the relationship between the gamma function and the binomial coefficient, we can write:

$$\begin{aligned} f(k;n,p) &= \frac{\Gamma(n+1)}{\Gamma(k+1)\Gamma(n-k+1)}p^k(1-p)^{n-k} \ &= \frac{1}{B(k+1,n-k+1)}p^k(1-p)^{n-k} \end{aligned}$$

Substituting this expression for $f(k;n,p)$ into the expression we derived earlier for $B(x;a,b)$, we get:

$$\begin{aligned} B(x;a,b) &= x^{a-1}(1-x)^{b-1} B(a,b) \ &= \frac{x^{a-1}(1-x)^{b-1}}{B(a,b)}\sum_{k=0}^n \frac{1}{B(k+1,n-k+1)}p^k(1-p)^{n-k} \end{aligned}$$

This expression relates the incomplete beta function to the binomial function. However, it should be noted that this is not a closed-form solution for the incomplete beta function, since it still involves a summation over the values of k. The incomplete beta function is usually evaluated numerically using software such as R or MATLAB.








Sure, let me try to simplify the proof.

The binomial function is defined as:

$$f(k;n,p) = \binom{n}{k} p^k (1-p)^{n-k}$$

where $\binom{n}{k}$ is the binomial coefficient, $p$ is the probability of success, and $n$ is the number of trials.

To show that all values of the binomial function sum to 1, we need to consider the sum of the function over all possible values of $k$ from 0 to $n$. That is, we need to show that:

$$\sum_{k=0}^{n} \binom{n}{k} p^k (1-p)^{n-k} = 1$$

The binomial function gives the probability of getting $k$ successes in $n$ trials, where the probability of success in each trial is $p$. The sum on the left-hand side of the equation is the probability of getting 0 successes, 1 success, 2 successes, and so on, up to $n$ successes in $n$ trials.

The proof relies on the fact that the sum of probabilities of all possible outcomes of a random experiment is always equal to 1. In this case, the random experiment is the binomial experiment with $n$ trials and probability of success $p$.

To prove the equation, we use the binomial theorem, which states that:

$$(x+y)^n = \sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k}$$

Setting $x=p$ and $y=1-p$, we can rewrite the binomial function as:

$$f(k;n,p) = \binom{n}{k} p^k (1-p)^{n-k} = \binom{n}{k} (p(1-p))^k (1-p)^n$$

Using the binomial theorem with $x=p$ and $y=1-p$, we get:

$$(p+(1-p))^n = \sum_{k=0}^{n} \binom{n}{k} p^k (1-p)^{n-k}$$

Substituting this expression for the sum on the left-hand side of the equation, we get:

$$\sum_{k=0}^{n} \binom{n}{k} p^k (1-p)^{n-k} = (p+(1-p))^n = 1^n = 1$$

Therefore, we have shown that the sum of the binomial function over all possible values of $k$ is always equal to 1, which means that the total probability of all possible outcomes in a binomial experiment is always equal to 1, as expected.

