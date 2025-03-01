---
title: "Probability, Likelihood, and Maximum Likelihood Estimation" 
draft: false
summary: 'Probability and likelihood are discussed in the context of a coin flipping scenario and it is shown that only probabilities sum to one. Although likelihoods cannot be interpreted as probabilities, they can be used to determine the set of parameter values that most likely produced a data set (maximum likelihood estimates). Maximum likelihood estimation provides one efficient method for determining maximum likelihood estimates and is applied in the binomial and Gaussian cases.' 
date: "2023-03-19"
article_type: technical
output:
  bookdown::html_document2:
     keep_md: true
always_allow_html: true
header-includes: 
bibFile: content/technical_content/MLE/biblio.json    
tags: []
---   








# Probability Mass Functions: The Probability of Observing Each Possible Outcome Given One Set of Parameter Values

Consider an example where a researcher obtains a coin and believes it to be unbiased, $P(\theta) = P(head) = 0.50$. To test this hypothesis, the researcher intends to flip the coin 10 times and record the result as a `1` for heads and `0` for tails. Thus, a vector of 10 observed scores is obtained, $\mathbf{y} \in \\{0, 1 \\}^{n}$, where $n = 10$. Before collecting the data to test their hypothesis, the researcher would like to get an idea of the probability of observing any given number of heads given that the coin is unbiased and there are 10 coin flips, $P(\mathbf{y}|\theta, n)$. Thus, the outcome of interest is the number of heads, $h$, where $\\{h|0 \le h \le10\\}$. Because the coin flips have a dichotomous outcome and the result of any given flip is independent of all the other flips, the probability of obtaining any given number of heads will be distributed according to a binomial distribution, $h \sim B(n, h)$. To compute the probability of obtaining any given number of heads, the *binomial function* shown below in Equation \ref{eq:prob-mass-function} can be used:

$$
\begin{align}
P(h|\theta, n) = {n \choose h}(\theta)^{h}(1-\theta)^{(n-h)},
\label{eq:prob-mass-function}
\end{align}
$$
where ${n \choose h}$ gives the total number of ways in which $h$ heads (or successes) can be obtained in a series of $n$ attempts (i.e., coin flips) and $(\theta)^{h}(1-\theta)^{(n-h)}$ gives the probability of obtaining a given number of $h$ heads and $n-h$ tails in a given set of $n$ flips. Thus, the binomial function (Equation \ref{eq:prob-mass-function}) has an underlying intuition: To compute the probability of obtaining a specific number of $h$ heads given $n$ flips and a certain $\theta$ probability of success, the probability of obtaining $h$ heads in a given set of $n$ coin flips, $(\theta)^{h}(1-\theta)^{(n-h)}$, is multiplied by the total number of ways in which $h$ heads can be obtained in $n$ coin flips, ${n \choose h}$.

As an example, the probability of obtaining four heads ($h=4$) in 10 coin flips ($n = 10$) is calculated below. 

$$
\begin{spreadlines}{0.5em}
\begin{align}
P(h = 4|\theta = 0.50, n = 10) &= {10 \choose 4}(0.50)^{4}(1-0.50)^{(10-4)}   \nonumber \\\\
&= \frac{10!}{4! (10 - 4)!}(0.50)^{4}(1-0.50)^{(10-4)} \nonumber \\\\
&= 210(0.5)^{10}\nonumber \\\\
&= 0.205 \nonumber
\end{align}
\end{spreadliness}
$$
Thus, there are 210 possible ways of obtaining four heads in a series of 10 coin flips, with each way having a probability of $(0.5)^{10}$ of occurring. Altogether, four heads have a probability of .205 of occurring given a probability of heads of .50 and 10 coin flips. 

In order to calculate the probability of obtaining each possible number of heads in a series of 10 coin flips, the binomial function (Equation \ref{eq:prob-mass-function}) can be computed for each number of heads, $h$. The resulting probabilities of obtaining each number of heads can then be plotted to produce a *probability mass function*: A distribution that gives the probability of obtaining each possible value of a discrete random variable[^1] (see Figure \ref{fig:prob-mass-binom}). Importantly, probability mass functions have two conditions: 1) the probability of obtaining each value is non-negative and 2) the sum of all probabilities is one. The R code block below (see lines <a href="#1">1--65</a>) produces a probability mass function for the current binomial example.


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

#create pmf plot 
pmf_plot <- ggplot(data = prob_distribution, aes(x = num_successes, y = probability)) + 
  geom_bar(stat = 'identity', 
           fill =  ifelse(test = prob_distribution$num_successes == most_likely_number, 
                                no =  "#002241", 
                                yes = "#00182d")) +  #calculate sum of probability for each num_successes

    scale_y_continuous(name = bquote(italic("P(h")*"|"*italic(theta == .(prob_success)*","~n == .(num_trials)*")"))) + 
  scale_x_continuous(name = bquote("Number of Heads ("*italic("h")~")"), 
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
        axis.line = element_line(color = "#002241"), 
        axis.ticks = element_line(color =  "#002241"), 
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
    Probability Mass Function With an Unbiased Coin (<span class = "theta">&theta;</span> = 0.50) and Ten Coin Flips (n = 10)
  </div>
    <img src="images/pmf_plot.png" width="70%" height="70%"> 
  
  <div class="figNote">
      <span><em>Note. </em> Number emboldened on the x-axis indicates the number of heads that is most likely to occur with an unbiased coin and 10 coin flips, with the corresponding bar in darker blue  indicating the corresponding probability.</span> 
  </div>
</div>

Figure \ref{fig:prob-mass-binom} shows the probability mass function that results with an unbiased coin ($\theta = 0.50$) and ten coin flips ($n = 10$). In looking across the probability values of obtaining each number of heads (x-axis), 5 heads is the most likely value, as indicated by the emboldened number on the x-axis and the bar above it with a darker blue color. As an aside, the R code below (lines <a href="#66">66--70</a>) verifies the two conditions of probability mass functions for the current example (for a mathematical proof, see [Appendix A](#proof-pmf)). 

```r 
#Condition 1: All probability values have nonnegative values. 
sum(prob_distribution$probability >= 0) #11 nonnegative values 

#Condition 2: Sum of all probability values is 1. 
sum(prob_distribution$probability)  #1
```

<pre><code class='r-code'>[1] 11
[1] 1
</code></pre>

With a probability mass function that shows the probability of obtaining each possible number of heads, the researcher now has an idea what outcomes to expect after flipping the coin ten times. Unfortunately, the probability mass function in Figure \ref{fig:prob-mass-binom} gives no insight into the coin's actual probability of heads, $\theta$, after data have been collected; in computing the probability mass function, the probability of heads is fixed. Thus, the researcher must use a different type of distribution to determine the coin's probability of heads. 


# Likelihood Distributions: The Probability of Observing Each Possible Set of Parameter Values Given a Specific Outcome

Continuing with the coin flipping example, the researcher flips the coin 10 times and obtains seven heads. With these data, the researcher wants to determine the probability value of heads, $\theta$, that most likely produced the data, $P(h, n|\theta)$[^2].  Before continuing, it is important to explain why the researcher is no longer dealing with probabilities and is instead dealing with likelihoods. 

[^2]: It should be noted that Bayes' formula can also be used to determine the value of $\theta$ that most likely produced the data. Instead of calculating, $P(h, n|\theta)$, however, Bayes' formula uses prior information about an hypothesis to calculate the probability of $\theta$ given the data, $P(\theta|h, n)$ (for a review, see {{< citePara "etz2018" >}}). 

## Likelihoods are not Probabilities{#like-prob}

Because we are interested in determining which value of $\theta \in \[0, 1\]$ most likely produced the data, the probability of observing the data must be computed for each of these values, $P(h = 7, n = 10|\theta)$. Thus, we now fix the data, $h = 7, n = 10$, and vary the parameter value of $\theta$. Although we also use the binomial function to compute $P(h = 7, n = 10|\theta)$ for each $\theta \in \[0, 1\]$, the resulting values are not probabilities because they do not sum to one. Indeed, the R code block below ( lines <a href="#73">73--80</a>) shows that the values sum to 9.09. Thus, when fixing the data and varying the parameter values, the resulting values do not sum to one (for a mathematical proof with the binomial function, see [Appendix B](#proof-likelihood) and are, therefore, not probabilities: they are likelihoods. To signify the shift from probabilities to likelihoods, a different notation is used. Instead of computing the probability of the data given a parameter value, $P(h = 7, n = 10|\theta)$, the likelihood of the parameter given the data is computed, $L(\theta|h, n)$. 
  
```r 
num_trials <- 10
num_successes <- 7
prob_success <- seq(from = 0, to = 1, by = 0.01) #manipulated variable 

#compute P(h, n|theta) for each theta in [0, 1].
likelihood_distribution <- compute_binom_mass_density(num_trials = num_trials, num_successes =  num_successes, prob_success = prob_success)

sum(likelihood_distribution$probability)
```

<pre><code class='r-code'>[1] 9.09091
</code></pre>

In computing likelihoods, it is important to note that, because they do not sum to one, they cannot be interpreted as probabilities. As an example, the likelihood of 0.108 obtained for $L(\theta = .50|h=7, n=10)$ does not mean that, given a probability of heads of .50, there is a 10.80% chance that seven heads will arise in 10 coin flips: The value of $L(\theta = .50|h=7, n=10) = 0.108$ provides a measure of how strongly the data are expected under the hypothesis that $\theta = .50$. To gain a better understanding of whether the likelihood value of 0.108 is a high value, the likelihood values of all the other $\theta$ can be computed. 

## Creating a Likelihood Distribution to Find the Maximum Likelihood Estimate

Figure \ref{fig:likelihood-dist} shows the likelihood distribution of for all values of $\theta \in \[0, 1\]$. By plotting the likelihoods, the parameter value that most likely produced the data or the *maximum likelihood estimate* can be identified. The maximum likelihood estimate of $\theta$ in this example is .70, which is emboldened on the x-axis and its likelihood indicated by the height of the vertical bar. The R code block below (lines <a href="#82">82--126</a>) plots computes the likelihood values for all $\theta \in \[0, 1\]$. 


```r 
num_trials <- 10
num_successes <- 7
prob_success <- seq(from = 0, to = 1, by = 0.01) #manipulated variable 

likelihood_distribution <- compute_binom_mass_density(num_trials = num_trials, num_successes =  num_successes, prob_success = prob_success)

#create data set for shaded rectangle that indicates the most likely value 
##index of highest probability 
highest_number_ind <- which.max(likelihood_distribution$probability) 
##most likely number of successes
maximum_likelihood_estimate <- likelihood_distribution$prob_success[highest_number_ind] 
##probability value of most likely number of successes
highest_prob <- max(likelihood_distribution$probability) 

rectangle_data <- data.frame(
  'xmin' = maximum_likelihood_estimate - .005, 
  'xmax' = maximum_likelihood_estimate + .005,
  'ymin' = -0.5,
  'ymax' = highest_prob)


likelihood_plot <- ggplot(data = likelihood_distribution, aes(x = prob_success, y = probability)) + 
  geom_line(color = "#002241") + 
  geom_rect(inherit.aes = F, 
             fill = "#002241",
            data = rectangle_data, mapping = aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax))+ 
  
    scale_x_continuous(name = bquote(paste("Probability of Heads (", theta, ")")), 
                       breaks = seq(0, 1, 0.10), 
                       labels = scales::number_format(accuracy = 0.01)) +
  
    scale_y_continuous(name = bquote(italic("L(")*italic(theta)*"| "*italic("h")== .(num_successes)*","~italic("n") == .(num_trials)*")"),  
                       labels = scales::number_format(accuracy = 0.01), 
                       breaks = seq(from = 0, to = .30, by = .10)) +
  coord_cartesian(ylim = c(0, .30)) + 

   theme_classic(base_family = "Helvetica", base_size = 18) +

  theme(axis.text.x = element_text(face = ifelse(seq(0, 1, 0.10) == maximum_likelihood_estimate, "bold", "plain")),
        text = element_text(color = "#002241"),
        axis.line = element_line(color = "#002241"), 
        axis.ticks = element_line(color =  "#002241"), 
        axis.text = element_text(color = "#002241"))

ggsave(filename = 'images/likelihood_plot.png', plot = likelihood_plot, height = 6, width = 8)
```

<div class="figure">
  <div class="figDivLabel">
    <caption>
      <span class = 'figLabel'>Figure \ref{fig:likelihood-dist}<span> 
    </caption>
  </div>
   <div class="figTitle">
    <span>Likelihood Distribution With Seven Heads (h = 7) and Ten Coin Flips (n = 10)</span>
  </div>
    <img src="images/likelihood_plot.png" width="70%" height="70%"> 
  
  <div class="figNote">
      <span><em>Note. </em> Number emboldened on the x-axis indicates the maximum likelihood estimate for <span class = "theta">&theta;</span> and the corresponding bar in dark blue indicates the likelihood value.</span> 
  </div>
</div>

Although maximum likelihood estimates can be identified by creating likelihood distributions, this method is not efficient. Under many circumstances, creating such distributions is computationally demanding when a large range of parameter values must be considered. Even more important, many situations arise where many parameters are estimated, and this can make plotting the likelihood distribution impossible. As an example, if a researcher wants to estimate six parameters and plot the likelihood distribution, then six dimensions would have to be represented on a 2D plot, which is a nearly impossible task.  Thus, a more efficient method is needed to find maximum likelihood estimates that does not rely on plotting. 


# Using Maximum Likelihood Estimation to Find the Most Likely Set of Parameter Values

Maximum likelihood estimation identifies maximum likelihood estimates by using calculus to find a peak on the likelihood distribution. In mathematical parlance, maximum likelihood estimation solves for the parameter value where the derivative (i.e., rate of change) is zero. Assuming the likelihood only has one peak (i.e., it is convex), then the parameter value at the zero-derivative point will have the highest likelihood and will, therefore, be the maximum likelihood estimate. In mathematical notation, then, the maximum likelihood estimate, $\theta_{MLE}$, is the value of $\theta$ that maximizes the likelihood function

$$ 
\begin{align}
\theta_{MLE} &= \underset{\theta}{\arg\max}  L(\theta|D).
\label{eq:MLE-general}
\end{align}
$$
In the two sections that follow, I will apply maximum likelihood estimation for the binomial and Gaussian cases. 

## Maximum Likelihood Estimation for the Binomial Case 

In the binomial case, there is only one parameter value of interest: the probability of heads, $\theta$. Thus, maximum likelihood estimation will find the value $\theta$ that maximizes the likelihood function,

$$
\begin{align}
\underset{\theta}{\arg\max}\text{ } L(\theta|h, n) &= \underset{\theta}{\arg\max}\text{ }{n \choose h}(\theta)^{h}(1-\theta)^{(n-h)}.
\label{eq:mle-binomial}
\end{align}
$$

Before computing the maximum likelihood estimate, however, it is important to apply a $\log$ transformation on Equation \ref{eq:mle:binomial} for two reasons. First, applying a $\log$ transformation to the likelihood function of Equation \ref{eq:mle-binomial} greatly simplifies the computation of the derivative because taking the derivative of the log-likelihood does not involve a lengthy application of the quotient, product, and chain rules. Second, log-likelihoods are necessary to avoid *underflow*: the rounding of small numbers to zero in computers. As an example, in a coin flipping example with a moderate number of flips such as $n = 100$ and $h=70$, many likelihood values become extremely small (e.g., 1.20E-73) and can easily be rounded down to zero within computers. Instead of directly representing extremely small values, $\log$ likelihoods can be used to retain numerical precision. For example, the value of 1.2E-73 becomes -72.9208188 on a log scale (base 10), $\log_{10}{1.2e73} = -72.92$. In applying a $\log$ transformation to the likelihood function, the log-likelihood function shown below in Equation \ref{eq:binom-log-likelihood} is obtained: 

$$
\begin{align}
\log\big(L(\theta|h,n)\big) &= \log {n \choose h}\ + h\log(\theta) + (n-h)\log(1-\theta)
\label{eq:binom-log-likelihood}
\end{align}
$$
To solve for $\theta_{MLE}$, the partial derivative of $\log[L(\theta|h,n)]$ with respect to $\theta$ is computed below and then set to zero (at a peak, the likelihood function has a zero-value rate of change with respect to $\theta$). 

$$  
\begin{spreadlines}{0.5em}
\begin{align}
\frac{\delta \log[L(\theta|h,n)]}{\delta \theta} &= \frac{\delta}{\delta \theta}  \Bigg(\log {n \choose h} + h\log(\theta) + (n-h)\log(1-\theta) \Bigg) \nonumber \\\\
&= 0 + h(\frac{1}{\theta}) + (n-h)(-1)(\frac{1}{1-\theta})\nonumber \\\\
0 &= \frac{h}{\theta}- \frac{n-h}{1-\theta} \nonumber \\\\
\frac{n-h}{1-\theta} &= \frac{h}{\theta} \nonumber \\\\
\theta n - \theta h &= h - \theta h \nonumber \\\\
\theta n &=h \nonumber \\\\
\theta &= \frac{h}{n}
\label{eq:theta-binom-ll}
\end{align}
\end{spreadlines}
$$
Therefore, the maximum likelihood estimate for the probability of heads, $\theta$, is found by dividing the number of observed headsby the number of flips, $\frac{h}{n}$ (see Equation \ref{eq:theta-binom-ll}). In the current example where seven heads were obtained in ten coin flips, the probability value of heads that that maximizes the probability of observing the data is .70, $\theta_{MLE} = \frac{7}{10} = .70$. 

### Maximum Likelihood Estimation for Several Binomial Cases 

To build on the current example, consider a more realistic example where a researcher decides to flip a coin over multiple sessions. Specifically, in each of 10 $k$ sessions, the researcher flips the coin 10 times. Across the 10 sessions, the following number of heads are obtained: $\mathbf{h} = \[1, 6, 4, 7, 3, 4 ,5, 10, 5, 3\]$. At this point, it may seem daunting to compute the partial derivative of the resulting likelihood function with respect to $\theta$
because the equation will contain $k=10$ terms. Thankfully, a simple equation can be derived that does not require a lengthy partial derivative computation. To derive a $\theta_{MLE}$ equation for multiple coin flipping sessions, I will compute the function for $\theta_{MLE}$ with only two coin flipping sessions that each have their corresponding number of flips, $\mathbf{n} = \[n_1, n_2\]$, and heads, $\mathbf{h} = \[h_1, h_2\]$.

$$  
\begin{spreadlines}{0.5em}
\begin{align} 
\frac{\delta \big( \log[L(\theta|h_1,n_1)] +  \log[L(\theta|h_2,n_2)]\big)}{\delta \theta} &= \frac{\delta}{\delta \theta}  \Bigg(\Big(\log {n_1 \choose h_1} + h_1\log(\theta) + (n_1-h_1)\log(1-\theta) \Big) + \nonumber \\\\
&\Big(\log {n_2 \choose h_2} + h_2\log(\theta) + (n_2-h_2)\log(1-\theta) \Big)\Bigg) \nonumber \\\\
&= 0 + h_1(\frac{1}{\theta}) + (n_1-h_1)(-1)(\frac{1}{1-\theta}) + h_2(\frac{1}{\theta}) + (n_2-h_2)(-1)(\frac{1}{1-\theta}) \nonumber \\\\
&= \frac{h_1}{\theta}+ \frac{-n_1+h_1}{1-\theta} + \frac{h_2}{\theta} + \frac{-n_2+h_2}{1-\theta} \nonumber \\\\
&= \frac{h_1 + h_2}{\theta}+  \frac{h_1 + h_2 - n_1 -n_2}{1-\theta} \nonumber \\\\
-\theta(h_1 + h_2 - n_1 -n_2) &= (1 - \theta)(h_1 + h_2) \nonumber \\\\
-\theta h_1  -\theta h_2 + \theta n_1  + \theta n_2 &= h_1 - \theta h_1 + h_2 - \theta h_2 \nonumber \\\\
\theta (n_1  + n_2) &= h_1 + h_2 \nonumber \\\\
\theta  &= \frac{h_1 + h_2}{n_1+ n_2} \nonumber \\\\
\theta_{MLE} &=  \frac{\sum^k_{i=1}h_i}{\sum^k_{i=1}n_i}
\label{eq:theta-binom-k}
\end{align}
\end{spreadlines}
$$

Therefore, to obtain $\theta_{MLE}$ when there are $k$ coin flipping sessions, the sum of heads,$\sum^k_{i=1} h_i$, is divided by the sum of coin flips across the sessions, $\sum^k_{i=1} n_i$. In the current example where $\mathbf{h} = \[1, 6, 4, 7, 3, 4 ,5, 10, 5, 3\]$ and each session has 10 coin flips, the maximum likelihood estimate for the probability of heads, $\theta_{MLE}$, is .48 (see lines below).

```r 
h <- c(1, 6, 4, 7, 3, 4 ,5, 10, 5, 3)
theta_mle <- sum(h)/sum(rep(x = 10, times = 10))
theta_mle
```
<pre><code class='r-code'>[1] 0.48
</code></pre>


## Maximum Likelihood Estimation for the Gaussian Case 

To explain maximum likelihood estimation for the Gaussian case, let's consider a new example where a researcher measures the heights of 100 males, $\mathbf{y} \in \mathcal{R}^{100}$. From previous studies, the researcher believes heights to be normally distributed and, thus, estimates a mean, $\mu$, and standard deviation, $\sigma$, for the population heights of males. To obtain population estimates for the mean and standard deviation, the Gaussian function shown below in Equation \ref{eq:gauss} can be used:

$$ 
\begin{align}
P(y_i|\sigma, \mu) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\big(\frac{y_i - \mu}{\sigma}\big)^2},
\label{eq:gauss}
\end{align}
$$
where the probability of observing a $y_i$ score given a population mean, $\mu$, and standard deviation, $\sigma$, is computed, $P(y_i|\sigma, \mu)$. Because the researcher is interested in determining the parameter values that most likely produced the data, parameter values will be varied and the data will be fixed. Thus, likelihoods and not probabilities will be used (see [Likelihood are not probabilities](#like-prob)). Although Equation \ref{eq:gauss} will still be used to compute likelihoods, I will rewrite Equation \ref{eq:gauss} to explicitly indicate that likelihoods will be computed, as shown below in Equation \ref{eq:gauss-like}:

$$
\begin{align}
L(\sigma, \mu|\mathbf{y_i}) =  \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\big(\frac{y_i - \mu}{\sigma}\big)^2}.
\label{eq:gauss-like}
\end{align}
$$

Importantly, Equation \ref{eq:gauss-like} above only computes the likelihood given one $y_i$ data point. Because the researcher wants to determine the parameter values that produced all the 100 data points, $y_i \in \mathbf{y}$, Equation \ref{eq:gauss-like} must be used each for each data point and all the resulting likelihood values must be multiplied together. Thus, a product of likelihoods must be computed to obtain the likelihood of the parameters given the entire data set, $L(\sigma, \mu|\mathbf{y})$, as shown below in Equation \ref{eq:gauss-prod}: 

$$
\begin{align}
L(\sigma, \mu|\mathbf{y_i}) &=  \prod^n_{i=1}\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\big(\frac{y_i - \mu}{\sigma}\big)^2}.
\label{eq:gauss-prod}
\end{align} 
$$

As in the binomial case, the likelihood equation must be transformed to a $\log$ scale to prevent underflow and to simplify the derivation of the partial derivatives. Given that the equation contains Euler's number, $e$, I will use log of base $e$ or the natural log, $\ln$, to further simplify the derivatives. Before applying the log transformation, Equation \ref{eq:gauss-prod} can be simplified to yield Equation \ref{eq:gauss-prod-s} below:  

$$ 
\begin{spreadlines}{0.5em}
\begin{align}
L(\sigma, \mu|\mathbf{y_i}) &=  \prod^n_{i=1}\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\big(\frac{y_i - \mu}{\sigma}\big)^2}
 \nonumber \\\\
&= \sigma^{-n}(2\pi)^{-\frac{n}{2}}e^{\Big(-\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mu)^2\Big)}.
\label{eq:gauss-prod-s}
\end{align} 
\end{spreadlines}
$$

With a simplified form of Equation \ref{eq:gauss-prod}, Equation \ref{eq:gauss-prod-s} can now be converted to a log scale by using the product rule and then the power rule to obtain the log-likelihood Gaussian function shown below in Equation \ref{eq:log-gauss}. 

$$ 
\begin{spreadlines}{0.5em}
\begin{align}
\text{Apply product rule }\text{ } &\Rightarrow \ln(\sigma^{-n}) + \ln \Big((\sqrt{2\pi})^{-\frac{n}{2}}\Big) + \ln \Big(e^{\big(-\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mu)^2\big)}\Big) \nonumber \\\\
\text{Apply power rule }\text{ } &\Rightarrow -n\ln(\sigma) -\frac{n}{2}\ln(\sqrt{2\pi}) -\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mu)^2 \ln(e) \nonumber \\\\
\ln L(\sigma, \mu|\mathbf{y})  &= -n\ln(\sigma) -\frac{n}{2}\ln (\sqrt{2\pi}) -\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mu)^2
\label{eq:log-gauss}
\end{align}
\end{spreadlines}
$$

The maximum likelihood estimate functions for the mean, $\mu$, and standard deviation, $\sigma$, can now be obtained by taking the derivative of the log-likelihood function with respect to each parameter. The derivation below solves for $\mu$. 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\frac{\delta \ln L(\sigma, \mu|\mathbf{y})}{\delta \mu} &= \frac{\delta}{\delta \mu} \Bigg(-n\ln(\sigma) -\frac{n}{2}\ln (\sqrt{2\pi}) -\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mu)^2\Bigg) \nonumber \\\\
&= 0 -0 -\frac{1}{2\sigma^2}\sum_{i=1}^n 2(y_i - \mu) \frac{\delta}{\delta \mu}(y_i - \mu) \nonumber \\\\
&= -\frac{1}{2\sigma^2}\sum_{i=1}^n 2(y_i - \mu) \cdot -1 \nonumber \\\\
&= \frac{1}{\sigma^2}\sum_{i=1}^n (y_i - \mu) \nonumber \\\\
\text {Set  } \frac{\delta \ln L(\sigma, \mu|\mathbf{y}) }{\delta \mu} = 0 \nonumber \\\\
0 &= \frac{1}{\sigma^2}\sum_{i=1}^n (y_i - \mu) \nonumber \\\\
0 &= \sum_{i=1}^n y_i - \sum_{i=1}^n \mu \nonumber \\\\
0 &= \sum_{i=1}^n y_i - n\mu \nonumber \\\\
\mu_{MLE} &= \frac{1}{n}\sum_{i=1}^n y_i 
\label{eq:mean-mle}
\end{align}
\end{spreadlines}
$$
Therefore, Equation \ref{eq:mean-mle} above shows that the maximum likelihood estimate for the mean can be obtained by simply computing the mean of the observed $y_i$ scores. The derivation below solves for $\sigma$. 


$$
\begin{spreadlines}{0.5em}
\begin{align}
\frac{\delta \ln L(\sigma, \mu|\mathbf{y})}{\delta \sigma} &= \frac{\delta}{\delta \sigma} \Bigg(-n\ln(\sigma) -\frac{n}{2}\ln (\sqrt{2\pi})  - \frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mu)^2\Bigg) \nonumber \\\\
&= -\frac{n}{\sigma} + 0 - \frac{1}{2}(-2\sigma^{-3})\sum_{i=1}^n (y_i - \mu)^2 \nonumber \\\\
&= -\frac{n}{\sigma}  + \frac{\sum_{i=1}^n (y_i - \mu)^2}{\sigma^3}  \nonumber \\\\
&= \frac{1}{\sigma^3}(\sum_{i=1}^n(y_i - \mu)^2 - n\sigma^2)  \nonumber \\\\
\text {Set }\frac{\delta \ln L(\sigma, \mu|\mathbf{y})}{\delta \sigma} &= 0  \nonumber \\\\
0 &= \sum_{i=1}^n(y_i - \mu)^2 - n\sigma^2  \nonumber \\\\
n\sigma^2 &= \sum_{i=1}^n(y_i - \mu)^2 \nonumber \\\\
\sigma_{MLE} &= \sqrt{\frac{1}{n} \sum_{i=1}^n(y_i - \mu)^2}
\label{eq:mle-sigma}
\end{align}
\end{spreadlines}
$$
Therefore, Equation \ref{eq:mle-sigma} above shows that the the maximum likelihood estimate for the standard deviation parameter, $\sigma$, is the square root of the average squared deviation from the mean observed score. 


Thus, as in the binomial case, maximum likelihood estimation provides a simple function for calculating maximum likelihood estimates for the Gaussian parameters. 


# Conclusion 

In conclusion, probabilities and likelihoods are fundamentally different. Probabilities sum to one, whereas likelihoods do not sum to one. Thus, likelihoods cannot be interpreted as probabilities. Although likelihoods cannot be interpreted as probabilities, they can be used to determine parameter values that most likely produce observed data sets (maximum likelihood estimates). Maximum likelihood estimation provides an efficient method for determining maximum likelihood estimates and was applied in the binomial and Gaussian cases.
 

# References


{{< bibliography cited >}}

# Appendix A: Proof That the Binomial Function is a Probability Mass Function  {#proof-pmf}

To prove that the binomial function is a probability mass function, two outcomes must be shown: 1) all probability values are non-negative and 2) the sum of all probabilities is one. 

For the first condition, the impossibility of negative values occurring in the binomial function becomes obvious when individually considering the binomial coefficient, $n \choose h$, and the binomial factors, $\theta^h (1-\theta)^{n-h}$. With respect to the binomial coefficient, $n \choose h$, it is always nonnegative because it is the product of two non-negative numbers; the number of trials, $n$, and the number of heads, $h$, can never be negative. With respect to the binomial factors, the resulting value is always nonnegative because all the constituent terms are nonnegative; in addition to the number of trials and heads ($n, h$, respectively),  the probability of heads and tails are also always nonnegative ($\theta, (1-\theta) \in \[0,1\]$). Therefore, probabilities can be conceptualized as the product of a nonnegative binomial coefficient and a nonnegative binomial factor, and so are always nonnegative.

For the second condition, the equality stated below in Equation \ref{eq:binomial-sum-one} must be proven: 

\begin{align}
1 = \sum^n_{h=0} {n \choose h} \theta^h(1-\theta)^{n-h}.  
\label{eq:binomial-sum-one}
\end{align}

Importantly, it can be proven that all probabilities sum to one by using the binomial theorem, which states below in Equation \ref{eq:binomial} that 

\begin{align}
(a + b)^n =  \sum^n_{k=0} {n \choose k} a^k(b)^{n-k}. 
\label{eq:binomial}
\end{align}

Given the striking resemblance between the binomial function in Equation \ref{eq:binomial-sum-one} and the binomial theorem in Equation \ref{eq:binomial}, it is possible to restate the binomial theorem with respect to the variables in the binomial function. Specifically, we can let $a = \theta$ and $b = 1-\theta$, which returns the proof as shown below: 

\begin{spreadlines}{0.5em}
\begin{align*}
(\theta + 1 -\theta)^n &= \sum^n_{h=0} {n \choose h} \theta^h(1-\theta)^{n-h} \\\\ \nonumber
1 &= \sum^n_{h=0} {n \choose h} \theta^h(1-\theta)^{n-h} \qquad\qquad _\blacksquare   \nonumber 
\end{align*}
\end{spreadlines}


For a proof of the binomial theorem, see [Appendix E](#proof-binomial). 


# Appendix B: Proof That Likelihoods are not Probabilities  {#proof-likelihood}

As a reminder, although the same formula is used to compute likelihoods and probabilities, the variables allowed to vary and those that are fixed differ when computing likelihoods and probabilities. With probabilities, the parameters are fixed (i.e., $\theta$) and the data are varied ($h, n$). With likelihoods, however, the data are fixed ($h, n$) and the parameters are varied ($\theta$). To prove that likelihoods are not probabilities, we have to prove that likelihoods do not satisfy one of the two conditions required by probabilities (i.e., likelihoods can have negative values or likelihoods do not sum to one). Given that likelihoods are calculated with the same function as probabilities and probabilities can never be negative (see [Appendix A](#proof-pmf)), likelihoods likewise can never be negative. Therefore, to prove that likelihoods are not probabilities, we must prove that likelihoods do not always sum to one. Thus, the following proposition must be proven: 
 
$$
\begin{align}
 \int_{0}^1{n \choose h} \theta^h(1 - \theta)^{n-h}  d\theta\neq 1. 
 \label{eq:int-sum-likelihood-binomial}
\end{align}
$$
That is, the integral of the binomial function with respect to $\theta$ does not equal one. To prove this proposition, it is important to realize that $ \int_0^1 \theta^h(1-\theta)^{n-h}$ can be restated in terms of the beta function, $\mathrm{B}(x, y)$, which is shown below. 
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
At this point, another proof becomes important because it allows us to express the beta function in terms of another function that will, ultimately, allow us to simplify Equation \ref{eq:beta-restate} and prove that likelihoods do not sum to one. Specifically, the beta function, $\mathrm{B}(x, y)$ can be stated in terms of the gamma function $\Gamma$ such that 

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
&= \frac{1}{n+1} \qquad\qquad _\blacksquare 
\label{eq:likelihood-proof}  
\end{align} 
\end{spreadlines}
$$

Therefore, binomial likelihoods sum to a multiple of $\frac{1}{1+n}$, where the multiple is the number of integration steps. The R code block below provides an example where the integral can be shown to be a multiple of the value in Equation \ref{eq:likelihood-proof}. In the example, the integral of the likelihood is taken over 100 equally spaced steps. Thus, the sum of likelihoods should be $100\frac{1}{1+n} = 9.09$, and this turns out to be true in the R code block below (lines <a href="#131">131--136</a>). 

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

Equation \ref{eq:beta-gamma-proof} below will be proven:

$$
\begin{align}
\mathrm{B}(x, y) = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}.
\label{eq:beta-gamma-proof}
\end{align}
$$
To begin, let's write out the expansions of the gamma function, $\Gamma(x)$, and the numerator of Equation \ref{eq:beta-gamma-proof}, $\Gamma(x)\Gamma(y)$, where 
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
Equation \ref{eq:gamma-function} shows the gamma function, $\Gamma(x)$, which will be useful as a reference and Equation \ref{eq:gamma-numerator} shows the expansion of the numerator in Equation \ref{eq:beta-gamma-proof}. To prove Equation \ref{eq:beta-gamma-proof}, we will begin by changing the variables of $s$ and $t$ in Equation \ref{eq:gamma-numerator} by reexpressing them in terms of $u$ and $v$. Importantly,  when changing variables in a double integral, the formula below in Equation \ref{eq:double-integral} must be followed: 

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
With $\det\mathbf{J}(u, v)$ computed, we can no express the new function with the changed variables, as shown below in Equation \ref{eq:gamma-reexp}. 

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

Therefore, the original integration limits of 0 to $\infty$ of $s$ and $t$ produce integration limits 0 to $\infty$ for $u$ and 0 to 1 for $v$.  Recalling the gamma function (Equation \ref{eq:gamma-function} and the beta function (Equation \ref{eq:beta-function}), the beta function can now be expressed in terms of the gamma function, proving Equation \ref{eq:beta-gamma-proof}. 

\begin{spreadlines}{0.5em}
\begin{align*}
\Gamma(x)\Gamma(y) &= \int_0^1 \int_0^\infty u^{x+y-1} e^{-u} v^{x-1} (1 - v)^{y-1} \,du\,dv \\\\
&=  \int_0^\infty  u^{x+y-1} e^{-u}\text{ } du \int_0^1v^{x-1} (1 - v)^{y-1} \,dv \\\\
&=  \Gamma(x + y)\mathrm{B}(x,y) \\\\
\mathrm{B}(x,y) &= \frac{\Gamma(x)\Gamma(y)}{ \Gamma(x + y)} \qquad\qquad _\blacksquare 
\end{align*}
\end{spreadlines}



# Appendix D: Proof of Relation Between Gamma and Factorial Functions  {#proof-gamma-factorial}

To prove the following proposition in Equation \ref{eq:gamma-factorial} that

$$
\begin{align}
\Gamma(x) &= \int_0^\infty t^{x-1}e^{-t} dt =(x-1)!, 
\label{eq:gamma-factorial}
\end{align}
$$
it is first helpful to prove the proposition below in Equation \ref{eq:gamma-pre-fac} that 
$$
\begin{align}
\Gamma(\alpha + 1) &= \alpha\Gamma(\alpha )
\label{eq:gamma-pre-fac}
\end{align}
$$
To prove Equation \ref{eq:gamma-pre-fac}, we first expand Equation \ref{eq:gamma-pre-fac} in Equation \ref{eq:gamma-expand} and then simplify Equation \ref{eq:gamma-expand} using integration by parts such that 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\Gamma(\alpha + 1) &= \int^\infty_0 t^\alpha e^{-t} \text{ } dt
\label{eq:gamma-expand} \\\\ 
\int u \text{ }dv &= uv - \int v \text{ } du.  
\label{eq:int-parts} \\\\
\text{Let } u &= t^\alpha \text{, } dv = e^{-t} \text{ } dt \text{, } \nonumber \\\\
du &= \alpha t^{\alpha - 1}\text{, and } v = -e^{-t}. \nonumber\\\\
\int u \text{ }dv &= -t^\alpha e^{-t}|^\infty_0 - \int^\infty_0(-e^{-t}) \alpha t^{\alpha - 1} 
\label{eq:gamma-int-parts}
\end{align}
\end{spreadlines}
$$
To simplify Equation \ref{eq:gamma-int-parts}, I will first focus on the evaluation of $-t^\alpha e^{-t}$ between $\infty$ and $0$ below. At $t = \infty$, 

$$
\begin{align}
-t^\alpha e^{-t} = \frac{-\infty^{\alpha}}{e^\infty}, 
\label{eq:inf-eval}
\end{align}
$$
and because $e^{\infty}$ approaches $\infty$ faster than $-\infty^\alpha$ approaches $-\infty$, Equation \ref{eq:inf-eval} becomes zero. At $t = 0$, 
$$
\begin{align*}
-t^\alpha e^{-t} = \frac{-0^{\alpha}}{e^0} = \frac{0}{1} = 0.
\end{align*}
$$
Therefore, Equation \ref{eq:gamma-int-parts} simplifies to 

$$
\begin{spreadlines}{0.5em}
\begin{align*}
\int u \text{ }dv &= 0 - 0 - \int^\infty_0(-e^{-t}) \alpha t^{\alpha - 1}  \\\\
&= \alpha  \int^\infty_0t^{\alpha - 1} e^{-t}  \\\\
&= \alpha \Gamma(\alpha) \qquad\qquad _\blacksquare
\end{align*} 
\end{spreadlines}
$$
Having proven that $\Gamma(\alpha + 1) = \alpha\Gamma(\alpha)$, it becomes easy to prove Equation \ref{eq:gamma-factorial} which states that $\Gamma(x) = (x-1)!$. If I continue to expand the gamma function, $\Gamma(x-n)$, where $n = x -1$, I will obtain 

$$
\begin{spreadlines}{0.5em}
\begin{align*}
\Gamma(x) &= (x-1)\Gamma(x-1) \\\\
\Gamma(x-1) &= (x-2)\Gamma(x-2) \\\\
&\vdots \\\\
\Gamma(x-n) &= (1)\Gamma(1) 
\end{align*}
\end{spreadlines}
$$
To evaluate $\Gamma(1)$, I write out its expansion and show that 

$$
\begin{spreadlines}{0.5em}
\begin{align*}
\Gamma(1) &= \int^\infty_0t^{1-1}e^{-t} \text{ } dt \\\\
&= e^{-t}|^\infty_0 \\\\
&= e^{-\infty} - e^{0} \\\\ 
&= 0 + 1 = 1\\\\
\end{align*}
\end{spreadlines}
$$
Therefore,  $\Gamma(x)$ expands to $(x-1)!$ because the last term will inevitably be  $1\times\Gamma(1) = 1$. 

$$
\begin{spreadlines}{0.5em}
\begin{align*}
\Gamma(x) = (x-1)(x-2)(x-3)...(x-n)\Gamma(x-n)  \\\\
=(x-1)(x-2)(x-3)...(1)\Gamma(1) \\\\
=(x-1)(x-2)(x-3)...(1)\Gamma(1) \\\\
=(x-1)! \qquad\qquad _\blacksquare
\end{align*}
\end{spreadlines}
$$

# Appendix E: Proof of Binomial Theorem  {#proof-binomial}

The binomial theorem provided below in Equation \ref{eq:binomial2} states that 

$$
(x + y)^n = \sum^n_{k=0} {n \choose k} x^{n-k}y^k.
\label{eq:binomial2}
$$
I will prove the binomial theorem using induction. Thus, I will first prove the binomial theorem in a base where $n=1$ so that I can later generalize the proof with a larger number of $n+1$.  In the base case, the binomial theorem is proven such that 

$$
\begin{spreadlines}{0.5em}
\begin{align*}
x + y &= {1 \choose 0} x^{1-0}y^0 + {1 \choose 1} x^{1-1}y^1 \\\\
&= x + y. 
\end{align*}
\end{spreadlines}
$$
Now, I will prove the binomial theorem with $n + 1$. Thus, 

$$
(x + y)^{n+1} = \sum^{n+1}_{k=0} {n+1 \choose k} x^{n+1-k}y^k.
\label{eq:binomial-induction}
$$
I now expand the left-hand side of Equation \ref{eq:binomial-induction}, to obtain 

$$
\begin{spreadlines}{0.5em}
\begin{align}
(x + y)^{n+1} &=(x + y)(x + y)^n \nonumber \\\\
 &= (x+y) \sum^n_{k=0} {n \choose k} x^{n-k}y^k \nonumber  \\\\
&=x\sum^n_{k=0} {n \choose k} x^{n-k}y^k + y\sum^n_{k=0} {n \choose k} x^{n-k}y^k \nonumber \\\\
&=\sum^n_{k=0} {n \choose k} x^{n+1-k}y^k + \sum^n_{k=0} {n \choose k} x^{n-k}y^{k+1} \nonumber \\\\
&=\sum^n_{k=0} {n \choose k} x^{(n+1)-k}y^k + \sum^{n+1}_{k=1} {n \choose k-1} x^{n-(k-1)}y^{(k-1)+1}
\label{eq:binom-sums}
\end{align}
\end{spreadlines}
$$
Now I, respectively, remove $k = 0$ and $k = n+1$ from the first and second terms of Equation \ref{eq:binom-sums} so that the sums iterate over the same range of $k=1$ to $k = n$.

$$
\begin{spreadlines}{0.5em}
\begin{align}
&= \binom{n}{0} x^{n+1-0}y^0 + \binom{n}{n} x^{n-(n + 1 -1)}y^{(n+1-1)+1} + \sum^n_{k=1} \binom{n}{k} x^{(n+1)-k}y^k + \sum^n_{k=1} \binom{n}{k-1} x^{n-(k-1)}y^{(k-1)+1} \nonumber \\\\
&=x^{n+1} + y^{n+1} + \sum^n_{k=1} {n \choose k} x^{n+1-k}y^k + \sum^n_{k=1} {n \choose k-1}  x^{n-k+1}y^{k}.
\label{eq:add-sums}
\end{align}
\end{spreadlines}
$$

I then apply Pascal's rule to simplify the addition of summations in Equation \ref{eq:add-sums} to obtain 

$$
\begin{spreadlines}{0.5em}
\begin{align*}
&=x^{n+1} + y^{n+1} + \sum^n_{k=1} {n+1 \choose k} x^{n+1-k}y^k \\\\
&= {n+1 \choose 0} x^{n+1-0}y^0 + {n+1 \choose n+1} x^{n+1-(n+1)}y^{n+1} + \sum^n_{k=1} {n+1 \choose k} x^{n+1-k}y^k \\\\
&= \sum^{n+1}_{k=0} {n+1 \choose k}x^{n-k+1}y^k \quad\quad _\blacksquare
\end{align*}
\end{spreadlines}
$$


