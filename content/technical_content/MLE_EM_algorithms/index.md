---
title: "Probability, Likelihood, and Maximum Likelihood Estimation" 
draft: false
summary: 'Explanation of post' 
article_type: technical
output:
  bookdown::html_document2:
     keep_md: true
always_allow_html: true
header-includes: 
  - \usepackage{amsmath}
bibFile: content/technical_content/em_algorithm/biblio.json    
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


# Appendix D: Proof of Relation Between Gamma and Factorial Functions  {#proof-gamma-factorial}

# Appendix E: Proof of Binomial Theorem  {#proof-binomial}



