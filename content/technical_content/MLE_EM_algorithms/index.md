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
bibFile: content/technical_content/MLE_EM_algorithms/biblio.json    
tags: []
---   




# Code Chunks
R code chunk below (see lines <a href="#1">1--22</a>). 

```r {hl_lines=[1,2,5]}
#this is a comment  more comment my website m , . , y 
website my website my website
#my website my website my website my website my website 
my website my website my   
#my website my website my website
print('my website my website my, , , ,. `  website my 
website my website my website my website
      website my website my   website my website my 
      website my website my website my 
      website')  

print(1 + 2)
mean(x = c(1, 2))
print('another')
print(NULL)
print(NA)
print(TRUE)
"\n"
list('number_measurements' = c(5, 7, 9, 11),
     'spacing' = c('equal', 'time_inc'))

var <- function(x1, x2){

  if (x1 > 2) {print(x1)} 
  else {print (x2)}
}
```
This is inline R code:{{< inline-src r >}}print(NULL){{< /inline-src >}}.

Python code chunk below (see lines <a href="#23">23--31</a>). 
```r {hl_lines=[1,3,4],language=python}
tup = ['Groucho', 'Marx', 'Xavier', 'Xavier', 'Xavier', 'Xavier', 'Xavier', 'Xavier', 'Xavier', 'Xavier']
list_ex = list(["food", 538, True, 1.454, "food", 538, True, 1.454, "food", 538, True, 1.454, "food", 538, True, 1.454])
sorted(tup)

list_ex2 = list([1 + 2, "a" * 5, 3])  

#deleting variables 
del(list_ex2)
list_ex.count(2)  
```
This is inline Python code: {{< inline-src python >}}print('NULL'){{< /inline-src >}}.

SQL code chunk below (see lines <a href="#32">32--43</a>). 
```r {hl_lines=[1,3,4],language=sql}
CREATE TABLE person 
  (person_id SMALLINT UNSIGNED,
  fname VARCHAR(20),
  lname VARCHAR(20),
  eye_color ENUM('BR','BL','GR'),
  birth_date DATE,
  street VARCHAR(30),
  city VARCHAR(20),
  state VARCHAR(20),
  country VARCHAR(20),
  postal_code VARCHAR(20),
CONSTRAINT pk_person PRIMARY KEY (person_id)
```
This is inline SQL code: {{< inline-src sql >}} CREATE TABLE person {{< /inline-src >}}.


Javascript code chunk below (see lines <a href="#44">44--59</a>). 
```r {hl_lines=[2,3,4],language=java}
let codeTable = document.createElement("table");
codeTable.setAttribute('id', "codeTable");

//add rows to table by adding each element of lines
for (let t = 0; t < lines.length; t++) {
  let row = codeTable.insertRow(-1);

  let newCell1 = row.insertCell(0); //insert line number
  let newCell2 = row.insertCell(1);
  let newCell3 = row.insertCell(2);

  newCell1.innerHTML = "<span class= 'line-number' data-number='" + (t+1)  + "'" + "id = '" + 
    (t+1) + "'></span>";
  newCell2.innerHTML = lines[t];
  newCell3.innerHTML = "";
}
```
This is inline Javascript code: {{< inline-src js >}}let codeTable = document.createElement("table");{{< /inline-src >}}. 

CSS code chunk below (see lines <a href="#60">60--65</a>). 
```r {hl_lines=[2,3,4],language=css}
div[language ='java'] code[data-lang='r'] table td:nth-child(2) { width: 85%;position: relative;

  background-color:  rgba(255,105,130, 0.20);
  border-left: 2pt solid rgba(255,105,130, 0.50);
  padding: 0;
}
```
This is inline CSS code: {{< inline-src css >}} background-color:  rgba(255,105,130, 0.20);{{< /inline-src >}}.

HTML code chunk below (see lines <a href="#66">66--68</a>). 

```r {hl_lines=[2,3],language=html}
<script src="{{ "js/external_links.js" | relURL }}"></script>
<script src="{{ "js/number_tables.js" | relURL }}"></script>
<script src="{{ "js/number_figures.js" | relURL }}"></script>
```
This is inline HTML code: {{< inline-src html >}} <script src="{{ "js/external_links.js" | relURL }}"></script>{{< /inline-src >}}.

Bash code chunk below (see lines <a href="#57">69--71</a>). 

```r {hl_lines=[1],language=bash}
ls
 
cd ~/Desktop/Home/blog_posts
```
This is inline bash code: {{< inline-src bash >}}cd ~/Desktop/Home/blog_posts{{< /inline-src >}}.








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

Figure \ref{fig:prob-mass-binom} shows the probability mass function that results with an unbiased coin ($\theta = 0.50$) and ten coin flips ($n = 10$). In looking across the probability values of obtaining each number of heads (x-axis), 5 heads is the most likely value, as indicated by the emboldened number on the x-axis and the bar above it with a darker blue color. As an aside, the R code below verifies the two conditions of probability mass functions for the current example (for a mathematical proof, see [Appendix A](#proof-pmf)). 

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
# References{#.unnumbered}


{{< bibliography cited >}}

# Appendix A: Proof That the Binomial Function is a Probability Mass Function  {#proof-pmf}
<div style="display:none">\(\setSection{A}\)</div>

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
\Gamma(x) = (x - 1)!.
\end{align}
$$

Given that the gamma function can be stated as a factorial, Equation \ref{eq:binomial-gamma} can be now be written with factorial terms and simplified to prove that likelihoods do not sum to one. 


\begin{spreadlines}{0.5em}
\begin{align}
 \int_0^1 L(\theta|h, n) \phantom{c} d\theta &= \frac{n!}{h!(n-h)!}\frac{h!(n-h)!}{(n + 1)!} \\\\ 
&= \frac{n!}{(n + 1)!}  \\\\ 
&= \frac{1}{n+1}. \label{eq:likelihood-proof} 
\end{align} 
\end{spreadlines}


Therefore, binomial likelihoods sum to a multiple of $\frac{1}{1+n}$, where the multiple is the number of integration steps. The R code block below provided an example where the integral can be shown to be a multiple of the value in Equation \ref{eq:likelihood-proof}. In the example, the integral of the likelihood is taken over 100 equally spaced steps. Thus, the sum of likelihoods should be $100\frac{1}{1+n} = 9.09$, and this turns out to be true in the code below. 




