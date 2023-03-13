---
title: "Example Post" 
draft: true
summary: 'Explanation of post' 
date: "2023-03-12"
article_type: technical
output:
  bookdown::html_document2:
     keep_md: true
always_allow_html: true
header-includes: 
  - \usepackage{amsmath}
bibFile: content/technical_content/example_post/biblio.json    
tags: []
---   







# Code Chunks
R code chunk below (see lines <a href="#1">1--22</a>). 

```r {hl_lines=[1,2,5]}
#this is a comment  more comment my website m , . , y website my website my website
#my website my website my website my website my website my website my website my   
#my website my website my website
print('my website my website my, , , ,. `  website my website my website my website my website
      website my website my   website my website my website my website my website my 
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


<pre><code class='r-code'>[1] "my website my website my, , , ,. `  website my website my website my website my website my website my website my website my website my website my website my website my website"
[1] 3
[1] 1.5
[1] "another"
NULL
[1] NA
[1] TRUE
[1] "\n"
$number_measurements
[1]  5  7  9 11

$spacing
[1] "equal"    "time_inc"
</code></pre>



<pre><code class='python-code'>['Groucho', 'Marx', 'Xavier']
0
</code></pre>


# In-Text Citations 
Some explanatory text {{< cite "fine2019;george2000" >}}. More text. {{< cite "fine2019;cole2003" >}}
*italic text*
[link](https://github.com/gohugoio/hugo/issues/9442)

If you liked how these _"generics"_ work in SystemVerilog and how the looks, check out the


# Figures 

Figure \ref{fig:prob-mass-binom} is below. 


<div class="figure">
  <div class="figDivLabel">
    <caption>
      <span class = 'figLabel'>Figure \ref{fig:prob-mass-binom}<span> 
    </caption>
  </div>
   <div class="figTitle">
    Probability Mass Function With an Unbiased Coin (<span class = "theta">&theta;</span> = 0.50) and Ten Coin Flips (n = 10)</span> 
  </div>
    <img src="images/pmf_plot.png" width="100%" height="100%"> 
  
  <div class="figNote">
      <span><em>Note. </em> Number emboldened on the x-axis indicates the number of heads that is most likely to occur with an unbiased coin and 10 coin flips, with the corresponding bar in darker blue  indicating the corresponding probability.</span> 
  </div>
</div>

# Tables

see [section](#section)

 Table \ref{tab:parameterValues}  Table \ref{tab:parameterValues}
Another paragraph begins and the spacing should not be too small from table above. 

 Table \ref{tab:parameterValues1}
 Table \ref{tab:parameterValues}
 Table \ref{tab:parameterValues1}
 
<table class="table" style="margin-left: auto; margin-right: auto;border-bottom: 0;">
<caption>(\#tab:parameterValues)Values Used for Multilevel Logistic Function Parameters</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> Parameter Means </th>
   <th style="text-align:center;"> Value </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Baseline, $\theta$ </td>
   <td style="text-align:center;"> 3.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Maximal elevation, $\alpha$ </td>
   <td style="text-align:center;"> 3.32 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Days-to-halfway elevation, $\upbeta$ </td>
   <td style="text-align:center;"> 180.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;padding-bottom: 1rem; border-bottom: 1.5px solid" indentlevel="1"> Triquarter-halfway delta, $\upgamma$ </td>
   <td style="text-align:center;padding-bottom: 1rem; border-bottom: 1.5px solid"> 20.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold; border-bottom: 1.5px solid"> Variability and Covariability Parameters (in Standard Deviations) </td>
   <td style="text-align:center;font-weight: bold; border-bottom: 1.5px solid">  </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Baseline standard deviation, $\uppsi_{\uptheta}$ </td>
   <td style="text-align:center;"> 0.05 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Maximal elevation standard deviation, $\uppsi_{\upalpha}$ </td>
   <td style="text-align:center;"> 0.05 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Days-to-halfway elevation standard deviation, $\uppsi_{\upbeta}$ </td>
   <td style="text-align:center;"> 10.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Triquarter-halfway delta standard deviation, $\uppsi_{\upgamma}$ </td>
   <td style="text-align:center;"> 4.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Baseline-maximal elevation covariability, $\uppsi_{\uptheta\upalpha}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Baseline-days-to-halfway elevation covariability, $\uppsi_{\uptheta\upbeta}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Baseline-triquarter-halfway delta covariability, $\uppsi_{\uptheta\upgamma}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Maximal elevation-days-to-halfway elevation covariability, $\uppsi_{\upalpha\upbeta}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Maximal elevation-triquarter-halfway delta covariability, $\uppsi_{\upalpha\upgamma}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Days-to-halfway elevation-triquarter-halfway delta covariability, $\uppsi_{\upbeta\upgamma}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Residual standard deviation, $\uppsi_{\upepsilon}$ </td>
   <td style="text-align:center;"> 0.05 </td>
  </tr>
</tbody>
<tfoot>
<tr><td style="padding: 0; " colspan="100%"><span style="font-style: italic;"> </span></td></tr>
<tr><td style="padding: 0; " colspan="100%">
<sup></sup> <em>Note</em>. The difference between $\alpha$ and $\theta$ corresponds to the 50$\mathrm{^{th}}$ percentile Cohen's $d$ value of 0.32 in organizational psychology (Bosco et al., 2015). Additional text to see what happens</td></tr>
</tfoot>
</table>

<table class="table" style="margin-left: auto; margin-right: auto;border-bottom: 0;">
<caption>(\#tab:parameterValues1)Values Used for Multilevel Logistic Function Parameters (see Table \ref{tab:parameterValues})</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> Parameter Means </th>
   <th style="text-align:center;"> Value </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Baseline, $\theta$ </td>
   <td style="text-align:center;"> 3.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Maximal elevation, $\alpha$ </td>
   <td style="text-align:center;"> 3.32 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Days-to-halfway elevation, $\upbeta$ </td>
   <td style="text-align:center;"> 180.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;padding-bottom: 1rem; border-bottom: 1.5px solid" indentlevel="1"> Triquarter-halfway delta, $\upgamma$ </td>
   <td style="text-align:center;padding-bottom: 1rem; border-bottom: 1.5px solid"> 20.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold; border-bottom: 1.5px solid"> Variability and Covariability Parameters (in Standard Deviations) </td>
   <td style="text-align:center;font-weight: bold; border-bottom: 1.5px solid">  </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Baseline standard deviation, $\uppsi_{\uptheta}$ </td>
   <td style="text-align:center;"> 0.05 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Maximal elevation standard deviation, $\uppsi_{\upalpha}$ </td>
   <td style="text-align:center;"> 0.05 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Days-to-halfway elevation standard deviation, $\uppsi_{\upbeta}$ </td>
   <td style="text-align:center;"> 10.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Triquarter-halfway delta standard deviation, $\uppsi_{\upgamma}$ </td>
   <td style="text-align:center;"> 4.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Baseline-maximal elevation covariability, $\uppsi_{\uptheta\upalpha}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Baseline-days-to-halfway elevation covariability, $\uppsi_{\uptheta\upbeta}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Baseline-triquarter-halfway delta covariability, $\uppsi_{\uptheta\upgamma}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Maximal elevation-days-to-halfway elevation covariability, $\uppsi_{\upalpha\upbeta}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Maximal elevation-triquarter-halfway delta covariability, $\uppsi_{\upalpha\upgamma}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Days-to-halfway elevation-triquarter-halfway delta covariability, $\uppsi_{\upbeta\upgamma}$ </td>
   <td style="text-align:center;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;padding-left: 4em;" indentlevel="1"> Residual standard deviation, $\uppsi_{\upepsilon}$ </td>
   <td style="text-align:center;"> 0.05 </td>
  </tr>
</tbody>
<tfoot>
<tr><td style="padding: 0; " colspan="100%"><span style="font-style: italic;"> </span></td></tr>
<tr><td style="padding: 0; " colspan="100%">
<sup></sup> <em>Note</em>. The difference between $\alpha$ and $\theta$ corresponds to the 50$\mathrm{^{th}}$ percentile Cohen's $d$ value of 0.32 in organizational psychology (Bosco et al., 2015). see Table \ref{tab:parameterValues} and Figure \ref{fig:gg-oz-plot1}.</td></tr>
</tfoot>
</table>


# Excerpt Section Example

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
\begin{align}
P(h = 4|\theta = 0.50, n = 10) &= {10 \choose 4}(0.50)^{4}(1-0.50)^{(10-4)}   \nonumber \\\\
&= \frac{10!}{4! (10 - 4)!}(0.50)^{4}(1-0.50)^{(10-4)} \nonumber \\\\
&= 210(0.5)^{10}\nonumber \\\\
&= 0.205 \nonumber
\end{align}
$$

# References


{{< bibliography cited >}}

# Appendix A: Example of an Appendix With Equations   {#proof-pmf}

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





