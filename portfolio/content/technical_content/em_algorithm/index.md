---
title: "Expectation-Maximization Algorithm: A Method to Solve Latent Variable Problems" 
draft: false
summary: 'This is a summary of the post that brielfy explains the main points of the post to provide an anchor for the reader.' 
article_type: technical
output:
  bookdown::html_document2:
     keep_md: true
always_allow_html: true
bibFile: content/technical_content/em_algorithm/biblio.json    
tags: []
---   




  
  
```r 
1 + "a"
```

```
Error in 1 + "a": non-numeric argument to binary operator
```




<table id="myTable">
  <tr>
    <td><span data-text="1"></span></td>
    <td id="secondCol-Row-1" class="second-col"> <div class = 'highlight-wrapper'> <pre tabindex="0" class="chroma"><code><span class="o">+</span> <span class="s">"a"</span></code></pre><button class="copy-button"> <button class="copy-button"><span data-text="Copy"></span></button> </div></td>
    <td colspan="1"><button id="collapseButton"><span data-text="Hide"></span></button></td>
  </tr>
  <tr>
    <td><span data-text="2"></span></td>
    <td  id="secondCol-Row-2" class="second-col">Second Column - Row 2</td>
    <td></td>
  </tr>
  <tr>
    <td><span data-text="3"></span></td>
    <td id="secondCol-Row-3" class="second-col">Second Column - Row 3</td>
    <td></td>
  </tr>
</table>

This is some text. 


<style>
  
table[id = "myTable"]{
  width: 100%;
   border: none;
}

[data-text]::after {
  content: attr(data-text);
}

/*removes white space between columns */
table[id = "myTable"] td:nth-child(3){
  padding: 0; 
  vertical-align: top;
}


/*slightly increase spacing between borders and number */
table[id = "myTable"] td:nth-child(1){
  padding-right: 0.25rem;
  
}
  
 table td:first-child, table td:nth-child(3) {
    width: 7.5%;
    text-align: right;
    -webkit-user-select: none; /* Safari */
    -ms-user-select: none; /* IE 10 and IE 11 */
    user-select: none; /* Standard syntax */
  }
  
  td:nth-child(2) {
  position: relative; 
    background-color: rgba(116, 170, 219, 0.30);
    width: 85%;
    border-left: 2pt solid rgba(116, 170, 219, 0.50); 
    padding-top: 0; 
    padding-right: 0; 
  }
  
  /* Style for second column */
table td:nth-child(2)::selection {
    background: #b3d4fc;
    color: #000;
}




.copy-button{
  position: absolute; 
  font-weight: 500;
  height: 20px; 
  letter-spacing: 0.5px;
  right: 0%; 
  top: 0%; 
  font-size: 13px; 
  color: white;
  opacity: .95;
  background-color: var(--navbar-color);
  border: var(--navbar-color);
  border-top-left-radius: 0;
  border-top-right-radius: 0px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 4px;
  text-align: center; 
  cursor: pointer;
  float: right;
  display: block;
  width: calc(7.5/85*100%);
  padding-right: 0; 
  padding-top: 0;
  z-index: 1; 
  -webkit-user-select: none; /* Safari */
  -ms-user-select: none; /* IE 10 and IE 11 */
  user-select: none; /* Standard syntax */
}

button[id = "collapseButton"] {
  background-color:  rgba(57, 57, 57, 1);
  cursor: pointer;
  font-family: Helvetica;
  color: white;
  text-align: center;
  font-size: 13px;
  font-weight: 500;
  opacity: .95;
  letter-spacing: 0.5px;
   width: 100%;
  border: rgba(57, 57, 57, 1);
  vertical-align: top; 
  font-size: 13px;
  font-weight: 500;
  height: 20px; 
  border-top-left-radius: 0;
  border-top-right-radius: 0px;
  border-bottom-right-radius: 4px;
  border-bottom-left-radius: 0;
    -webkit-user-select: none; /* Safari */
  -ms-user-select: none; /* IE 10 and IE 11 */
  user-select: none; /* Standard syntax */
}

button[id = "collapseButton"]:hover{
  font-weight: bold;
   background: #343333;
}

</style>






That's some text with a footnote.[^1]$^{,}$ This is a change. More9
[^1]: And that's the footnote (see Table \ref{tab:parameterValues}).

```r {hl_lines=[2,3-5]}
#this is a comment  more comment my website m , . , y website my website my website 
#my website my website my website my website my website my website my website my 
#website my website my website my website my website
print('my website my website my, , , ,. `  website my website my website my website my website my website my website my website my website my website my website my website my website')  
print(1 + 2)
mean(x = c(1, 2))
print('another')
print(NULL)
print(NA)
print(TRUE)
"\n"
list('number_measurements' = c(5, 7, 9, 11),'spacing' = c('equal', 'time_inc'))
 
var <- function(x1, x2){
shiny::HTML("&nbsp;")
 
  if (x1 > 2) {print(x1)} 
  else {print (x2)}
}
```

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


Some explanatory text Some explanatory tex {{< cite "fine2019;george2000" >}} Some explanatory text Some explanatory textSome explanatory text{{< cite "fine2019;cole2003" >}}
Some explanatory text Some explanatory textSome explanatory  text{{< cite "fine2019;liu2022" >}}
[link](https://github.com/gohugoio/hugo/issues/9442)

If you liked how these _"generics"_ work in SystemVerilog and how the looks, check out the
```r {hl_lines=[2,3,4],language=python}
tup = ['Groucho', 'Marx', 'Xavier']
list_ex = list(["food", 538, True, 1.454])
sorted(tup)
 
list_ex2 = list([1 + 2, "a" * 5, 3])  
 
#deleting variables 
del(list_ex2)
list_ex.count(2)  
```


<pre><code class='python-code'>['Groucho', 'Marx', 'Xavier']
0
</code></pre>








The slope of the regression is 3.9324088. This is gamma $\gamma\$. $\frac{1}{2}$ This is (see Equation \ref{eq:multiline}; another comment) {{< inline-src python >}}print('NULL'){{< /inline-src >}} 

$$
\begin{alignat}{2}
I & = \int \rho R^{2} dV & + P \nonumber \\\\
Y & = 1 + x
\label{eq:multiline}
\end{alignat}
$$

### Why the EM algorithm is needed?{#section}

```r {hl_lines=[1],language=bash}
ls
 
cd ~/Desktop/Home/blog_posts
```

see [section](#section)

 Table \ref{tab:parameterValues}  Table \ref{tab:parameterValues}
Another paragraph begins and the spacing should not be too small from table above. {{< inline-src bash >}}cd ~/Desktop/Home/blog_posts{{< /inline-src >}} 

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



<div class= "figure"> <caption class = "figTop"> <span class="figLabel">Figure \ref{fig:gg-oz-plot}</span> <br> <span class = "figTitle"> <em>Title of Figure</em></span> </caption> <img src=index_files/figure-html/gg-oz-plot-1.png> <span class="figNote"><em>Note. </em>Life expectancy from 1952 - 2007 for Australia. Life expentancy increases steadily except from 1962 to 1969. We can safely say that our life expectancy is higher than it has ever been!</span> </div>   
 


Figure \ref{fig:gg-oz-plot} 


### References

{{< bibliography cited >}}






