---
title: "Example post" 
draft: false
summary: 'This is a summary of the post that briefly explains the main points of the post to provide an anchor for the reader.' 
article_type: technical
output:
  bookdown::html_document2: 
     keep_md: true
always_allow_html: true
bibFile: content/technical_content/example_post/biblio.json    
tags: []
---   





That's some text with a footnote.[^1]$^{,}$ 
[^1]: And that's the footnote (see Table \ref{tab:parameterValues}).

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


Some explanatory text Some explanatory tex {{< cite "fine2019;george2000" >}} Some explanatory text Some explanatory textSome explanatory text{{< cite "fine2019;cole2003" >}}
Some explanatory text Some explanatory textSome explanatory  text{{< cite "fine2019;liu2022" >}}
[link](https://github.com/gohugoio/hugo/issues/9442)

If you liked how these _"generics"_ work in SystemVerilog and how the looks, check out the


<pre><code class='python-code'>['Groucho', 'Marx', 'Xavier']
0
</code></pre>





The slope of the regression is 3.9324088. This is gamma $\gamma\$. $\frac{1}{2}$ This is (see Equation \ref{eq:multiline}; another comment) 

$$
\begin{alignat}{2}
I & = \int \rho R^{2} dV & + P \nonumber \\\\
Y & = 1 + x
\label{eq:multiline}
\end{alignat}
$$

# Tables

see [section](#section)

 Table \ref{tab:parameterValues}  Table \ref{tab:parameterValues}
Another paragraph begins and the spacing should not be too small from table above. 

 Table \ref{tab:parameterValues1}
 Table \ref{tab:parameterValues}
 Table \ref{tab:parameterValues1}
 





# Figures






# References

{{< bibliography cited >}}






