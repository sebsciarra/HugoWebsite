---
title: "The Theory, Meaning, and Applications of the Singular Value Decomposition" 
draft: true
summary: ""
date: "2024-11-22"
article_type: technical
output:
  bookdown::html_document2:
     keep_md: true
header-includes: 
  - \usepackage{blkarray}
always_allow_html: true
bibFile: content/technical_content/svd/refs.json    
imgThumbnail: ""
tags: []

---   





Three points require mentioning before beginning this whitepaper. First, I used Python and R code throughout this whitepaper and often imported objects created in Python into R for creating plots. To use Python and R coinjointly, I used the `reticulate` package made for R and created a conda environment to run Python code (see lines <a href="#2">2--13</a> below). 

```r 
library(reticulate)

#create and use conda environment
conda_create(envname = 'blog_posts',  python_version = '3.10.11')
use_condaenv(condaenv = 'blog_posts')

#install packages in conda environment
py_packages <- c('numpy', 'pandas', 'scikit-learn', "plotnine", "statsmodels", "manim", "factor_analyzer")
conda_install(envname = 'blog_posts', packages = py_packages, pip=T)

#useful for checking what packages are loaded
py_list_packages(envname = 'blog_posts', type = 'conda')
```

Second, the Python packages and modules in the Python code block are needed to run all the Python code in this paper (see lines <a href="#15">15--24</a> below). 




Third, although I often include code in papers so that readers can explore concepts, I decided to not include the Python code I used to create mathematical animations given the considerable length of the script. For readers interested in how I created my animations, the source code can be viewed in [this GitHub repository](#). 


# Introduction 


# Beginning With Fundamentals: A Necessary Review of Linear Algebra{#fundamentals}

## Basis Vectors: Standard and Non-Standard Bases

Basis vectors can be understood as the coordinates used to define some *n*-dimension space, $\mathbb{R}^n$. Beginning with the familiar standard basis vectors---that is, vectors with only one non-zero element equivalent to one---are often used to define *n*-dimensional spaces. As an example, consider a two-dimensional space, $\mathbb{R}^2$. To span $\mathbb{R}^2$ (i.e., define any vector in $\mathbb{R}^2$), the standard basis vectors of $\mathbf{b}\_{e\_x} = \[1, 0\]$ and $\mathbf{b}\_{e\_y}= \[0, 1\]$ can be used, and these vectors can be combined to form the basis matrix shown below in Equation \ref{eq:standardMatrix}: 


$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{B}_e = \begin{bmatrix} 
1 & 0 \\\\
0 & 1
\end{bmatrix}. 
\label{eq:standardMatrix}
\end{align}
\end{spreadlines}
$$
Note that each column of $\mathbf{B}_e$ represents a basis vector.

If $\mathbf{b}\_{e\_x}$ is multiplied by one and $\mathbf{b}\_{e\_y}$ is multiplied by two, the vector $\mathbf{g}_e = \[1, 2\]$ is obtained, and this is shown below in Animation \ref{anim:standardBasis}. 


{{< insert-video "media/videos/anim/480p15/standardBasis.mp4" "standardBasis" "Coordinates of Vector $\mathbf{g}_e$ in Standard Basis (Equation \ref{eq:standardMatrix})" "Using the standard basis vectors of $\mathbf{b}_{e_x} = [1, 0]$ and $\mathbf{b}_{e_y} = [0, 1]$, the pink vector ($\mathbf{g}$) has coordinates defined by $\mathbf{g}_e = [2, 1]$.">}}


Importantly, outside from the standard basis vectors being familiar and simple, there isn't always a strong justification for using them in applied settings. Thus, non-standard basis vectors are often used in applied settings. As an example of non-standard basis vectors, consider defining $\mathbb{R}^2$ with basis vectors $\mathbf{b}\_{n\_x} = \[0, 1\]$ and $\mathbf{b}\_{n\_y} = \[1, 2\]$, which are defined as a matrix below in Equation \ref{eq:nonStandardBasis}: 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{B}_n = \begin{bmatrix} 
1 & 0 \\\\
2 & 1
\end{bmatrix}. 
\label{eq:nonStandardBasis}
\end{align}
\end{spreadlines}
$$

To define $\mathbf{g}_e$ in the non-standard basis of Equation \ref{eq:nonStandardBasis}, the vector coordinates of $\mathbf{g}_n = \[1.5, 0.5\]$ are used, and this is shown below in Animation \ref{anim:nonStandardBasis}. (In later sections, I will show how to obtain coordinates in non-standard bases.)


{{< insert-video "media/videos/anim/480p15/nonStandardBasis.mp4" "nonStandardBasis" "Coordinates of Vector $\mathbf{g}$ in Non-Standard Basis (Equation \ref{eq:nonStandardBasis}) " "Using the non-standard basis vectors of $\mathbf{b}_{n_x} = [1, 2]$ and $\mathbf{b}_{n_y} = [0, 1]$, the pink vector ($\mathbf{g}$) has coordinates defined by $\mathbf{g}_n = [1.5, 0.5]$." >}}


## Translating Between Standard and Non-Standard Basis Vectors 

Although it may not look it, the non-standard basis vectors of Equation \ref{eq:nonStandardBasis} do indeed span $\mathbb{R}^2$; that is, they can define any vector in $\mathbb{R}^2$. In fact, because any set of *n* non-linearly dependent vectors will span any *n*-dimension space, an infinite number of *n* basis vector sets can be used to span any *n*-dimension space. 

With an infinite number of basis vector sets being able to define a space, it is likely that users may want to translate between basis vectors. Thus, in this section, I will show how to translate from standard to non-standard basis vectors (from the previous section) and vice-versa. As a recap, I have reprinted the standard (Equation \ref{eq:standardMatrix}) and non-standard (Equation \ref{eq:nonStandardBasis}) basis vectors below in their matrix form ($\mathbf{B}_e$ and $\mathbf{B}_n$, respectively):


$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{B}_e = \begin{bmatrix} 
1 & 0 \\\\
0 & 1
\end{bmatrix}. 
\tag{\ref{eq:standardMatrix} revisited}
\end{align}
\end{spreadlines}
$$

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{B}_n = \begin{bmatrix} 
1 & 1 \\\\
0 & 2
\end{bmatrix}. 
\tag{\ref{eq:nonStandardBasis} revisited}
\end{align}
\end{spreadlines}
$$

Beginning with a translation from a standard basis to a non-standard basis, recall the vector of $\mathbf{g}_e = \[2, 1\]$ defined in the standard basis. In the non-standard basis, $\mathbf{g}_e$ had coordinates of $\mathbf{g}_n = \[1.5, 0.5\]$. To redefine $\mathbf{g}_e$ in the non-standard basis of Equation \ref{eq:nonStandardBasis} and, thus, obtain $\mathbf{g}_n$, we first define $\mathbf{g}_e$ as some combination of scalar multiplications of each non-standard basis vector in $\mathbf{B}_n$ such that 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{B}_n \mathbf{g}_n = \mathbf{g}_e  \
\label{eq:nonBasisMult}
\end{align}
\end{spreadlines}
$$
Equation \ref{eq:nonBasisMult} can then be rearranged to solve for $\mathbf{g}_n$ to obtain 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{g}_n = \mathbf{B}_n^{-1} \mathbf{g}_e
\label{eq:basisNonBasisSol}
\end{align}.
\end{spreadlines}
$$
Therefore, pre-multiplying a vector defined in a standard basis (e.g., $\mathbf{g}_e$) by the inverse of a non-standard basis matrix (e.g., $\mathbf{B}_n$) returns the non-standard version of the matrix (e.g., $\mathbf{g}_n$}. 

Applying Equation \ref{eq:basisNonBasisSol} in the Python code block below, we see that the vector defined in the standard basis of $\mathbf{g}_e = \[1, 2\]$ becomes $\mathbf{g}_n = \[1.5, 0.5\]$ in the non-standard basis. 

```r {language=python}
# coordinates of g in standard basis (g_e)
h = np.array([2, 1]) 

# non-standard basis
B_n = np.array([[1, 1], 
                [0, 2]])
                
# coordinates of g in non-standard basis (g_n)
np.linalg.inv(B_n).dot(h)
```
<pre><code class='python-code'>array([1.5, 0.5])
</code></pre>

Lastly, and moving now to a translation from a non-standard to a standard basis, Equation \ref{eq:basisNonBasisSol} above can be modified to accomplish this translation. Recall the vector $\mathbf{g}_n$ has coordinates in the non-standard basis of $\[1.5, 0.5\]$. To obtain the coordinates of $\mathbf{g}_n$ in the standard basis, $\mathbf{g}_e$, we simply need to pre-multiply $\mathbf{g}_n$ by the non-standard basis matrix, as shown below in Equation \ref{eq:standardNonStandardSol}.

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{g}_e = \mathbf{B}_n \mathbf{g}_n 
\label{eq:standardNonStandardSol}  
\end{align}
\end{spreadlines}
$$

The solution presented in Equation \ref{eq:standardNonStandardSol} above for obtaining standard-basis coordinates of a vector defined in a non-standard basis has a strong intuition. To clearly explain this intuition, I will make the current example more personable by introducing two people: Tom and Sarah. Tom uses the standard basis vectors of Equation \ref{eq:standardMatrix} and Sarah uses the non-standard basis vectors of Equation \ref{eq:nonStandardBasis}. Although the non-standard coordinates in $\mathbf{g}_n = \[1.5, 0.5\]$ make little sense to Tom, he can easily understand Sarah's basis matrix, $\mathbf{B}_n$, because it is based on his basis vectors (i.e., the standard basis vectors). For example, Sarah's second non-standard basis vector of $\mathbf{b}\_{n\_x} = \[1, 2]$ can be obtained by multiplying Tom's first basis vector by one ($\mathbf{b}_e = \[1, 0\]$) and his second basis vector by 2 ($\mathbf{b}_e = \[0, 1\]$; Equation \ref{eq:standardMatrix}). Given that Sarah's definition of $\[1.5, 0.5\]$ for $\mathbf{g}_n$ simply means that her first and second basis vectors---which Tom understands---are respectively multiplied by 1.5 and 0.5, Tom can then simply pre-multiply $\mathbf{g}_n$ by Sarah's basis matrix, $\mathbf{B}_n$, to obtain the coordinates of $\mathbf{g}_n$ in his basis, $\mathbf{g}_e$. The Python code block below (lines ) shows that $\mathbf{g}_n = \[1.5, 0.5\]$ in Sarah's (non-standard) basis corresponds to coordinates of $\[2, 1\]$ in Tom's (standard) basis. 

```r {language=python}
#coordinates of g in Sarah's (non-standard) basis
g_n = np.array([1.5, 0.5]) 

# Sarah's basis vectors
B_n = np.array([[1, 1], 
                [0, 2]])
                
B_e = np.array([[1, 0], 
                [0, 1]])
                

# np.linalg.inv(B_e).dot(g_n)          
                
# Tom's coordinates for g
B_n.dot(g_n)
```
<pre><code class='python-code'>array([2., 1.])
</code></pre>



To recap, matrix multiplication was used to translate between standard and non-standard basis vectors. Given some vector defined in standard basis coordinates, pre-multiplying it by the inverse of a non-standard basis matrix returns the coordinates of this vector in the the non-standard basis (see Equation \ref{eq:basisNonBasisSol}). Going the othe way, pre-multiplying a a vector defined in a non-standard basis coordinates by the non-standard bases returns the vector in a standard basis (see Equation \ref{eq:standardNonStandardSol}). 


### Using Matrix Multiplication to Translate Between Non-Standard Bases 

In this section, I will show how to translate between non-standard bases. One non-standard basis that will be used has been used in previous sections, and I have reprinted it below in Equation \ref{eq:nonStandardBasis}.

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{B}_n = \begin{bmatrix} 
1 & 0 \\\\
2 & 1
\end{bmatrix}. 
\tag{\ref{eq:nonStandardBasis} revisited}
\end{align}
\end{spreadlines}
$$
The second non-standard basis matrix that I will use is shown below as $\mathbf{B}_m$ in Equation \ref{eq:nonStandardBasis2}: 


$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{B}_m = \begin{bmatrix} 
-1 & 0 \\\\
-0.5 & -1
\end{bmatrix}. 
\label{eq:nonStandardBasis2}
\end{align}
\end{spreadlines}
$$
In translating between two non-standard bases, I will show how to translate the vector of $\mathbf{g}_e = [2, 1]$ (defined in standard basis coordinates) from the non-standard basis in Equation \ref{eq:nonStandardBasis} to that of Equation \ref{eq:nonStandardBasis2}. 


```r {language=python}
g_e = np.array([2, 1])

b_n = np.array([[1, 0], 
                [2, 1]])




b_m = np.array([[-1, 0], 
                [-0.5, -1]])
                

np.linalg.inv(b_m).dot(b_n.dot(g_e))
```
<pre><code class='python-code'>array([-2., -4.])
</code></pre>


## Visualizing Matrix Multiplication as Rotation and/or Stretching of Basis Vectors

In the previous section, matrix multiplication was used to translate between standard and non-standard bases. In each translation of basis vectors, a strong intuition existed for the use of matrix multiplication. Perhaps fittingly, the strong intuition for using matrix multiplication to translate between basis vectors has a similarly intuitive geometry. 

Matrix multiplication transforms vectors because it rotates and/or stretches the bases that define the vectors. As a first example of the geometry of matrix multiplication, I show here how the matrices of the previous section pull the matrix space leftward and compressed it vertically. As a recap, I provide the basis vectors of the previous sections from Equations \ref{eq:standardMatrix} and \ref{eq:nonStandardBasis}.

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{B}_e = \begin{bmatrix} 
1 & 0 \\\\
0 & 1
\end{bmatrix}. 
\tag{\ref{eq:standardMatrix} revisited}
\end{align}
\end{spreadlines}
$$

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{B}_n = \begin{bmatrix} 
1 & 1 \\\\
0 & 2
\end{bmatrix}. 
\tag{\ref{eq:nonStandardBasis} revisited}
\end{align}
\end{spreadlines}
$$
To go from the standard to non-standard basis, Equation \ref{eq:basisNonBasisSol} is used, which results in the following computation: 

$$
\begin{spreadlines}{0.5em}
\begin{align}
&= \mathbf{B}_n^{-1} \mathbf{g}_e  \nonumber \\\\
&= \begin{bmatrix}
1 & -0.5 \\\\
0 & 0.5
\end{bmatrix}
\begin{bmatrix}
2 \\ 1
\end{bmatrix} \nonumber \\\\
&=\begin{bmatrix}
1.5 \\ 0.5
\end{bmatrix}.
\end{align}
\end{spreadlines}
$$

Animation \ref{anim:basisToNonBasis} below shows that the linear transformation applied by $\mathbf{B}^{-1}$ indeed results in the vector $\mathbf{g}\_{n}$ by pulling and compressing the standard basis space leftward. Specifically, the second basis vector of $\mathbf{b}\_{n_y} = \[0, 1\]$ is pulled leftward and compressed to become $\mathbf{b}^{-1}\_{n_x} = \[-0.5, 0.5\]$ (the first basis vector of $\mathbf{b}\_{n_x} = \[1, 0\]$ remains unchanged). In transforming the space defined by the standard basis vectors, the vector of $\mathbf{g}_e = \[2, 1\]$ becomes $\mathbf{g}_n = \[1.5, 0.5\]$. As discussed previously, the vector $\mathbf{g}_n$ indicates the scalar multiplications that must be applied to the basis vectors of $\mathbf{B}_n$ to obtain the vector $\mathbf{g}_e$. 

{{< insert-video "media/videos/anim/480p15/transformationExample.mp4" "basisToNonBasis" "Example of the Geometry of Matrix Multiplication from Equation \ref{eq:basisNonBasisSol})" "The example in this animation shows the geometry of pre-multiplying a vector, $\mathbf{g}_e = [2, 1]$, by $\mathbf{B}^{-1}_n$. In this case, the vector space is simply pulled leftward and compressed vertically. Specifically, the second basis vector of $\mathbf{b}_{n_y} = [0, 1]$ is pulled leftward and compressed to become $\mathbf{b}^{-1}_{n_x} = [-0.5, 0.5]$ (the first basis vector of $\mathbf{b}_{n_x} = [1, 0]$ remains the same). In transforming the space defined by the standard basis vectors, the vector of $\mathbf{g}_e = [2, 1]$ becomes $\mathbf{g}_n = [1.5, 0.5]$." >}}


With the example above showing how matrix multiplication transforms vectors by transforming the spaces that define them, I will show the geometry of the following matrices in the sections that follow. Below is a brief summary of the matrices whose geometries I will show and the transformations they apply:

1) Diagonal matrices stretch basis vectors
2) Orthonormal matrices rotate basis vectors
3) Rectangular matrices change dimension space
4) Inverse matrices un-rotate basis vectors


### Diagonal Matrices Stretch Basis Vectors{#diagonal}

Animation \ref{anim:diagonalMatrix} below shows that diagonal matrices simply stretch the basis vectors that define a space. In the current example, the matrix of 

$$
\begin{align}
\mathbf{D} = \begin{bmatrix}
2 & 0 \\\\
0 & 3
\end{bmatrix}
\end{align}
$$
stretches the first basis vector, $\mathbf{b}\_{e_x} = \[1, 0\]$, by a factor of two such that it becomes $\mathbf{b}\_{d_x} = \[2, 0\]$ and the second basis vector, $\mathbf{b}\_{e_y} = \[0, 1\]$, by a factor of three such that it becomes $\mathbf{b}\_{d_y} = \[0, 3\]$. As a result, the vector of $\mathbf{g}_e = \[2, 1\]$ becomes $\mathbf{g}_e = \[4, 3\]$. 


{{< insert-video "media/videos/anim/480p15/diagonalMatrix.mp4" "diagonalMatrix" "Geometry of Matrix Multiplication From Diagonal Matrix" "The example in this animation shows the geometry of pre-multiplying a vector, $\mathbf{g}_e = [2, 1]$, by the diagonal matrix $\mathbf{D}$. In this case, the space spanned by the standard basis vectors is stretched. Specifically, the first basis vector, $\mathbf{b}_{e_x} = [1, 0]$, is stretched by a factor of two such that it becomes $\mathbf{b}\_{d_x} = [2, 0]$ and the second basis vector,  $\mathbf{b}_{e_y} = [0, 1]$, is stretched by a factor of three such that it becomes $\mathbf{b}\_{d_y} = [0, 3]$. As a result, the vector of $\mathbf{g}_e = [2, 1]$ becomes $\mathbf{g}_e = [4, 3]$.">}}

### Orthonormal Matrices Only Rotate Basis Vectors (Length and Angle Preserving){#orthonormal}

Animation \ref{anim:orthonormalMatrix} below shows that orthonormal matrices simply rotate the basis vectors that define a space. In the current example, the matrix of 

$$
\begin{align}
\mathbf{Q} = \begin{bmatrix}
\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\\\
-\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}
\end{bmatrix}
\end{align}
$$
rotates the first basis vector, $\mathbf{b}\_{e_x} = \[1, 0\]$, clockwise to $\mathbf{b}\_{q_x} = \[\frac{\sqrt{2}}{2}, -\frac{\sqrt{2}}{2}\]$, and the second basis vector, $\mathbf{b}\_{e_y} = \[0, 1\]$, to $\mathbf{b}\_{q_y} = \[\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}\]$. As a result, the vector of $\mathbf{g}_e = \[2, 1\]$ becomes $\mathbf{g}_e = \[3\frac{\sqrt{2}}{2}, -\frac{\sqrt{2}}{2}\]$. 

 
{{< insert-video "media/videos/anim/480p15/orthonormalMatrix.mp4" "orthonormalMatrix" "Geometry of Orthonormal Matrix Multiplication" "The example in this animation shows the geometry of pre-multiplying a vector, $\mathbf{g}_e = [2, 1]$, by the orthonormal matrix $\mathbf{Q}$. In this case, the first basis vector, $\mathbf{b}_{e_x} = [1, 0]$, is rotated clockwise to $\mathbf{b}_{q_x} = [\frac{\sqrt{2}}{2}, -\frac{\sqrt{2}}{2}]$, and the second basis vector, $\mathbf{b}_{e_y} = [0, 1]$, is rotated clockwise to $\mathbf{b}_{q_y} = [\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}]$. As a result, the vector of $\mathbf{g}_e = [2, 1]$ becomes $\mathbf{g}_q = [3\frac{\sqrt{2}}{2}, -\frac{\sqrt{2}}{2}]$.">}}


Importantly, orthonormal matrices only rotate vector spaces. That is, there is no stretching/compressing of the vector space; vectors lengths remain unchanged and so do angles between vectors. In mathematical terms, orthonormal matrices preserve lengths and matrices and, given the ease with which each of these statements can be proven, I show each proof in turn. 

Beginning with the length-preserving property of orthonormal matrices, consider the formula for computing vector lengths in Equation \ref{eq:vectorLengths} below: 

$$
\begin{align}
\text{Length} &= \sum^{n}\_{i=1} v_i^2 = <\mathbf{v}, \mathbf{v}> = \lVert\mathbf{v} \rVert^2_2. 
\label{eq:vectorLengths}
\end{align}
$$
By applying Equation \ref{eq:vectorLengths} above to the length of a vector that is transformed by some orthonormal matrix, $\mathbf{Qv}$, it becomes clear that the vector's length remains unchanged. 

$$
\begin{spreadlines}{0.5em}
\begin{align}
 \lVert \mathbf{Qv} \rVert^2_2 &= (\mathbf{Qv})^\top \mathbf{Qv} \nonumber \\\\
 &= \mathbf{v}^\top\underbrace{\mathbf{Q}^\top\mathbf{Q}}\_{=\mathbf{I}}\\mathbf{v} \nonumber \\\\
 &= \mathbf{v}^\top\mathbf{v} \label{eq:lengthEquality} \qquad\qquad _\blacksquare 
\end{align}
\end{spreadlines}
$$

As an example of the length-preserving property of orthonormal matrices, the Python code block below (lines ...) shows that length of the original vector, $\mathbf{g}_e = \[2, 1\]$, and its orthonormal-transformed version, $\mathbf{g}_q = \[3\frac{\sqrt{2}}{2}, -\frac{\sqrt{2}}{2}\]$, remains unchanged.

```r {language=python}
# original vector
g_e = np.array([2, 1]) 

# transformed vector
g_q = np.array([3*(np.sqrt(2))/2, -np.sqrt(2)/2]) 
                
# lengths of each vector
print(g_e.dot(g_e), "\n",
      np.round(g_q.dot(g_q), 6))
```
<pre><code class='python-code'>5 
 5.0
</code></pre>


Ending with the angle-preserving feature of orthonormal matrices, consider the formula to computing the angles between two vectors, $\mathbf{v}$ and $\mathbf{w}$, shown below in Equation \ref{eq:anglesPreserve}:

$$
\begin{align}
\cos(\theta) &= \frac{\mathbf{v}^\top\mathbf{w}}{\lVert \mathbf{v} \rVert^2_2 \lVert \mathbf{w} \rVert^2_2} 
\label{eq:anglesPreserve}  \\\\
\end{align}
$$

By applying Equation \ref{eq:anglesPreserve} above to the angles between vectors following an orthonormal transformation, $\mathbf{Qv}$ and $\mathbf{Qw}$, it becomes clear that the angle between them remains unchanged (note how the proof shown in Equation \ref{eq:lengthEquality} above is used in the below proof). 

$$
\begin{align}
\cos(\theta) &= \frac{\mathbf{Qv}^\top\mathbf{Qw}}{\lVert \mathbf{Qv} \rVert^2_2 \lVert \mathbf{Qw} \rVert^2_2} \nonumber \\\\
&= \frac{\mathbf{v}^\top \mathbf{Q}^\top  \mathbf{Qw}}{(\mathbf{v}^\top \mathbf{Q}^\top  \mathbf{Qv}) (\mathbf{w}^\top \mathbf{Q}^\top  \mathbf{Qw})} \nonumber \\\\ 
\text{Note: } & \mathbf{Q}^\top\mathbf{Q} = \mathbf{I} \nonumber \\\\
&= \frac{\mathbf{v}^\top\mathbf{w}}{\lVert \mathbf{v} \rVert^2_2 \lVert \mathbf{w} \rVert^2_2} \qquad\qquad _\blacksquare  \\\\
\end{align}
$$

As an example of the angle-preserving property of orthonormal matrices, the Python code block below (lines ...) shows that the angle between the original standard basis vectors, $\mathbf{b}\_{e_x} = \[1, 0\]$ and $\mathbf{b}\_{e_y} = \[1, 0\]$, and their orthonormal-transformed versions, $\mathbf{Qb}\_{e_x}$ and $\mathbf{Qb}\_{e_y}$, remains unchanged. 

```r {language=python}
# original standard basis 
b_ex = np.array([1, 0])
b_ey = np.array([0, 1])

# orthonormal basis 
Q = np.array([[np.sqrt(2)/2, np.sqrt(2)/2], 
              [-np.sqrt(2)/2, np.sqrt(2)/2]])
                
# angle between original basis vectors
original_angle = np.arccos(b_ex.dot(b_ey)/(np.linalg.norm(b_ex) * np.linalg.norm(b_ey)))
transformed_angle =  np.arccos(b_ex.T.dot(Q.T).dot(Q).dot(b_ey)/(np.linalg.norm(Q.dot(b_ex)) * np.linalg.norm(Q.dot(b_ey))))

print(math.degrees(original_angle), "\n",
      math.degrees(transformed_angle))
```
<pre><code class='python-code'>90.0 
 90.0
</code></pre>


### Inverse Matrices Un-Transform Basis Vectors

Animation \ref{anim:inverseMatrix} shows that matrix inverses transform vector spaces in the opposite direction and magnitude of their non-inverted counterparts. In this way, matrix inverses can be conceptualized as un-transforming vector spaces. Within Animation \ref{anim:inverseMatrix}, the space is first transformed using the orthonormal matrix of 

$$
\begin{align}
\mathbf{Q} = \begin{bmatrix}
\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\\\
-\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}
\end{bmatrix}.
\end{align}
$$

The vector space is then un-transformed and brought back to the standard basis by applying the inverse of $\mathbf{Q}$. By applying $\mathbf{Q}^{-1}$, the basis vectors are rotated back to become the standard basis vectors, $\mathbf{b}\_{e_x}$ and $\mathbf{b}\_{e_y}$, and the transformed vector of $\mathbf{g}_n = \[\frac{3\sqrt{2}}{2}, -\frac{\sqrt{2}}{2}\]$ returns back to its original standard basis coordinates of $\mathbf{g}_e = \[2, 1\]$. 


{{< insert-video "media/videos/anim/480p15/inverseMatrix.mp4" "inverseMatrix" "Geometry of Inverse Matrix Multiplication" "The example in this animation shows the geometry of pre-multiplying a vector, $\mathbf{g}_e = [2, 1]$, by the orthonormal matrix $\mathbf{Q}$, and then pre-multiplying the transformed vector, $\mathbf{g}_n$, by $\mathbf{Q}^{-1}$. By applying $\mathbf{Q}^{-1}$, the basis vectors are rotated back to become the standard basis vectors, $\mathbf{b}_{e_x}$ and $\mathbf{b}_{e_y}$, and the transformed vector of $\mathbf{g}_n = [\frac{3\sqrt{2}}{2}, -\frac{\sqrt{2}}{2}]$ returns back to its original standard basis coordinates of $\mathbf{g}_e = [2, 1]$.">}}


### Rectangular Matrices Change Dimension Space {#rectangular}

So far, I have only shown linear transformations that result from applying square matrices. When multiplying spaces by square matrices, the input and output dimension spaces are the same. When multiplying by rectangular matrices, however, the dimensions of the input and output spaces are different; that is, rectangular matrices change the dimension space. As an example, Animation \ref{anim:rectangularMatrix} below shows the transformation that results from applying the rectangular matrix of 

$$
\begin{align}
\mathbf{B}_r = \begin{bmatrix}
1 & 2
\end{bmatrix}.
\end{align}
$$
The matrix $\mathbf{B}_r$ takes an input space of two dimensions and returns values in an output space of one dimension. Thus, two-dimensions vectors become scalars (i.e., one dimensional). In this case, the first basis vector, $\mathbf{b}\_{e_x} = \[1, 0\]$, is transformed to become a scalar (i.e., one-dimensional) value of $\textrm{b}\_{r_x}=1$, and, likewise, the second basis vector, $\mathbf{b}\_{e_y} = \[0, 1\]$, becomes $\textrm{b}\_{r_y}=2$. As a result, the vector of $\mathbf{g}_e = \[2, 1\]$ becomes $\textrm{g}_r = \[4\]$.

{{< insert-video "media/videos/anim/480p15/rectangularMatrix.mp4" "rectangularMatrix" "Geometry of Rectangular Matrix Multiplication" "The example in this animation shows the geometry of pre-multiplying a vector, $\mathbf{g}_e = [2, 1]$, by the rectangular matrix $\mathbf{B}_r$.  In this case, the first basis vector, $\mathbf{b}_{e_x} = [1, 0]$, is transformed to become a scalar (i.e., one-dimensional) value of $\textrm{b}_{r_x}=1$, and, likewise, the second basis vector, $\mathbf{b}_{e_y} = [0, 1]$, becomes $\textrm{b}_{r_y}=2$. As a result, the vector of $\mathbf{g}_e = [2, 1]$ becomes $\textrm{g}_r = [4]$.">}}

Importantly, because the input and output dimensions spaces are different, the input and output vector live in different dimension spaces, and so no input vector can lie in the span of any output vector (and vice-versa). 


### Eigenvectors (and Eigenvalues)

Due to the inherent connection between eigenvectors and singular value decomposition, it is necessary to provide an overview of the geometry of eigenvectors. That being said, although it is useful to understand the geometry of eigenvectors, their importance in singular value decomposition comes more from the meaning of these values (which will be explained in later sections) and less so from the geometry. Nonetheless, I briefly present the geometry of eigenvectors and eigenvalues.

Animation \ref{anim:eigenvectorMatrix} below shows that eigenvectors remain on their span following some linear transformation and eigenvalues represent the extent to which the eigenvectors are stretched. The geometry of eigenvectors and eigenvalues is summarized in Equation \ref{eq:eigenvector} below 

$$
\begin{align}
\mathbf{Av} = \lambda \mathbf{v}.
\label{eq:eigenvector}
\end{align}
$$

In words, if some eigenvector, $\mathbf{v}$, is pre-multiplied by a (square) matrix, $\mathbf{A}$, then the resulting vector is simply a scalar multiplication of $\mathbf{v}$, where $\lambda$ represents the scalar value. As an example from Animation \ref{anim:eigenvectorMatrix} shown below, when the eigenvector $\mathbf{v}_1 = \[\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}\]$ remains on its span following a pre-multiplication 

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\\\
0 & 2
\end{bmatrix}, 
$$
because the resulting vector, $\lambda\mathbf{v}_1$ is simply twice the value of $\mathbf{v}_1$. In Animation \ref{anim:eigenvectorMatrix}, the light blue lines represent the eigenbases and each eigenvector remains on its basis following a linear transformation by $\mathbf{A}$. 

{{< insert-video "media/videos/anim/480p15/eigenvectorMatrix.mp4" "eigenvectorMatrix" "Geometry of Eigenvectors and Eigenvalues" "Eigenvectors of the matrix $\mathbf{A}$ are shown to remain on their span (light blue lines) following a linear transformation applied by $\mathbf{A}$. The first eigenvector, $\mathbf{v}_1$, is multiplied by two, $\lambda_1 = 2$, and the second eigenvector, $\mathbf{v}_2$, is multiplied by a one, $\lambda_2= 1$, following as linear transformation of $\mathbf{A}$.">}}


One last important point to note is that only square matrices can have eigenvectors because non-square (or rectangular matrices) change the dimension space (see section on [rectangular matrices](#rectangular)). Because rectangular matrices change the dimension space, it is impossible for any vector to remain on its span.

# The Singular Value Decomposition and its Visualization
 
Having covered the necessary fundamentals, I will introduce the singular value decomposition and its geometry. In short, any matrix $\mathbf{A} \in \mathbb{R}^{n \times p}$ ($\mathbf{A}$ can also be complex) can be decomposed such that  

$$
\begin{align}
\mathbf{A} &= \mathbf{U}\Sigma \mathbf{V}^\top
\label{eq:svd}, 
\end{align}
$$
where each of the three matrices in the decomposition has the following characteristics: 

1. $\mathbf{U} \in \mathbb{R}^{(n \times n)}$: left singular vectors of $\mathbf{A}$, which are the eigenvectors of $\mathbf{A}\mathbf{A}^\top$. Because symmetric matrices have full sets of orthogonal eigenvectors (see [Appendix A3](#full-set)), $\mathbf{U}$ is orthonormal. 
2. $\mathbf{\Sigma} \in \mathbb{R}^{(n \times m)}$: rectangular matrix with *singular values* along its diagonal. Singular values equivalent to the square roots of the eigenvalues of $\mathbf{A}\mathbf{A}^\top$ (or equivalently of $\mathbf{A}^\top\mathbf{A}$).
3. $\mathbf{V} \in \mathbb{R}^{(m \times m)}$: right singular vectors of $\mathbf{A}$, which are the eigenvectors of $\mathbf{A}^\top\mathbf{A}$. Because symmetric matrices have full sets of orthogonal eigenvectors (see [Appendix A3](#full-set)), $\mathbf{V}$ is orthonormal. 

Applying what I presented previously in the [fundamentals of linear algebra](#fundamentals), the singular value decomposition implies that the linear transformation applied by any matrix can be broken down into three constituent transformations in the following order: 

1) Rotation: $\mathbf{V}^\top$ is an orthogonal (unnormalized version of an orthonormal matrix) matrix and so rotates basis vectors (see section on [orthonormal matrices](#orthonormal)). The astute reader will notice that the transpose of an orthogonal matrix is equivalent to its inverse, so $\mathbf{V}^\top$ is more technical an un-rotation of basis vectors. 
2) Stretching with possible dimension change: because $\mathbf{\Sigma}$ only has nonzero values along its diagonal, these values will stretch basis vectors (see section on [diagonal matrices](#diagonal)).  The dimension space can also change following a transformation by $\Sigma$. For example, if the number of rows, $n$, is less than the number of columns $p$, then the $p-n$ remaining columns of $\mathbf{\Sigma}$ will contain zeroes that will remove the $p-n$ remaining dimensions of $\mathbf{V}^\top$.
3) Rotation: $\mathbf{U}$ is an orthogonal (unnormalized version of an orthonormal matrix) matrix and thus rotates basis vectors (see section on [orthonormal matrices](#orthonormal)).


Animation \ref{anim:svdMatrix} below provides an geometric visualization of each transformation applied by each matrix of the singular value decomposition. First, it is shown that $\mathbf{A}$ transforms the standard basis vectors such that $\mathbf{b_x}$ becomes $[2, 0]$, $\mathbf{b_y}$ becomes $[1, 2]$, and $\mathbf{b_z}$ becomes $[1, 3]$. The second animation then shows that each standard basis vector lands on its coordinates after three transformations in the order of a rotation, stretching (with a dimension reduction), and a rotation. First, the (transposed) matrix of right singular vectors is orthonormal and so applies a rotation. Second, the matrix of singular values is off-diagonal and so applies a stretching of basis vectors along with a reduction in the dimension (from $\mathbb{R}^3$ to $\mathbb{R}^2$. Third, and last, the matrix of left singular vectors applies a rotation of the basis vectors.

{{< insert-video "media/videos/anim/480p15/svd_anim_proof.mp4" "svdMatrix" "Visualization of the Three Transformations That Constitute the Singular Value Decomposition" "An animation of each matrix of the singular valude decomposition of $\mathbf{A}$. First, it is shown that $\mathbf{A}$ transforms the standard basis vectors such that $\mathbf{b_x}$ becomes $[2, 0]$, $\mathbf{b_y}$ becomes $[1, 2]$, and $\mathbf{b_z}$ becomes $[1, 3]$. The second animation then shows that each standard basis vector lands on its coordinates after three transformations in the order of a rotation, stretching (with possible dimension change), and a rotation. First, the (transposed) matrix of right singular vectors is orthonormal and so applies a rotation. Second, the matrix of singular values is off-diagonal and so applies a stretching of basis vectors along with a reduction in the dimension (from $\mathbb{R}^3$ to $\mathbb{R}^2$. Third, and last, the matrix of left singular vectors applies a rotation of the basis vectors.">}}


# Proving the Singular Value Decomposition

With an geometric understanding of the singular value decomposition, I now provide a proof. Consider the symmetric matrix produced by $\mathbf{A}^\top\mathbf{A}$. Because is it symmetric, it has a full set of orthonormal eigenvectors (see [Appendix A3](#full-set)) and so can be re-expressed as 

$$
\begin{align}
\mathbf{A}^\top\mathbf{A} &= \mathbf{V \Lambda V}^\top. 
\label{eq:symRexp}
\end{align}
$$
Because $\Lambda$ is a diagonal matrix (of eigenvalues), it can be square-rooted to obtain

$$
\begin{align}
\mathbf{\Sigma} &= \mathbf{\Lambda}^{\frac{1}{2}}.
\label{eq:matrixRoot}
\end{align}
$$
Equation \ref{eq:matrixRoot} can then replace $\Lambda$ in Equation \ref{eq:symRexp} to obtain 

$$
\begin{align}
\mathbf{V \Lambda V}^\top &= \mathbf{V \mathbf{\Sigma}^\top \mathbf{\Sigma} V}^\top = \mathbf{A}^\top\mathbf{A}.
\end{align}
$$
Although it may not look it, the singular value decomposition is proven once we consider two truths. First, $\mathbf{V} \mathbf{\Sigma}^\top$  (or conversely $\mathbf{\Sigma} V}^\top$) and $ \mathbf{A}^\top$ (or conversely $\mathbf{A}$ are both positive semi-definite matrices (see [Appendix B](#pos-semi)). Second, because the matrix product of each positive semi-definite matrix is equivalent, the condition of unitary freedom is satisfied and so there must exist some orthonormal matrix, $\mathbf{U}$, that can be used to translate between the basis vectors of each matrix (see [Appendix C](#unitary)). Mathematically, 

$$
\begin{spreadlines}{0.5em}
\begin{align*}
\mathbf{A}^\top\mathbf{A} &= \mathbf{V \mathbf{\Sigma}^\top \mathbf{\Sigma} V}^\top \\\\
\text{Let } \mathbf{B} = \mathbf{\Sigma V}^\top \\\\
\mathbf{A}^\top\mathbf{A} &=  (\mathbf{\Sigma V}^\top)^\top (\mathbf{\Sigma V}^\top) \\\\
&=\mathbf{B}^\top \mathbf{B}  \\\\
\therefore \mathbf{A} &= \mathbf{UB} \\\\
&=\mathbf{U}\Sigma \mathbf{V}^\top \qquad\qquad _\blacksquare
\end{align*}
\end{spreadlines}
$$

# Understanding the Singular Value Decomposition 

In this section, I will provide a deeper explanation of the singular value decomposition by explaining the following three points in turn: 

1) Point 1: The number of nonzero singular values determines the number of eigenvectors that account for variance 
2) Point 2: The number of right and left singular vectors that account for variance is always equivalent 
3) Point 3: Left and right singular vectors represent unweighted loadings of people/variables onto principal axes 


## Point 1: The Number of Nonzero Singular Values Determines the Number of Eigenvectors That Account for Variance{#point-1}

To understand this point, it is first important to understand that an eigenvalue represents the amount of total variance accounted for by its corresponding eigenvector. Consider a mean-centered matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ with covariance matrix 

$$
\begin{align}
\mathbf{C} &= \frac{\mathbf{A}^\top\mathbf{A}}{n-1}
\end{align}
$$

and eigenvectors $\mathbf{v}_1, ..., \mathbf{v}_n$. Importantly, to compute the variance accounted for any given eigenvector, $\mathbf{v}_i$, the projections of the data onto the eigenvector are first needed. The projected values can be obtained using 

$$
\begin{align}
\mathbf{y} = \mathbf{Av}_i,
\end{align}
$$
with the variance of the projected values then being 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{C}_y &= \frac{\mathbf{y}^\top\mathbf{y}}{n-1} \nonumber \\\\
&= \frac{(\mathbf{Av}_i)^\top \mathbf{Av}_i}{n-1} \nonumber \\\\
&= \frac{\mathbf{v}_i^\top\mathbf{A}^\top \mathbf{Av_i}}{n-1} \nonumber \\\\
&= \mathbf{v}_i^\top \mathbf{C} \mathbf{v}_i 
\label{eq:eigInter}
\end{align}.
\end{spreadlines}
$$
Because $\mathbf{v}_i$ is an eigenvector of the covariance matrix, $\mathbf{C}$, the eigenvector equation (Equation \ref{eq:eigenvector}) can be leveraged to simplify Equatin \ref{eq:eigInter} above and prove that an eigenvalue represents that amount of total variance accounted for by an eigenvector. Note that, because $\mathbf{v}_i$ is a unit vector, $\mathbf{v}_i \mathbf{v}_i = 1$, 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\text{Let } \quad \mathbf{C}\mathbf{v}_i &= \lambda \mathbf{v}_i, \nonumber \\\\
\mathbf{C}_y &= \mathbf{v}_i^\top \lambda \mathbf{v}_i, \nonumber \\\\
&= \lambda \underbrace{\mathbf{v}_i^\top \mathbf{v}_i}\_{=1} \nonumber \\\\
&= \lambda \qquad\qquad _\blacksquare
\end{align}.
\end{spreadlines}
$$
Given that eigenvalues represent the amount of total variance accounted for by an eigenvector and that singular values are simply square roots of eigenvalues (see Equation \ref{eq:matrixRoot), the number of nonzero singular values, therefore, represents the number of eigenvectors that account for variance. 

## Point 2: The Number of Left and Right Singular Vectors Accounting for Variance is Always Equivalent

Proving that an equal number of left and right singular vectors that account for variance simply results from showing that these vectors have the same set of eigenvalues. Beginning first with the left singular vectors (which are the eigenvectors of $\mathbf{AA}^\top$), their eigenvalues are equivalent to the squared singular values of $\mathbf{A}$, $\mathbf{E} = \mathbf{\Sigma}^2 = \mathbf{\Sigma}\mathbf{\Sigma}$.

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{AA}^\top &= \mathbf{U \Sigma V}^\top (\mathbf{U \Sigma V}^\top)^\top \nonumber \\\\
&= \mathbf{U \Sigma V}^\top \mathbf{V \Sigma}^\top \mathbf{U}^\top \nonumber \\\\
&= \mathbf{U \Sigma} \mathbf{\Sigma}^\top \mathbf{U}^\top \nonumber \\\\
&= \mathbf{U} \mathbf{E} \mathbf{U}^\top
\end{align}
\end{spreadlines}
$$
Likewise, and ending with the right singular vectors (which are the eigenvectors of $\mathbf{X}^\top\mathbf{X}$), their eigenvalues are also equivalent to the squared singular values of $\mathbf{A}$. 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{A}^\top \mathbf{A} &=  (\mathbf{U \Sigma V}^\top)^\top\mathbf{U \Sigma V}^\top \nonumber \\\\
&= \mathbf{V}\mathbf{\Sigma U}^\top \mathbf{U \Sigma}^\top \mathbf{V}^\top \nonumber \\\\
&= \mathbf{V \Sigma} \mathbf{\Sigma}^\top \mathbf{V}^\top \nonumber \\\\
&= \mathbf{V} \mathbf{E} \mathbf{V}^\top
\end{align}
\end{spreadlines}
$$
Therefore, given that the left and right singular vectors (which are eigenvectors) have identical corresponding sets of eigenvalues, then, along with [Point 1](#point-1) that eigenvalues represent the amount of variance accounted for by eigenvectors, the same number of right and singular vectors account for variance.  $\qquad\qquad _\blacksquare$

## Point 3: Left and Right Singular Vectors Represent Unweighted Loadings of People/Variables Onto Principal Axes

Given that an equivalent number of left and right singular vectors account for variance, one logical question centers around whether there is a deeper meaning to the singular vectors. The short answer is yes. To understand the meaning of the left and right singular vectors, consider a data set of wine ratings (0--100 points), $\mathbf{A}$ (Equation \ref{eq:wineMatrix}), from seven novice drinkers for the following set of four wines: 1) cabernet sauvignon (CS), 2) merlot (M), 3) rosé (R), and 4) champagne (Ch).

$$
\begin{align}
    \mathbf{A} = 
    \begin{array}{c}
        \begin{array}{cccc}
            \text{CS} & \text{M} & \text{Ch} & \text{R} \\\\
        \end{array} \\\\
        \left[
        \begin{array}{cccc}
            73 & 76 & 58 & 61 \\\\
            73 & 71 & 55 & 55 \\\\
            84 & 88 & 71 & 71 \\\\
            80 & 80 & 69 & 73 \\\\
            47 & 49 & 40 & 39 \\\\
            54 & 49 & 69 & 69 \\\\
            46 & 46 & 63 & 66 \\\\
            63 & 61 & 89 & 89 \\\\
            70 & 67 & 90 & 90 \\\\
            59 & 58 & 79 & 76 \\\\
        \end{array}
        \right]
    \end{array}
\end{align}
$$

Within the matrix of wine ratings, I created each person's scores to reflect one of three types of wine drinker: 

1) *Dinner Wine Drinker (Din)*: prefers drinking red wines such as cabernet sauvignon (CS) and merlot (M)
2) *Celebratory Wine Drinker (Cel)*: prefers drinking champagne (Chp) and rosé (R)

In the paragraphs that follow, I will show that the left singular and right singular vectors respectively contain the unweighted loadings of people and variables onto these underlying wine drinker types. (Note that the meanings of the left and right singular vectors swap places if the matrix in question has each *p* person's data in a column and each *n* variable's data in a row.)

To understand the left and right singular vectors, consider first the set of singular values for $\mathbf{A}$. The Python code block below (lines <a href="#1">1--</a>) computes the eigenvalues (or squared singular values) of $\mathbf{A}$ and the percentage of total variance accounted for by each eigenvector. Given that over 99% of the total variance can be recovered from using two eigenvectors, it can be argued that there are only two eigenvectors worth considering. In other words, 99% of the total variance in the data can be retained by projecting the original scores onto only two of the four eigenvectors.







As an aside, here we can see that the eigenvalues are indeed the squared singular values (dividing by a scaling constant). 

```r {language=python}
print("Eigenvalues:", np.round(eig_values, decimals=3))
print("Squared singular values:", np.round((S**2)/6, decimals=3))
```


Given that three of the five eigenvectors can be used to obtain a high-fidelity representation of the data, let's now consider the reduced versions of the left and right singular vectors. Beginning with the left singular vector, I show it below in its original form  




# Applications of the Singular Value Decomposition 


## Principal Components Analysis 

## Data Reduction 

## Recommendations 

# Conclusion 


# References

{{< bibliography cited >}}

# Appendix A: Proof of Spectral Theorem {#spectral}

The spectral theorem has three propositions that I prove in turn: 

1) Symmetric matrices must have at least one eigenvalue (and, therefore, at least one eigenvector).
2) For any symmetric matrix $\mathbf{S}$, all of its eigenvalues are real. 
3) For any symmetric matrix $\mathbf{S}$, an orthonormal set of eigenvectors exists that spans the space of $\mathbf{S}$. In other words, a full set of orthogonal eigenvectors exists. 


## Appendix A1: Intuition For Why Symmetric Matrices Must Have at Least One Eigenvalue and Eigenvector{#one-eig}

To understand why all symmetric matrices have at least one eigenvalue, it is important to first understand how eigenvalues are obtained. To derive an eigenvalue, the equation of 

$$
\begin{align}
\mathbf{Av} = \lambda \mathbf{v} 
\tag{\ref{eq:eigenvector} revisited}
\end{align}
$$

can be re-expressed below in Equation \ref{eq:eigenvalue}:

$$
\begin{align}
\mathbf{Av} - (\lambda \mathbf{I})\mathbf{v} = 0 \nonumber \\\\
(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0 
\label{eq:eigenvalue}
\end{align}
$$
At this point, a brief discussion of determinants is apropos. Geometrically, determinants represent the extent to which a matrix changes the space spanned by a set of basis vectors changes (for an excellent video, see [3blue1brown](https://www.youtube.com/watch?v=Ip3X9LOh2dk)). Importantly, if a matrix has a zero-value determinant, it applies a transformation that converts the space spanned by a set of basis vectors to zero. In the current case where only one vector is under consideration, $\mathbf{v}$, if the  matrix of $(\mathbf{A} - \lambda \mathbf{I})$ has a determinant of zero, then $\mathbf{v}$ will go to zero after pre-multiplication (Equation \ref{eq:eigenvalue}). Therefore, to solve Equation \ref{eq:eigenvalue} above, we need to find the eigenvalue, $\lambda$, that, when subtracted from the diagonal of $\mathbf{A}$, causes $\mathbf{A}$ to have a zero-value determinant. 

Fortunately, the fundamental theorem of algebra guarantees that any square matrix will have at least one eigenvalue (real or complex) if some constant $\lambda$ is subtracted from the diagonal. Although outside the context of this paper,  a great explanation of this proof can be found at [fundamental theorem of linear algebra](https://www.youtube.com/watch?v=shEk8sz1oOw); note that this video provides an explanation for there being at least one complex eigenvalue). 

To provide a basic intuition for why square matrices must have at least one eigenvalue (whether complex or real), it is important to realize that the determinant equation is simply a polynomial one and that polynomial equations always have at least one root (whether complex or real). For example, consider the 2x2 matrix 

$$
\mathbf{A} = \begin{bmatrix}
5 - \lambda & 4 \\\\
3 & 1 - \lambda. 
\end{bmatrix}
$$
The determinant of $\mathbf{A}$ is then 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\det|A| &= (5-\lambda)(1-\lambda) - 4(3) \\\\
&= \lambda^2 - 6\lambda -7 
\end{align}
\end{spreadlines}
$$
and the eigenvalues are 

$$
\begin{spreadlines}{0.5em}
\begin{align}
0 &= (\lambda -7)(\lambda + 1) \\\\
\lambda &=7, -1
\end{align}
\end{spreadlines}
$$
Although the determinant of a 2x2 matrix is a rather simple case, it generalizes to larger matrices and more complex polynomial equations. Because the existence of at least eigenvalue is guaranteed for square matrices, then there will be at least one eigenvector that accompanies the eigenvalue.

## Appendix A2: Proof That All Eigenvalues of Symmetric Matrices are Real 

In the previous section, an argument was provided for why all square matrices must have at least one eigenvalue (and consequently at least one eigenvector). In this section, I will provide a proof that symmetric matrices only have real eigenvalues. As some necessary background, it is first important to have a basic understanding of complex numbers and complex conjugates. A complex number, $z$, can be conceptualized as the sum of a real-number, $a$, and imaginary component, $bi$, as shown below in Equation \ref{eq:complexNum}:

$$ 
\begin{align}
z = a + bi.
\label{eq:complexNum}
\end{align}
$$

Importantly, complex numbers have corresponding entities called *complex conjugates*, $\bar{z}$, that have real components of equivalent magnitude and imaginary components of equivalent magnitude in the opposite direction. Given the complex number defined above in Equation \ref{eq:complexNum}, its complex conjugate would be defined as 

$$
\begin{align}
\bar{z} = a - bi.
\label{eq:complexConjugate}
\end{align}
$$

In circumstances where a complex number is equivalent to its conjugate, the imaginary component does not exist, and so the complex and its conjugate are equivalent to the real number $a$ (see derivation below). 

$$
\begin{spreadlines}{0.5em}
\begin{align*}
\text{If } z &= \bar{z}, \\\\
\text{then } a + bi &= a - bi \\\\ 
bi &= -bi \\\\
i &= 0 \\\\
z &= \bar{z} = a. 
\end{align*}
\end{spreadlines}
$$
With a basic understanding of complex numbers, I will now prove that the eigenvalues of any given symmetric matrix, $\mathbf{A}$, are real. That is, for any set of eigenvectors $\mathbf{v}$, the complex and complex conjugate eigenvalues (i.e., $\lambda$ and $\bar{\lambda}$, respectively) are equivalent. To prove that the eigenvalues are real, I will derive complex and complex conjugate forms of an eigenvalue-based equation. Before doing do, it is important to first obtain the complex conjugate of the eigenvector equation, shown below in Equation \ref{eq:complexConjEig}:

$$
\begin{align}
\mathbf{A\bar{v}} &= \bar{\lambda} \bar{\mathbf{v}}. 
\label{eq:complexConjEig}
\end{align}
$$
Note that the matrix $\mathbf{A}$ has no complex conjugate because it consists of only real numbers. Using Equation \ref{eq:complexConjEig}, we can now derive complex and complex conjugate forms of an eigenvalue-based equation that are equivalent. As a reminder, $\mathbf{A}=\mathbf{A}^\top$ because  $\mathbf{A}$ is symmetric. 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{Av} &= \lambda \mathbf{v} \quad \text{Post-multiply by} \quad \bar{\mathbf{v}} \nonumber \\\\
(\mathbf{Av})^\top\bar{\mathbf{v}} &=  (\lambda \mathbf{v})^\top \bar{\mathbf{v}} \nonumber \\\\ 
 &= \lambda\mathbf{v}^\top\bar{\mathbf{v}} \label{eq:eigComplex} \\\\ 
(\mathbf{Av})^\top\bar{\mathbf{v}} &=  \mathbf{v}^\top\mathbf{A}^\top\bar{\mathbf{v}} \nonumber \\\\
&=  \mathbf{v}^\top\mathbf{A}\bar{\mathbf{v}} \quad \text{Substitute} \quad \mathbf{A}\bar{\mathbf{v}}=\bar{\lambda} \bar{\mathbf{v}} (\text{Equation } \ref{eq:complexConjEig}) \nonumber \\\\
&= \mathbf{v}^\top\bar{\lambda} \bar{\mathbf{v}} \nonumber \\\\
&= \bar{\lambda}\mathbf{v}^\top\mathbf{v} 
\label{eq:eigComplexConj} \qquad\qquad _\blacksquare
\end{align}
\end{spreadlines}
$$
Given that  the complex and complex conjugate equations of Equations \ref{eq:eigComplex} and \ref{eq:eigComplexConj}, respectively, are equivalent (note that they are both equivalent to $(\mathbf{Av})^\top\bar{\mathbf{v}}$), any eigenvalue contained within $\lambda$ must then be real.  

## Appendix A3: Proof That Symmetric Matrices Have Full Sets of Orthonormal Eigenvectors{#full-set}

At this point, I have proven that symmetric matrices have real eigenvalues and that there must be at least one eigenvalue with a corresponding eigenvector. With this information, it can now be proven that symmetric matrices have full sets of orthonormal eigenvectors. That is, when applying the transformation of a symmetric matrix, there exists a set of orthonormal eigenvectors that span the entire space of the symmetric matrix. 

To prove that symmetric matrices have full sets of orthogonal eigenvectors, I show that any symmetric matrix can be diagonalized by an orthonormal matrix. This proof applies the Gram-Schmidt procedure to a symmetric matrix to generate a set of orthonormal vectors that is then shown to diagonalize the symmetric matrix. Mathematically then, and central to the spectral theorem, any symmetric matrix $\mathbf{A}$ can be transformed into a diagonal matrix, $\Lambda$, using a matrix of eigenvectors, $\mathbf{V}$, such that 

$$
\begin{align}
\Lambda & = \mathbf{V}^{-1}\mathbf{AV}, 
\label{eq:eigenMatrix}
\end{align}
$$

where $\Lambda$ is a diagonal matrix of eigenvalues. Importantly, because $\mathbf{V}$ is obtained with the Grahm-Schmidt process, the eigenvectors are orthonormal. 

To prove Equation \ref{eq:eigenMatrix} for any symmetric matrix, I provide a two-step proof below. First, I apply the Gram-Schmidt procedure to some symmetric matrix to obtain a full set of orthonormal vectors and show that the first basis vector of $\mathbf{A}$ can be diagonalized. Second, I show that the remaining columns of $\mathbf{V}$ are also eigenvectors by showing they can be diagonalized. 

Beginning with the first step of the proof, a symmetric matrix, $\mathbf{A}$, can be orthonormalized by applying the Gram-Schmidt procedure to obtain a matrix $\mathbf{P}$ (for an explanation of the Gram-Schmidt procedure, see this [video](https://www.youtube.com/watch?v=rHonltF77zI)) such that each column of $\mathbf{P}$ is an orthonormal vector. Importantly, $\mathbf{P}$ can be applied to $\mathbf{A}$ to obtain 

$$
\begin{align*}
\mathbf{B} & = \mathbf{P}^{\mathrm{T}}\mathbf{AP}, 
\end{align*}
$$
which is symmetric because 

$$
\begin{spreadlines}{0.5em}
\begin{align*}
\mathbf{B} &= \mathbf{B} ^{\mathrm{T}} \\\\
&= (\mathbf{P}^{\mathrm{T}}\mathbf{AP})^{\mathrm{T}} \\\\
&= (\mathbf{PA}^{\mathrm{T}}) \mathbf{P} \\\\ 
&=  \mathbf{P}^{\mathrm{T}}\mathbf{AP}
\end{align*}
\end{spreadlines}
$$
Additionally, and important for the first step of this proof, the first column of $\mathbf{A}$ can be diagonalized because the first column of $\mathbf{P}$ is assumed to be an eigenvector, of which there must be at least one for a symmetric matrix (for an explanation, see [Appendix A1](#one-eig)). Mathematically then, the first column of $\mathbf{B}$ follows a diagonal form (i.e., it only has one nonzero value) such that 

$$
\begin{spreadlines}{0.5em}
\begin{align*}
&= \mathbf{Be}_1 \\\\
&=  \mathbf{P}^{\mathrm{T}}\mathbf{A} \underbrace{\mathbf{Pe}_1}\_{\mathbf{v}_1} \\\\
&= \mathbf{P}^{\mathrm{T}}\underbrace{\mathbf{Av}}\_{\lambda\mathbf{v}} \\\\
&= \lambda \mathbf{P}^{\mathrm{T}}\mathbf{v} \\\\
&= \lambda \begin{bmatrix} 1 \\\\ 0 \\\\ 0   \end{bmatrix}.
\end{align*}
\end{spreadlines}
$$
Because $\mathbf{B}$ is symmetric, the first row then has the same structure as the first column, and so $\mathbf{B}$ follows a diagonal structure in the first row and column (see Equation \ref{eq:orthonormalInitial} below) such that

$$
\begin{spreadlines}{0.5em}
\begin{align*}
\mathbf{B} &= \begin{bmatrix}  
\lambda & 0 & \cdots & 0 \\\\
0 & \mathbf{C}  \\\\
\vdots \\\\
0
\end{bmatrix}
\label{eq:orthonormalInitial}
\end{align*}
\end{spreadlines}
$$
where $\mathbf{C} \in \mathbb{R}^{(n-1) \times (n-1)}$ and represents the remainder of $\mathbf{B}$.

Having shown that the first column of a symmetric matrix an be diagonalized, I now show in this second part of the proof that the remaining columns of a symmetric matrix can be diagonalized. In other words, there exists some orthonormal matrix $\mathbf{Q}$ such that 

$$
\begin{align}
\mathbf{Q}^{-1}\mathbf{CQ} = \mathbf{D}, 
\label{eq:fullEigen}
\end{align}
$$
where $\mathbf{D}$ is a diagonal matrix. To prove Equation \ref{eq:fullEigen} above, I first define 

$$
\mathbf{V} = \mathbf{P}\begin{bmatrix}1 & 0 \nonumber \\\\
0  &\mathbf{Q} \end{bmatrix}.
$$
As with $\mathbf{B}$, $\mathbf{V}$ is also equal to its inverse because it is orthogonal (inherited from $\mathbf{Q}$).

$$
\begin{spreadlines}{0.5em}
\begin{align*}
\mathbf{V} &= \mathbf{V}^{-1}  \nonumber \\\\
&= \begin{bmatrix}  
1 & 0 & \cdots & 0 \\\\
0 & \mathbf{Q}^{-1}  \\\\
\vdots \\\\
0
\end{bmatrix}\mathbf{P}^{-1}  \nonumber \\\\
&= \begin{bmatrix}  
1 & 0 & \cdots & 0 \\\\
0 & \mathbf{Q}^{\mathrm{T}}  \\\\
\vdots \\\\
0
\end{bmatrix}\mathbf{P}^{\mathrm{T}}   \nonumber
\end{align*}
\end{spreadlines}
$$

Now, the central component of the spectral theorem can be proven by showing that $\mathbf{V}^{-1}\mathbf{A}\mathbf{V}$ yields a diagonal matrix. 

$$
\begin{spreadlines}{0.5em}
\begin{align*}
\mathbf{V}^{-1}\mathbf{A}\mathbf{V} &= \begin{bmatrix}  
1 & 0 & \cdots & 0 \\\\
0 & \mathbf{Q}^{-1}  \\\\
\vdots \\\\
0
\end{bmatrix} \mathbf{P}^{-1} \mathbf{A} \mathbf{P} \begin{bmatrix}  
1 & 0 & \cdots & 0 \\\\
0 & \mathbf{Q}  \\\\
\vdots \\\\
0
\end{bmatrix} \\\\
&= \begin{bmatrix}  
1 & 0 & \cdots & 0 \\\\
0 & \mathbf{Q}^{-1}  \\\\
\vdots \\\\
0
\end{bmatrix} \mathbf{B} \begin{bmatrix}  
1 & 0 & \cdots & 0 \\\\
0 & \mathbf{Q}  \\\\
\vdots \\\\
0
\end{bmatrix}  \nonumber \\\\ 
&= \begin{bmatrix}  
1 & 0 & \cdots & 0 \\\\
0 & \mathbf{Q}^{-1}  \\\\
\vdots \\\\
0
\end{bmatrix} \begin{bmatrix}  
\lambda & 0 & \cdots & 0 \\\\
0 & \mathbf{C}  \\\\
\vdots \\\\
0
\end{bmatrix} \begin{bmatrix}  
1 & 0 & \cdots & 0 \\\\
0 & \mathbf{Q}  \\\\
\vdots \\\\
0
\end{bmatrix}  \nonumber \\\\ 
&= \begin{bmatrix}  
\lambda  & 0 & \cdots & 0 \\\\
0 & \mathbf{Q}^{-1}\mathbf{CQ}  \\\\
\vdots \\\\
0
\end{bmatrix} \\\\
&= \begin{bmatrix}  
\lambda & 0 & \cdots & 0 \\\\
0 & \mathbf{D}  \\\\
\vdots \\\\
0
\end{bmatrix}  \nonumber \qquad\qquad _\blacksquare 
\end{align*}
\end{spreadlines}
$$
Therefore, symmetric matrices can be fully diagonalized because they must have full sets of eigenvectors that are orthonormal. 


# Appendix B: $AA^\top$ is a Positive Semi-Definite Matrix {#pos-semi}

A positive semi-definite matrix, $\mathbf{P}$, is one whose product with any nonzero vector, $\mathbf{v}$, is greater than or equal to zero such that 

$$
\begin{align}
\mathbf{v}^\top \mathbf{Pv} \ge 0
\label{eq:psd}
\end{align}.
$$
If we replace $\mathbf{P}$ with the symmetric matrix of $\mathbf{A}^\top\mathbf{A}$, the condition above in Equation \ref{eq:psd} is met because the computation becomes a dot product or the 2-norm of identical vectors. In simpler terms, 

$$ 
\begin{align*}
\mathbf{v}^\top \mathbf{A}^\top\mathbf{Av} \ge 0
\end{align*}
$$
because $\mathbf{v}^\top\mathbf{A}^\top$ and $\mathbf{Av}$ result in identical vectors. Therefore, 

$$ 
\begin{align}
\mathbf{v}^\top \mathbf{A}^\top\mathbf{Av} \ge 0 \nonumber \\\\
\lVert \mathbf{Av}\rVert^2 \ge 0 \qquad\qquad _\blacksquare
\end{align}
$$

# Appendix C: Unitary Freedom of Positive Semi-Definite Matrices{#unitary}

In this section, I will first state the unitary freedom of positive semi-definite matrices and will then provide an intuitive explanation. In formal terms, this property of positive semi-definite matrices states that, given
$\mathbf{B}, \mathbf{C} \in \mathbb{R}^{m \times n}$ and 

$$
\begin{align}
\mathbf{B}^\top\mathbf{B} = \mathbf{C}^\top\mathbf{C}, 
\label{eq:unitaryFreedom}
\end{align}
$$

then there must exist some orthogonal matrix $\mathbf{Q} \in \mathbb{R}^{m \times n}$ that allows us to go from one positive semi-definite matrix to another. Mathematically then, we can define 

$$
\begin{align}
\mathbf{B} = \mathbf{QC}
\end{align}
$$

$$
\begin{align}
\mathbf{B}^\top\mathbf{B} &= \mathbf{QC}^\top\mathbf{QC} \nonumber \\\\
&= \mathbf{C}^\top \mathbf{Q}^\top \mathbf{QC} \nonumber \\\\
&=\mathbf{C}^\top\mathbf{C} \qquad\qquad _\blacksquare
\end{align} 
$$

Although the math above is indeed true, it does not provide an intuitive understanding for why there must exist an orthogonal matrix that allows us to translate between the two matrices. Therefore, I now provide an intuitive explanation for this proof. 

As discussed in the section on [orthonormal matrices](#orthonormal), these matrices simple rotate basis vectors and so the vector lengths and angles between them remain unchanged. Therefore, in looking at Equation \ref{eq:unitaryFreedom}, the basis vectors of $\mathbf{B}$ and $\mathbf{C}$ must have identical lengths and angles. 

Beginning with the lengths of $\mathbf{B}$ and $\mathbf{C}$, the lengths are identical because the 2-norm of each $i$ basis vector in either matrix produces the outcome value (e.g., $\mathbf{A}_i$). Mathematically, 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{B}^\top\mathbf{B} = \mathbf{C}^\top\mathbf{C} = \mathbf{A} \\\\
\begin{bmatrix}
\rule[.5ex]{5ex}{0.5pt} & \mathbf{b}_1 & \rule[.5ex]{5ex}{0.5pt} \\\\
\rule[.5ex]{5ex}{0.5pt} & \mathbf{b}_2 & \rule[.5ex]{5ex}{0.5pt} \\\\
& \vdots  \\\\
\rule[.5ex]{5ex}{0.5pt} & \mathbf{b}_n & \rule[.5ex]{5ex}{0.5pt}
\end{bmatrix}
\begin{bmatrix}
\rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} &  \rule[-1ex]{0.5pt}{2.5ex} \\\\
\mathbf{b}_1 & \mathbf{b}_2 & \cdots &  \mathbf{b}_n \\\\
\rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex}
\end{bmatrix} 
& = \mathbf{A}
\nonumber \\\\ 
\begin{bmatrix}
\rule[.5ex]{5ex}{0.5pt} & \mathbf{c}_1 & \rule[.5ex]{5ex}{0.5pt} \\\\
\rule[.5ex]{5ex}{0.5pt} & \mathbf{c}_2 & \rule[.5ex]{5ex}{0.5pt} \\\\
& \vdots \\\\
\rule[.5ex]{5ex}{0.5pt} & \mathbf{c}_n & \rule[.5ex]{5ex}{0.5pt}
\end{bmatrix}
\begin{bmatrix}
\rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} \\\\
\mathbf{c}_1 & \mathbf{c}_2 & \cdots &  \mathbf{c}_n \\\\
\rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex}
\end{bmatrix} 
& = \mathbf{A}
\nonumber \\\\ 
\lVert \mathbf{b_i} \rVert^2_2 = \lVert \mathbf{c_i} \rVert^2_2 &= \mathbf{A}_i
\label{eq:identicalLengths}
\end{align}. 
\end{spreadlines}
$$
Ending with the relative angles between the basis vectors, consider any two basis vectors in the matrices of $\mathbf{B}$ and $\mathbf{C}$, $\mathbf{b}_i,\mathbf{b}_j$ and $\mathbf{c}_i,\mathbf{c}_j$. If the two matrices satisfy the unitary freedom condition (Equation \ref{eq:unitaryFreedom}), then the dot product between any corresponding sets of two basis vectors will be equivalent. Mathematically, 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\mathbf{B}^\top\mathbf{B} &= \mathbf{C}^\top\mathbf{C} = \mathbf{A} \\\\
\mathbf{b}_i \mathbf{b}_j &= \mathbf{c}_i \mathbf{c}_j = \mathbf{a}\_{ij}. 
\label{eq:equalDotProducts}
\end{align}
\end{spreadlines}
$$

In looking at the formula for the angle between two vectors below

$$
\begin{spreadlines}{0.5em}
\begin{align}
\cos(\theta) &= \frac{\mathbf{v}^\top\mathbf{w}}{\lVert \mathbf{v} \rVert^2_2 \lVert \mathbf{w} \rVert^2_2},
\tag{\ref{eq:anglesPreserve} revisited}
\end{align}
\end{spreadlines}
$$

and applying the equality of vectors lengths (Equation \ref{eq:identicalLengths}) and dot products (Equation \ref{eq:equalDotProducts}), it becomes evident that the angle between any sets of two vectors, $\mathbf{b}_i,\mathbf{b}_j$ and $\mathbf{c}_i,\mathbf{c}_j$, must be identical be. Mathematically, 

$$
\begin{spreadlines}{0.5em}
\begin{align}
\cos(\theta) = \frac{\mathbf{b}_i^\top\mathbf{b}_j}{\lVert \mathbf{b}_i \rVert^2_2 \lVert \mathbf{b}_j \rVert^2_2} &= \frac{\mathbf{c}_i^\top\mathbf{c}_j}{\lVert \mathbf{c}_i \rVert^2_2 \lVert \mathbf{c}_j \rVert^2_2} \nonumber \\\\
=\frac{\mathbf{a}\_{ij}}{\lVert \mathbf{b}_i \rVert^2_2 \lVert \mathbf{b}_j \rVert^2_2} &= \frac{\mathbf{a}\_{ij}}{\lVert \mathbf{c}_i \rVert^2_2 \lVert \mathbf{c}_j \rVert^2_2} \nonumber
\end{align}.
\end{spreadlines}
$$

To summarize, if two positive semi-definite matrices satisfy the condition of unitary freedom (Equation \ref{eq:unitaryFreedom}), then the basis vectors of each matrix will have identical lengths and the angles between any two sets of corresponding basis vectors will have identical angles. This equivalence of angles and lengths means that there must exist some orthonormal matrix that can be used to translate between the two positive semi-definite matrices. $\qquad\qquad _\blacksquare$

<script>
    pseudocode.renderClass("pseudocode");
</script>
