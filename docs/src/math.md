# math

---
editor:
    render-on-save: true
bibliography: references.bib
csl: ieee.csl
---

## Preliminaries on mathematical notation

Before jumping directly into the Bayesian inference pipeline, let us establish
the mathematical notation used throughout this paper. We define
(column) vectors as underlined lowercase symbols such as
```math
\underline{x} = \begin{bmatrix}
    x_1\\
    x_2\\
    \vdots\\
    x_N
\end{bmatrix}.
\tag{1}
```

In the same way, we define matrices as double-underline uppercase symbols such
as

```math
\underline{\underline{A}} =
\begin{bmatrix}
    A_{11} & A_{12} & \cdots & A_{1N}\\
    A_{21} & A_{22} & \cdots & A_{2N}\\
    \vdots & \vdots & \ddots & \vdots\\
    A_{M1} & A_{M2} & \cdots & A_{MN}\\
\end{bmatrix}.
\tag{2}
```math

## Fitness model

Empirically, the barcode relative frequency trajectories follow an exponential
function of the form

```math
f_{t+1}^{(b)} = f_{t}^{(b)} \mathrm{e}^{(s^{(b)} - \bar{s}_t)\tau},
\tag{3}
```

where $f_{t}^{(b)}$ is the frequency of barcode $b$ at the end of cycle number
$t$, $s^{(b)}$ is the relative fitness with respect to the reference
strain---the quantity we want to infer from the data---$\bar{s}_t$ is the mean
fitness of the culture at the end of cycle number $t$, and $\tau$ is the time
pass between cycle $t$ and $t+1$. We can rewrite Eq. 3 as

```math
\frac{1}{\tau}\ln \frac{f_{t+1}^{(b)}}{f_{t}^{(b)}} = (s^{(b)} - \bar{s}_t).
\tag{4}
```
Eq. 4 separates the measurements---the barcode frequencies---from the unobserved
(sometimes referred to as latent) parameters we want to infer from the
data---the population mean fitness and the barcode relative fitness. This is
ultimately the functional form used in our inference pipeline. Therefore, the
relative fitness is computed by knowing the log frequency ratio of each barcode
throughout the growth-dilution cycles.

The presence of the neutral lineages facilitates the determination of the
population mean fitness value $\bar{s}_t$. Since every relative fitness is
determined relative to the neutral lineage that dominates the culture, we define
their fitness to be $s^{(n)} = 0$, where the superscript $(n)$ specifies their
neutrality. This means that Eq. 4 for a neutral lineage takes the simpler form

```math
\frac{1}{\tau}\ln \frac{f_{t+1}^{(n)}}{f_{t}^{(n)}} = - \bar{s}_t.
\tag{5}
```

Therefore, we can use the data from these reference barcodes to directly infer
the value of the population mean fitness.

It is important to notice that the frequencies $f_{t}^{(b)}$ are not the allele
frequencies in the population (most of the culture is not sequenced since the
reference strain is not barcoded), but rather the relative frequencies in the
total number of sequencing reads. A way to conceptualize this subtle but
important point is to assume exponential growth in the *number of cells*
$N_t^{(b)}$ of the form

```math
N_{t+1}^{(b)} = N_{t}^{(b)} \mathrm{e}^{\lambda^{(b)}\tau},
\tag{6}
```

for every barcode $b$ with growth rate $\lambda^{(b)}$. However, when we
sequence barcodes, we do not directly measure the number of cells, but some
number of reads $r_t^{(b)}$ that map to barcode $b$. In the simplest possible
scenario, we assume

```math
r_{t}^{(b)} \propto N_{t}^{(b)},
\tag{7}
```

where, importantly, the proportionality constant depends on the total number of
reads for the library for cycle $t$, which might vary from library to library.
Therefore, to compare the number of reads between libraries at different time
points, we must normalize the number of reads to the same scale. The simplest
form is to define a relative abundance, i.e., a frequency with respect to the
total number of reads,

```math
f_{t}^{(b)} \equiv \frac{r_{t}^{(b)}}{\sum_{b'} r_{t}^{(b')}}.
\tag{8}
```

This is the frequency Eq. 3 describes.

Our ultimate objective is to infer the relative fitness $s^{(m)}$ for each of
the $M$ relevant barcodes in the experiment. To do so, we account for the three
primary sources of uncertainty in our model:

- Uncertainty in the determination of frequencies. Our model relates
frequencies between adjacent growth-dilution cycles to the fitness of the
corresponding strain. However, we do not directly measure frequencies. Instead,
our data for each barcode consists of a length $T$ vector of counts
$\underline{r}^{(b)}$ for each of the $T$ cycles in which the measurements were
taken.
- Uncertainty in the value of the population mean fitness. We define neutral
lineages to have fitness $s^{(n)} = 0$, helping us anchor the value of the
population mean fitness $\bar{s}_t$ for each pair of adjacent growth cycles.
Moreover, we take this parameter as an empirical parameter to be obtained from
the data, meaning that we do not impose a functional form that relates
$\bar{s}_t$ to $\bar{s}_{t+1}$. Thus, we must infer the $T-1$ values of this
population mean fitness with their uncertainty that must be propagated to the
value of the mutants' relative fitness.
- Uncertainty in each of the mutants' fitness values. 

To account for all these sources of uncertainty in a principled way, in the next
section, we develop a Bayesian inference pipeline.

## Bayesian inference 

Our ultimate objective is to infer the vector of relative fitness values

```math
\underline{s}^M = (s^{(1)}, s^{(2)}, \ldots, s^{(M)})^\dagger,
\tag{9}
```

where $^\dagger$ indicates the transpose. Our data consists of an $T \times B$
matrix $\underline{\underline{R}}$, where $B = M + N$ is the number of unique
barcodes given by the sum of the number of unique, relevant barcodes we care
about, $M$, and the number of unique neutral barcodes, $N$, and $T$ is the
number of growth cycles where measurements were taken. The data matrix is then
of the form

```math
\underline{\underline{R}} = \begin{bmatrix}
- & \underline{r}_1 & - \\
- & \underline{r}_2 & - \\
 & \vdots & \\
- & \underline{r}_T & - \\
\end{bmatrix},
\tag{10}
```

where each row $\underline{r}_t$ is a $B$-dimensional array containing the raw
barcode counts at cycle $t$. We can further split each vector $\underline{r}_t$
into two vectors of the form

```math
\underline{r}_t = \begin{bmatrix}
\underline{r}_t^{N} \\
\underline{r}_t^{M}
\end{bmatrix},
\tag{11}
```

i.e., the vector containing the neutral lineage barcode counts
$\underline{r}_t^{N}$ and the corresponding vector containing the mutant barcode
counts $\underline{r}_t^{M}$. Following the same logic, matrix
$\underline{\underline{R}}$ can be split into two matrices as

```math
\underline{\underline{R}} = \left[ 
\underline{\underline{R}}^N \; \underline{\underline{R}}^M
\right],
\tag{12}
```

where $\underline{\underline{R}}^N$ is a $T \times N$ matrix with the barcode
reads time series for each neutral lineage and $\underline{\underline{R}}^M$ is
the equivalent $T \times M$ matrix for the non-neutral lineages.

Our objective is to compute the joint probability distribution for all relative
fitness values given our data. We can express this joint posterior distribution
using Bayes theorem as

```math
\pi(\underline{s}^M \mid \underline{\underline{R}}) = \frac{
\pi(\underline{\underline{R}} \mid \underline{s}^M) 
\pi(\underline{s}^M)}
{\pi(\underline{\underline{R}})},
\tag{13}
```

where hereafter $\pi(\cdot)$ defines a probability density function, unless
otherwise stated. When defining our statistical model, we need not to focus on
the denominator on the right-hand side of @eq-bayes_obj. Thus, we can write

```math
\pi(\underline{s}^M \mid \underline{\underline{R}}) \propto
\pi(\underline{\underline{R}} \mid \underline{s}^M) 
\pi(\underline{s}^M).
\tag{14}
```

However, when implementing the model computationally, the normalization constant
on the right-hand side of @Eq. 13 must be computed. This can be done from
the definition of the model via an integral of the form

```math
\pi(\underline{\underline{R}}) = \int d^M \underline{s}^M
\pi(\underline{\underline{R}} \mid \underline{s}^M) 
\pi(\underline{s}^M),
\tag{15}
```

also known as a marginalization integral. Hereafter, differentials of the form
$d^n$ imply a $n$-dimensional integral.

Although @Eq. 13 and @Eq. 14 seem simple enough, recall that
@eq-fitness relates barcode frequency values and the population mean fitness to
the mutant relative fitness. Therefore, we must include these nuisance
parameters as part of our inference problem. To include these nuisance
parameters, let 

```math
\underline{\bar{s}}_T = (\bar{s}_1, \bar{s}_2, \ldots, \bar{s}_{T-1})^\dagger,
\tag{14}
```

be the vector containing the $T-1$ population mean fitness we compute from the
$T$ time points where measurements were taken. We have $T-1$ since the value
of any $\bar{s}_t$ requires cycle numbers $t$ and $t+1$. Furthermore, let the
matrix $\underline{\underline{F}}$ be a $T \times B$ matrix containing all
frequency values. As with @Eq. 12, we can split
$\underline{\underline{F}}$ into two matrices of the form

```math
\underline{\underline{F}} = \left[ 
\underline{\underline{F}}^N \; \underline{\underline{F}}^M
\right],
\tag{15}
```

to separate the corresponding neutral and non-neutral barcode frequencies. With
these nuisance variables in hand, the full inference problem we must solve takes
the form

```math
\pi(
    \underline{s}^M, \underline{\bar{s}}_T, \underline{\underline{F}} \mid
    \underline{\underline{R}}
) \propto
\pi(
    \underline{\underline{R}} \mid
    \underline{s}^M, \underline{\bar{s}}_T, \underline{\underline{F}}
)
\pi(
    \underline{s}^M, \underline{\bar{s}}_T, \underline{\underline{F}}
).
\tag{16}
```

To recover the marginal distribution over the non-neutral barcodes relative
fitness values, we can numerically integrate out all nuisance parameters, i.e.,

```math
\pi(\underline{s}^M \mid \underline{\underline{R}}) =
\int d^{T-1}\underline{\bar{s}}_T
\int d^{B}\underline{f}_1 \cdots
\int d^{B}\underline{f}_T
\;
\pi(
    \underline{s}^M, \underline{\bar{s}}_T, \underline{\underline{F}} \mid
    \underline{\underline{R}}
).
\tag{17}
```

### Factorizing the posterior distribution 

The left-hand side of @Eq. 16is extremely difficult to work with.
However, we can take advantage of the structure of our inference problem to
rewrite it in a more manageable form. Specifically, the statistical dependencies
of our observations and latent variables allow us to factorize the joint
distribution into the product of multiple conditional distributions. To gain
some intuition about this factorization, let us focus on the inference of the
population mean fitness values $\underline{\bar{s}}_T$. @eq-logfreq_neutral
relates the value of the population mean fitness to the neutral lineage
frequencies and nothing else. This suggests that when writing the posterior for
these population mean fitness parameters, we should be able to condition it only
on the neutral lineage frequency values, i.e., $\pi(\underline{\bar{s}}_T \mid
\underline{\underline{F}}^N)$. We point the reader to @sec-bayes_def for the
full mathematical details on this factorization. For our purpose here, it
suffices to say we can rewrite the joint probability distribution as a product
of conditional distributions of the form

```math
\pi(
    \underline{s}^M, \underline{\bar{s}}_T, \underline{\underline{F}} \mid
    \underline{\underline{R}}
) =
\pi(
    \underline{s}^M \mid \underline{\bar{s}}_T, \underline{\underline{F}}^M
)
\pi(
    \underline{\bar{s}}_T \mid \underline{\underline{F}}^N
)
\pi(\underline{\underline{F}} \mid \underline{\underline{R}}).
\tag{18}
```

Written in this form, @Eq. 18 captures the three sources of
uncertainty listed in @sec-fitness_model in each term. Starting from right to
left, the first term on the right-hand side of @Eq. 18 accounts for
the uncertainty when inferring the frequency values given the barcode reads. The
second term accounts for the uncertainty in the values of the mean population
fitness at different time points. The last term accounts for the uncertainty in
the parameter we care about---the mutants' relative fitnesses. We refer the
reader to @sec-bayes_def for an extended description of the model with specific
functional forms for each term on the left-hand side of @Eq. 18 as
well as the extension of the model to account for multiple experimental
replicates or hierarchical genotypes.