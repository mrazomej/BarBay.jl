# math

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

where ``f_{t}^{(b)}`` is the frequency of barcode ``b`` at the end of cycle
number ``t``, ``s^{(b)}`` is the relative fitness with respect to the reference
strain---the quantity we want to infer from the data---``\bar{s}_t`` is the mean
fitness of the culture at the end of cycle number ``t``, and ``\tau`` is the
time pass between cycle ``t`` and ``t+1``. We can rewrite Eq. 3 as

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
population mean fitness value ``\bar{s}_t``. Since every relative fitness is
determined relative to the neutral lineage that dominates the culture, we define
their fitness to be ``s^{(n)} = 0``, where the superscript ``(n)`` specifies
their neutrality. This means that Eq. 4 for a neutral lineage takes the simpler
form

```math
\frac{1}{\tau}\ln \frac{f_{t+1}^{(n)}}{f_{t}^{(n)}} = - \bar{s}_t.
\tag{5}
```

Therefore, we can use the data from these reference barcodes to directly infer
the value of the population mean fitness.

It is important to notice that the frequencies ``f_{t}^{(b)}`` are not the
allele frequencies in the population (most of the culture is not sequenced since
the reference strain is not barcoded), but rather the relative frequencies in
the total number of sequencing reads. A way to conceptualize this subtle but
important point is to assume exponential growth in the *number of cells*
``N_t^{(b)}`` of the form

```math
N_{t+1}^{(b)} = N_{t}^{(b)} \mathrm{e}^{\lambda^{(b)}\tau},
\tag{6}
```

for every barcode ``b`` with growth rate ``\lambda^{(b)}``. However, when we
sequence barcodes, we do not directly measure the number of cells, but some
number of reads ``r_t^{(b)}`` that map to barcode ``b``. In the simplest
possible scenario, we assume

```math
r_{t}^{(b)} \propto N_{t}^{(b)},
\tag{7}
```

where, importantly, the proportionality constant depends on the total number of
reads for the library for cycle ``t``, which might vary from library to library.
Therefore, to compare the number of reads between libraries at different time
points, we must normalize the number of reads to the same scale. The simplest
form is to define a relative abundance, i.e., a frequency with respect to the
total number of reads,

```math
f_{t}^{(b)} \equiv \frac{r_{t}^{(b)}}{\sum_{b'} r_{t}^{(b')}}.
\tag{8}
```

This is the frequency Eq. 3 describes.

Our ultimate objective is to infer the relative fitness ``s^{(m)}`` for each of
the ``M`` relevant barcodes in the experiment. To do so, we account for the
three primary sources of uncertainty in our model:

- Uncertainty in the determination of frequencies. Our model relates frequencies
between adjacent growth-dilution cycles to the fitness of the corresponding
strain. However, we do not directly measure frequencies. Instead, our data for
each barcode consists of a length ``T`` vector of counts ``\underline{r}^{(b)}``
for each of the ``T`` cycles in which the measurements were taken.
- Uncertainty in the value of the population mean fitness. We define neutral
lineages to have fitness ``s^{(n)} = 0``, helping us anchor the value of the
population mean fitness ``\bar{s}_t`` for each pair of adjacent growth cycles.
Moreover, we take this parameter as an empirical parameter to be obtained from
the data, meaning that we do not impose a functional form that relates
``\bar{s}_t`` to ``\bar{s}_{t+1}``. Thus, we must infer the ``T-1`` values of
this population mean fitness with their uncertainty that must be propagated to
the value of the mutants' relative fitness.
- Uncertainty in each of the mutants' fitness values. 

To account for all these sources of uncertainty in a principled way, in the next
section, we develop a Bayesian inference pipeline.

## Bayesian inference 

Our ultimate objective is to infer the vector of relative fitness values

```math
\underline{s}^M = (s^{(1)}, s^{(2)}, \ldots, s^{(M)})^\dagger,
\tag{9}
```

where ``^\dagger`` indicates the transpose. Our data consists of an ``T \times
B`` matrix ``\underline{\underline{R}}``, where ``B = M + N`` is the number of
unique barcodes given by the sum of the number of unique, relevant barcodes we
care about, ``M``, and the number of unique neutral barcodes, ``N``, and ``T``
is the number of growth cycles where measurements were taken. The data matrix is
then of the form

```math
\underline{\underline{R}} = \begin{bmatrix}
- & \underline{r}_1 & - \\
- & \underline{r}_2 & - \\
 & \vdots & \\
- & \underline{r}_T & - \\
\end{bmatrix},
\tag{10}
```

where each row ``\underline{r}_t`` is a ``B``-dimensional array containing the
raw barcode counts at cycle ``t``. We can further split each vector
``\underline{r}_t`` into two vectors of the form

```math
\underline{r}_t = \begin{bmatrix}
\underline{r}_t^{N} \\
\underline{r}_t^{M}
\end{bmatrix},
\tag{11}
```

i.e., the vector containing the neutral lineage barcode counts
``\underline{r}_t^{N}`` and the corresponding vector containing the mutant
barcode counts ``\underline{r}_t^{M}``. Following the same logic, matrix
``\underline{\underline{R}}`` can be split into two matrices as

```math
\underline{\underline{R}} = \left[ 
\underline{\underline{R}}^N \; \underline{\underline{R}}^M
\right],
\tag{12}
```

where ``\underline{\underline{R}}^N`` is a ``T \times N`` matrix with the
barcode reads time series for each neutral lineage and
``\underline{\underline{R}}^M`` is the equivalent ``T \times M`` matrix for the
non-neutral lineages.

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

where hereafter ``\pi(\cdot)`` defines a probability density function, unless
otherwise stated. When defining our statistical model, we need not to focus on
the denominator on the right-hand side of Eq. 13. Thus, we can write

```math
\pi(\underline{s}^M \mid \underline{\underline{R}}) \propto
\pi(\underline{\underline{R}} \mid \underline{s}^M) 
\pi(\underline{s}^M).
\tag{14}
```

However, when implementing the model computationally, the normalization constant
on the right-hand side of Eq. 13 must be computed. This can be done from the
definition of the model via an integral of the form

```math
\pi(\underline{\underline{R}}) = \int d^M \underline{s}^M
\pi(\underline{\underline{R}} \mid \underline{s}^M) 
\pi(\underline{s}^M),
\tag{15}
```

also known as a marginalization integral. Hereafter, differentials of the form
``d^n`` imply a ``n``-dimensional integral.

Although Eq. 13 and Eq. 14 seem simple enough, recall that Eq. 3 relates
barcode frequency values and the population mean fitness to the mutant relative
fitness. Therefore, we must include these nuisance parameters as part of our
inference problem. To include these nuisance parameters, let 

```math
\underline{\bar{s}}_T = (\bar{s}_1, \bar{s}_2, \ldots, \bar{s}_{T-1})^\dagger,
\tag{14}
```

be the vector containing the ``T-1`` population mean fitness we compute from the
``T`` time points where measurements were taken. We have ``T-1`` since the value
of any ``\bar{s}_t`` requires cycle numbers ``t`` and ``t+1``. Furthermore, let
the matrix ``\underline{\underline{F}}`` be a ``T \times B`` matrix containing
all frequency values. As with Eq. 12, we can split
``\underline{\underline{F}}`` into two matrices of the form

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

The left-hand side of Eq. 16is extremely difficult to work with. However, we
can take advantage of the structure of our inference problem to rewrite it in a
more manageable form. Specifically, the statistical dependencies of our
observations and latent variables allow us to factorize the joint distribution
into the product of multiple conditional distributions. To gain some intuition
about this factorization, let us focus on the inference of the population mean
fitness values ``\underline{\bar{s}}_T``. Eq. 4 relates the value of the
population mean fitness to the neutral lineage frequencies and nothing else.
This suggests that when writing the posterior for these population mean fitness
parameters, we should be able to condition it only on the neutral lineage
frequency values, i.e., ``\pi(\underline{\bar{s}}_T \mid
\underline{\underline{F}}^N)``. We point the reader to sec-bayes_def for the
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

Written in this form, Eq. 18 captures the three sources of uncertainty listed
in sec-fitness_model in each term. Starting from right to left, the first term
on the right-hand side of Eq. 18 accounts for the uncertainty when inferring
the frequency values given the barcode reads. The second term accounts for the
uncertainty in the values of the mean population fitness at different time
points. The last term accounts for the uncertainty in the parameter we care
about---the mutants' relative fitnesses.

In the next sections we will explicitly develop each of the terms in Eq. 18.

### Frequency uncertainty ``\pi(\underline{\underline{F}} \mid \underline{\underline{R}})``

We begin with the probability of the frequency values given the raw barcode
reads. The first assumption is that the inference of the frequency values for
time ``t`` is independent of any other time. Therefore, we can write the joint
probability distribution as a product of independent distributions of the form

```math
\pi(\underline{\underline{F}} \mid \underline{\underline{R}}) =
\prod_{t=1}^T \pi(\underline{f}_t \mid \underline{r}_t),
\tag{19}
```

where ``\underline{f}_t`` and ``\underline{r}_t`` are the ``t``-th row of the
matrix containing all of the measurements for time ``t``. We imagine that when
the barcode reads are obtained via sequencing, the quantified number of reads is
a Poisson sample from the "true" underlying number of barcodes within the pool.
This translates to assuming that the number of reads for each barcode at any
time point ``r^{(b)}_t`` is an independent Poisson random variable, i.e.,

```math
r^{(b)}_t \sim \operatorname{Poiss}(\lambda^{(b)}_t),
\tag{20}
```

where the symbol "``\sim``" is read "distributed as." Furthermore, for a Poisson
distribution, we have that

```math
\lambda^{(b)}_t = \left\langle r^{(b)}_t \right\rangle = 
\left\langle 
    \left( r^{(b)}_t - \left\langle r^{(b)}_t \right\rangle \right)^2
\right\rangle,
\tag{21}
```

where ``\left\langle \cdot \right\rangle`` is the expected value. In other words
the Poisson parameter is equal to the mean and variance of the distribution. The
Poisson distribution has the convenient property that for two Poisson
distributed random variables ``X \sim \operatorname{Poiss}(\lambda_x)`` and ``Y
\sim \operatorname{Poiss}(\lambda_y)``, we have that

```math
Z \equiv X + Y \sim \operatorname{Poiss}(\lambda_x + \lambda_y).
\tag{22}
```

This additivity allows us to write the total number of reads at time ``t`` ``n_t``
also as a Poisson-distributed random variable of the form

```math
n_t \sim \operatorname{Poiss}\left( \sum_{b=1}^B \lambda^{(b)}_t \right),
\tag{23}
```

where the sum is taken over all ``B`` barcodes.

If the total number of reads is given by Eq. 23, the array with the number of
reads for each barcode at time ``t``, ``\underline{r}_t`` is then distributed as

```math
\underline{r}_t \sim \operatorname{Multinomial}(n_t, \underline{f}_t),
\tag{24}
```

where each of the ``B`` entries of the frequency vector ``\underline{f}_t`` is a
function of the ``\underline{\lambda}_t`` vector, given by

```math
f_t^{(b)} \equiv f_t^{(b)}(\underline{\lambda}_t) = 
\frac{\lambda_t^{(b)}}{\sum_{b'=1}^B \lambda_t^{(b')}}.
\tag{25}
```

In other words, we can think of the ``B`` barcode counts as independent Poisson
samples or as a single multinomial draw with a random number of total draws,
``n_t``, and the frequency vector ``\underline{f}_t`` we are interested in.
Notice that Eq. 25 is a deterministic function that connects the Poisson
parameters to the frequencies. Therefore, we have the equivalence that

```math
\pi(\underline{f}_t \mid \underline{r}_t) = 
\pi(\underline{\lambda}_t \mid \underline{r}_t),
\tag{26}
```

meaning that the uncertainty comes from the ``\underline{\lambda}_t`` vector. By
Bayes theorem, we therefore write

```math
\pi(\underline{\lambda}_t \mid n_t, \underline{r}_t) \propto
\pi(n_t, \underline{r}_t \mid \underline{\lambda}_t) \pi(\underline{\lambda}_t),
\tag{27}
```

where we explicitly include the dependence on ``n_t``. This does not affect the
distribution or brings more uncertainty because ``\underline{r}_t`` already
contains all the information to compute ``n_t`` since

```math
n_t = \sum_{b=1}^B r_t^{(b)}.
\tag{28}
```

But adding the variable allows us to factorize Eq. 27 as

```math
\pi(\underline{\lambda}_t \mid n_t, \underline{r}_t) \propto
\pi(\underline{r}_t \mid n_t, \underline{\lambda}_t)
\pi(n_t \mid \underline{\lambda}_t)
\pi(\underline{\lambda}_t)
\tag{29}
```

We then have

```math
\underline{r}_t \mid n_t, \underline{\lambda}_t \sim
\operatorname{Multinomial}(n_t, \underline{f}_t(\underline{\lambda}_t)).
\tag{30}
```

Furthermore, we have

```math
n_t \mid \underline{\lambda}_t \sim 
\operatorname{Poiss}\left(\sum_{b=1}^B \lambda_t^{(b)}\right).
\tag{31}
```

Finally, for our prior ``\pi(\underline{\lambda}_t)``, we first assume each 
parameter is independent, i.e.,

```math
\pi(\underline{\lambda}_t) = \prod_{b=1}^B \pi(\lambda_t^{(b)}).
\tag{32}
```

A reasonable prior for each ``\lambda_t^{(b)}`` representing the expected number
of reads for barcode ``b`` should span several orders of magnitude. Furthermore,
we assume that no barcode in the dataset ever goes extinct. Thus, no frequency
can equal zero, facilitating the computation of the log frequency ratios needed
to infer the relative fitness. The log-normal distribution satisfies these
constraints; therefore, for the prior, we assume

```math
\lambda_t^{(b)} \sim 
\log\mathcal{N}(\mu_{\lambda_t^{(b)}}, \sigma_{\lambda_t^{(b)}}),
\tag{33}
```

with ``\mu_{\lambda_t^{(b)}}, \sigma_{\lambda_t^{(b)}}`` as the user-defined 
parameters that characterize the prior distribution.

#### Summary

Putting all the pieces developed in this section together gives a term for our
inference of the form

```math
\pi(\underline{\underline{F}} \mid \underline{\underline{R}}) \propto
\prod_{t=1}^T\left\{
    \pi(\underline{r}_t \mid n_t, \underline{\lambda}_t)
    \pi(n_t \mid \underline{\lambda}_t)
    \left[ 
        \prod_{b=1}^B \pi(\lambda_t^{(b)})
    \right]
\right\}
\tag{34}
```

where

```math
\underline{r}_t \mid n_t, \underline{\lambda}_t \sim
\operatorname{Multinomial}(n_t, \underline{f}_t(\underline{\lambda}_t)),
\tag{35}
```

```math
n_t \mid \underline{\lambda}_t \sim 
\operatorname{Poiss}\left(\sum_{b=1}^B \lambda_t^{(b)}\right).
\tag{36}
```

and

```math
\lambda_t^{(b)} \sim 
\log\mathcal{N}(\mu_{\lambda_t^{(b)}}, \sigma_{\lambda_t^{(b)}}),
\tag{37}
```

### Population mean fitness uncertainty ``\pi(\underline{\bar{s}}_T \mid \underline{\underline{F}}, \underline{\underline{R}})``

Next, we turn our attention to the problem of determining the population mean
fitnesses ``\underline{\bar{s}}_T``. First, we notice that our fitness model in
eq-fitness does not include the value of the raw reads. They enter the
calculation indirectly through the inference of the frequency values we
developed in sec-bayes_freq. This means that we can remove the conditioning of
the value of ``\underline{\bar{s}}_T`` on the number of reads, obtaining a
simpler probability function

```math
\pi(
    \underline{\bar{s}}_T \mid 
    \underline{\underline{F}}, \underline{\underline{R}}
) = 
\pi(
    \underline{\bar{s}}_T \mid 
    \underline{\underline{F}}
).
\tag{38}
```

Moreover, our fitness model does not directly explain how the population mean
fitness evolves over time. In other words, our model cannot explicitly compute
the population mean fitness at time ``t+1`` from the information we have about
time ``t``. Given this model limitation, we are led to assume that we must infer
each ``\bar{s}_t`` independently. Expressing this for our inference results in

```math
\pi(
    \underline{\bar{s}}_T \mid 
    \underline{\underline{F}}
) =
\prod_{t=1}^{T-1} \pi(\bar{s}_t \mid \underline{f}_t, \underline{f}_{t+1}),
\tag{39}
```

where we split our matrix ``\underline{\underline{F}}`` for each time point and
only kept the conditioning on the relevant frequencies needed to compute the
mean fitness at time ``t``.

Although our fitness model in eq-fitness also includes the relative fitness
``s^{(m)}``, to infer the population mean fitness we only utilize data from the
neutral lineages that, by definition, have a relative fitness ``s^{(n)} = 0``.
Therefore, the conditioning on Eq. 39can be further simplified by only keeping
the frequencies of the neutral lineages, i.e.,

```math
\pi(\bar{s}_t \mid \underline{f}_t, \underline{f}_{t+1}) =
\pi(\bar{s}_t \mid \underline{f}_t^N, \underline{f}_{t+1}^N).
\tag{40}
```

Earlier, we emphasized that the frequencies ``f_t^{(n)}`` do not represent the
true frequency of a particular lineage in the population but rather a
"normalized number of cells." Therefore, it is safe to assume each of the ``N``
neutral lineages' frequencies is changing independently. The correlation of how
increasing the frequency of one lineage will decrease the frequency of others is
already captured in the model presented in sec-bayes_freq. Thus, we write


```math
\pi(\bar{s}_t \mid \underline{f}_t^N, \underline{f}_{t+1}^N) =
\prod_{n=1}^N \pi(\bar{s}_t \mid f_t^{(n)}, f_{t+1}^{(n)}).
\tag{41}
```

Now, we can focus on one of the terms on the right-hand side of Eq. 41. Writing
Bayes theorem results in

```math
\pi(\bar{s}_t \mid f_t^{(n)}, f_{t+1}^{(n)}) \propto
\pi(f_t^{(n)}, f_{t+1}^{(n)} \mid \bar{s}_t) \pi(\bar{s}_t).
\tag{42}
```

Notice the likelihood defines the joint distribution of neutral barcode
frequencies conditioned on the population mean fitness. However, rewriting our
fitness model in eq-fitness for a neutral lineage to leave frequencies on one
side and fitness on the other results in

```math
\frac{f_{t+1}^{(n)}}{f_t^{(n)}} = \mathrm{e}^{- \bar{s}_t\tau}.
\tag{43}
```

Eq. 43 implies that our fitness model only relates **the ratio** of frequencies
and not the individual values. To get around this complication, we define

```math
\gamma_t^{(b)} \equiv \frac{f_{t+1}^{(b)}}{f_t^{(b)}},
\tag{44}
```

as the ratio of frequencies between two adjacent time points for any barcode
``b``. This allows us to rewrite the joint distribution ``\pi(f_t^{(n)},
f_{t+1}^{(n)} \mid \bar{s}_t)`` as

```math
\pi(f_t^{(n)}, f_{t+1}^{(n)} \mid \bar{s}_t) =
\pi(f_t^{(n)}, \gamma_{t}^{(n)} \mid \bar{s}_t).
\tag{45}
```

Let us rephrase this subtle but necessary change of variables since it is a key
part of the inference problem: our series of independence assumptions lead us to
Eq. 42that relates the value of the population mean fitness ``\bar{s}_t`` to
the frequency of a neutral barcode at times ``t`` and ``t+1``. However, as shown
in Eq. 43, our model functionally relates the ratio of frequencies---that we
defined as ``\gamma_t^{(n)}``---and not the independent frequencies to the mean
fitness. Therefore, instead of writing for the likelihood the joint distribution
of the frequency values at times ``t`` and ``t+1`` conditioned on the mean
fitness, we write the joint distribution of the barcode frequency at time ``t``
and the ratio of the frequencies. These **must be** equivalent joint
distributions since there is a one-to-one mapping between ``\gamma_t^{(n)}`` and
``f_{t+1}^{(n)}`` for a given value of ``f_t^{(n)}``. Another way to phrase this
is to say that knowing the frequency at time ``t`` and at time ``t+1`` provides
the same amount of information as knowing the frequency at time ``t`` and the
ratio of the frequencies. This is because if we want to obtain ``f_{t+1}^{(n)}``
given this information, we simply compute

```math
f_{t+1}^{(n)} = \gamma_t^{(n)} f_t^{(n)}.
\tag{46}
```

The real advantage of rewriting the joint distribution as in Eq. 45 comes from
splitting this joint distribution as a product of conditional distributions of
the form

```math
\pi(f_t^{(n)}, \gamma_{t}^{(n)} \mid \bar{s}_t) =
\pi(f_t^{(n)} \mid \gamma_{t}^{(n)}, \bar{s}_t)
\pi(\gamma_{t}^{(n)} \mid \bar{s}_t).
\tag{47}
```

Written in this form, we can finally propose a probabilistic model for how the
mean fitness relates to the frequency ratios we determine in our experiments.
The second term on the right-hand side of Eq. 47 relates how the determined
frequency ratio ``\gamma_t^{(b)}`` relates to the mean fitness ``\bar{s}_t``.
From Eq. 43 and Eq. 44, we can write

```math
\ln \gamma_t^{(n)} = - \bar{s}_t + \varepsilon_t^{(n)},
\tag{48}
```

where, for simplicity, we set ``\tau = 1``. Note that we added an extra term,
``\varepsilon_t^{(n)}``, characterizing the deviations of the measurements from
the theoretical model. We assume these errors are normally distributed with mean
zero and some standard deviation ``\sigma_t``, implying that

```math
\ln \gamma_t^{(n)} \mid \bar{s}_t, \sigma_t  \sim 
\mathcal{N}\left(-\bar{s}_t, \sigma_t \right),
\tag{49}
```

where we include the nuisance parameter ``\sigma_t`` to be determined. If we
assume the log frequency ratio is normally distributed, this implies the
frequency ratio itself is distributed log-normal. This means that

```math
\gamma_t^{(n)} \mid \bar{s}_t, \sigma_t  \sim 
\log \mathcal{N}\left(-\bar{s}_t, \sigma_t \right).
\tag{50}
```

Having added the nuisance parameter ``\sigma_t`` implies that we must update
Eq. 42 to

```math
\pi(\bar{s}_t, \sigma_t \mid f_t^{(n)}, f_{t+1}^{(n)}) \propto
\pi(f_t^{(n)}, \gamma_t^{(n)} \mid \bar{s}_t, \sigma_t) 
\pi(\bar{s}_t) \pi(\sigma_t),
\tag{51}
```

where we assume the prior for each parameter is independent, i.e.,

```math
\pi(\bar{s}_t, \sigma_t) = \pi(\bar{s}_t) \pi(\sigma_t).
\tag{52}
```

For numerical stability, we will select weakly-informative priors for both of
these parameters. In the case of the nuisance parameter ``\sigma_t``, the prior
must be restricted to positive values only, since standard deviations cannot be
negative.

For the first term on the right-hand side of Eq. 47, ``\pi(f_t^{(n)} \mid
\gamma_{t}^{(n)}, \bar{s}_t)``, we remove the conditioning on the population
mean fitness since it does not add any information on top of what the frequency
ratio ``\gamma_t^{(n)}`` already gives. Therefore, we have

```math
\pi(f_t^{(n)} \mid \gamma_{t}^{(n)}, \bar{s}_t) =
\pi(f_t^{(n)} \mid \gamma_{t}^{(n)}).
\tag{53}
```

The right-hand side of Eq. 53 asks us to compute the probability of observing a
frequency value ``f_t^{(n)}`` given that we get to observe the ratio
``\gamma_{t}^{(n)}``. If the ratio happened to be ``\gamma_{t}^{(n)} = 2``, we
could have ``f_{t+1}^{(n)} = 1`` and ``f_{t+1}^{(n)} = 0.5``, for example.
Although, it would be equally likely that ``f_{t+1}^{(n)} = 0.6`` and
``f_{t+1}^{(n)} = 0.3`` or ``f_{t+1}^{(n)} = 0.1`` and ``f_{t+1}^{(n)} = 0.05``
for that matter. If we only get to observe the frequency ratio
``\gamma_t^{(n)}``, we know that the numerator ``f_{t+1}^{(n)}`` can only take
values between zero and one, all of them being equally likely given only the
information on the ratio. As a consequence, the value of the frequency in the
denominator ``f_{t}^{(n)}`` is restricted to fall in the range

```math
f_{t}^{(n)} \in \left(0, \frac{1}{\gamma_t^{(n)}} \right].
\tag{54}
```

A priori, we do not have any reason to favor any value over any other, therefore
it is natural to write

```math
f_t^{(n)} \mid \gamma_t^{(n)} \sim 
\operatorname{Uniform}\left( 0, \frac{1}{\gamma_t^{(n)}} \right).
\tag{55}
```

#### Summary

Putting all the pieces we have developed in this section together results in an
inference for the population mean fitness values of the form

```math
\pi(
    \underline{\bar{s}}_T, \underline{\sigma}_T \mid \underline{\underline{F}}
) \propto
\prod_{t=1}^{T-1} \left\{
    \prod_{n=1}^N \left[
        \pi(f_t^{(n)} \mid \gamma_t^{(n)}) 
        \pi(\gamma_t^{(n)} \mid \bar{s}_t, \sigma_t)
    \right]
    \pi(\bar{s}_t) \pi(\sigma_t)
\right\},
\tag{56}
```

where we have

```math
f_t^{(n)} \mid \gamma_t^{(n)} \sim 
\operatorname{Uniform} \left(0, \frac{1}{\gamma_t^{(n)}} \right),
\tag{57}
```

```math
\gamma_t^{(n)} \mid \bar{s}_t, \sigma_t \sim 
\log\mathcal{N}(\bar{s}_t, \sigma_t),
\tag{58}
```

```math
\bar{s}_t \sim \mathcal{N}(0, \sigma_{\bar{s}_t}),
\tag{59}
```

and

```math
\sigma_t \sim \log\mathcal{N}(\mu_{\sigma_t}, \sigma_{\sigma_t}),
\tag{60}
```

where ``\sigma_{\bar{s}_t}``, ``\mu_{\sigma_t}``, and ``\sigma_{\sigma_t}`` are
user-defined parameters.

### Mutant relative fitness uncertainty ``\pi(\underline{s}^M \mid \underline{\bar{s}}_T, \underline{\underline{F}}, \underline{\underline{R}})`` 

The last piece of our inference is the piece that we care about the most: the
probability distribution of all the mutants' relative fitness, given the
inferred population mean fitness and the frequencies. First, we assume that all
fitness values are independent of each other. This allows us to write

```math
\pi(
    \underline{s}^M \mid 
    \underline{\bar{s}}_T, \underline{\underline{F}}, \underline{\underline{R}}
) = 
\prod_{m=1}^M \pi(
    s^{(m)} \mid
    \underline{\bar{s}}_T, \underline{\underline{F}}, \underline{\underline{R}}
).
\tag{61}
```

Furthermore, as was the case with the population mean fitness, our fitness model
relates frequencies, not raw reads. Moreover, the fitness value of mutant ``m``
only depends on the frequencies of such mutant. Therefore, we can simplify the
conditioning to

```math
\pi(
    s^{(m)} \mid
    \underline{\bar{s}}_T, \underline{\underline{F}}, \underline{\underline{R}}
) = 
\pi(s^{(m)} \mid \underline{\bar{s}}_T, \underline{f}^{(m)}),
\tag{62}
```

where

```math
\underline{f}^{(m)} = (f_0^{(m)}, f_1^{(m)}, \ldots, f_T^{(m)})^\dagger,
\tag{63}
```

is the vector containing the frequency time series for mutant ``m``. Writing
Bayes' theorem for the right-hand side of Eq. 62 results in

```math
\pi(s^{(m)} \mid \underline{\bar{s}}_T, \underline{f}^{(m)}) \propto
\pi(\underline{f}^{(m)} \mid \underline{\bar{s}}_T, s^{(m)})
\pi(s^{(m)} \mid \underline{\bar{s}}_T).
\tag{64}
```

Notice the conditioning on the mean fitness values ``\underline{\bar{s}}_T`` is
not inverted since we already inferred these values.

Following the logic used in sec-bayes_meanfit, let us define

```math
\underline{\gamma}^{(m)} = 
(\gamma_0^{(m)}, \gamma_1^{(m)}, \ldots, \gamma_{T-1}^{m})^\dagger,
\tag{65}
```

where each entry ``\gamma_t^{(m)}`` is defined by eq-gamma_def. In the same way
we rewrote the joint distribution between two adjacent time point frequencies to
the joint distribution between one of the frequencies and the ratio of both
frequencies in eq-joint_freq_gamma, we can rewrite the joint distribution of
the frequency time series for mutant ``m`` as

```math
\pi(\underline{f}^{(m)} \mid \underline{\bar{s}}_T, s^{(m)}) =
\pi(f_0^{(m)}, \underline{\gamma}^{(m)} \mid \underline{\bar{s}}_T, s^{(m)}).
\tag{66}
```

One can think about Eq. 66 as saying that knowing the individual frequencies at
each time point contain equivalent information as knowing the initial frequency
and the subsequent ratios of frequencies. This is because if we want to know the
value of ``f_1^{(m)}`` given the ratios, we only need to compute

```math
f_1^{(m)} = \gamma_0^{(m)} f_0^{(m)}.
\tag{67}
```

Moreover, if we want to know ``f_2^{(m)}``, we have

```math
f_2^{(m)} = \gamma_1^{(m)} f_1^{(m)} =
\gamma_1^{(m)} \left(\gamma_0^{(m)} f_0^{(m)}\right),
\tag{68}
```

and so on. We can then write the joint distribution on the right-hand side of
Eq. 66 as a product of conditional distributions of the form

```math
\begin{aligned}
\pi(f_0^{(m)}, \underline{\gamma}^{(m)} \mid \underline{\bar{s}}_T, s^{(m)}) =
&\pi(
    f_0^{(m)} \mid 
    \gamma_0^{(m)}, \ldots, \gamma_{T-1}^{(m)}, \underline{\bar{s}}_T, s^{(m)}
) \times \\
&\pi(
    \gamma_0^{(m)} \mid 
    \gamma_1^{(m)}, \ldots, \gamma_{T-1}^{(m)}, \underline{\bar{s}}_T, s^{(m)}
) \times \\
&\pi(
    \gamma_1^{(m)} \mid 
    \gamma_2^{(m)}, \ldots, \gamma_{T-1}^{(m)}, \underline{\bar{s}}_T, s^{(m)}
) \times \\
&\vdots \\
&\pi(
    \gamma_{T-2}^{(m)} \mid \gamma_{T-1}^{(m)}, \underline{\bar{s}}_T, s^{(m)}
) \times \\
&\pi(\gamma_{T-1}^{(m)} \mid \underline{\bar{s}}_T, s^{(m)}).
\end{aligned}
\tag{69}
```

Writing the fitness model in eq-fitness as

```math
\gamma_t^{(m)} = \frac{f_{t+1}^{(m)}}{f_t^{(m)}} = 
\mathrm{e}^{(s^{(m)} - s_t)\tau},
\tag{70}
```

reveals that the value of each of the ratios ``\gamma_t^{(m)}`` only depends on
the corresponding fitness value ``\bar{s}_t`` and the relative fitness
``s^{(m)}``. Therefore, we can remove most of the conditioning on the right-hand
side of Eq. 69 resulting in a much simpler joint distribution of the form

```math
\begin{aligned}
\pi(f_0^{(m)}, \underline{\gamma}^{(m)} \mid \underline{\bar{s}}_T, s^{(m)}) =
&\pi(f_0^{(m)} \mid \gamma_0^{(m)}) \times \\
&\pi(\gamma_0^{(m)} \mid \bar{s}_0, s^{(m)}) \times \\
&\pi(\gamma_1^{(m)} \mid \bar{s}_1, s^{(m)}) \times \\
&\vdots \\
&\pi(\gamma_{T-2}^{(m)} \mid \bar{s}_{T-2}, s^{(m)}) \times \\
&\pi(\gamma_{T-1}^{(m)} \mid \bar{s}_{T-1}, s^{(m)}),
\end{aligned}
\tag{71}
```

where for the first term on the right-hand side of Eq. 71 we apply the same
logic as in eq-freq_cond_gamma to remove all other dependencies. We emphasize
that although Eq. 71 looks like a series of independent inferences, the value
of the relative fitness ``s^{(m)}`` is shared among all of them. This means that
the parameter is not inferred individually for each time point, resulting in
different estimates of the parameter, but each time point contributes
independently to the inference of a single estimate of ``s^{(m)}``.

Using equivalent arguments to those in sec-bayes_meanfit, we assume

```math
f_0^{(m)} \mid \gamma_0^{(m)} \sim 
\operatorname{Uniform}\left(0, \frac{1}{\gamma_0^{(m)}} \right),
\tag{72}
```
and

```math
\gamma_t^{(m)} \mid \bar{s}_t, s^{(m)}, \sigma^{(m)} \sim 
\log\mathcal{N}\left(s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
\tag{73}
```

where we add the nuisance parameter ``\sigma^{(m)}`` to the inference. Notice
that this parameter is not indexed by time. This means that we assume the
deviations from the theoretical prediction do not depend on time, but only on
the mutant. Adding the nuisance parameter demands us to update Eq. 64to

```math
\pi(
    s^{(m)}, \sigma^{(m)} \mid \underline{\bar{s}}_T, \underline{f}^{(m)}
) \propto
\pi(\underline{f}^{(m)} \mid \underline{\bar{s}}_T, s^{(m)}, \sigma^{(m)})
\pi(s^{(m)}) \pi(\sigma^{(m)}),
\tag{74}
```

where we assume independent priors for both parameters. We also removed the
conditioning on the values of the mean fitness as knowing such values does not
change our prior information about the possible range of values that the
parameters can take. As with the priors on sec-bayes_meanfit, we will assign
weakly-informative priors to these parameters.

#### Summary

With all pieces in place, we write the full inference of the relative fitness
values as

```math
\pi(
    \underline{s}^M ,\underline{\sigma}^M \mid 
    \underline{\bar{s}}_T, \underline{\underline{F}}
) \propto
\prod_{m=1}^M \left\{ 
    \pi(f_0^{(m)} \mid \gamma_0^{(m)})
    \prod_{t=0}^{T-1} \left[
        \pi(\gamma_t^{(m)} \mid \bar{s}_t, s^{(m)}, \sigma^{(m)})
    \right]
    \pi(s^{(m)}) \pi(\sigma^{(m)})
\right\},
\tag{75}
```

where

```math
f_0^{(m)} \mid \gamma_0^{(m)} \sim 
\operatorname{Uniform}\left(0, \frac{1}{\gamma_0^{(m)}} \right),
\tag{76}
```

```math
\gamma_t^{(m)} \mid \bar{s}_t, s^{(m)}, \sigma^{(m)} \sim 
\log\mathcal{N}\left(s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
\tag{77}
```

```math
s^{(m)} \sim \mathcal{N}(0, \sigma_{s^{(m)}}),
\tag{78}
```

and

```math
\sigma^{(m)} \sim \log\mathcal{N}(\mu_{\sigma^{(m)}}, \sigma_{\sigma^{(m)}}),
```

where ``\sigma_{s^{(m)}}``, ``\mu_{\sigma^{(m)}}``, and
``\sigma_{\sigma^{(m)}}`` are user-defined parameters.