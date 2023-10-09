# vi

## Primer on Variational Inference

In this section, we will briefly introduce the idea behind variational
inference. Recall that any Bayesian inference problem deals with the joint
distribution between observations ``\underline{x}`` and unobserved latent
variables ``\underline{\theta}``. This joint distribution can be written as the
product of a distribution of the observations ``\underline{x}`` conditioned on
the ``\underline{\theta}`` and the marginal distribution of these latent
variables, i.e.,

```math
\pi(\underline{x}, \underline{\theta}) =
\pi(\underline{x} \mid \underline{\theta}) \pi(\underline{\theta}).
\tag{1}
```

A Bayesian inference pipeline's objective is to compute the latent variables'
posterior probability given a set of observations. This computation is
equivalent to updating our prior beliefs about the set of values that the latent
variables take after taking in new data. We write this as Bayes theorem

```math
\pi(\underline{\theta} \mid \underline{x}) = 
\frac{
        \pi(\underline{x} \mid \underline{\theta})\pi(\underline{\theta})
    }{
        \pi(\underline{x})
    }.
\tag{2}
```

The main technical challenge for working with Eq. 2 comes from the computation
of the denominator, also known as the *evidence* or the *marginalized
likelihood*. The reason computing this term is challenging is because it
involves a (potentially) high-dimensional integral of the form

```math
\pi(\underline{x}) = 
\int\cdots\int d^K\underline{\theta}\; \pi(\underline{x}, \underline{\theta}) = 
\int\cdots\int d^K\underline{\theta}\; \pi(\underline{x} \mid \underline{\theta})
\pi(\underline{\theta}),
\tag{3}
```

where ``K`` is the dimesionality of the ``\underline{\theta}`` vector. Here, the
integrals are taken over the support---the set of values valid for the
distribution---of ``\pi(\underline{\theta})``. However, only a few selected
distributions have a closed analytical form; thus, in most cases
Eq. 3 must be solved numerically.

Integration in high-dimensional spaces can be computationally extremely
challenging. For a naive numerical quadrature procedure, integrating over a grid
of values for each dimension of ``\underline{\theta}`` comes with an exponential
explosion of the number of required grid point evaluations, most of which do not
contribute significantly to the integration. To gain visual intuition about this
challenge, imagine integrating the function depicted in fig-SI01. If the
location of the high-density region (dark peak) is unknown, numerical quadrature
requires many grid points to ensure we capture this peak. However, most of the
numerical evaluations of the function on the grid points do not contribute
significantly to the integral. Therefore, our computational resources are wasted
on insignificant evaluations. This only gets worse as the number of dimensions
increases since the number of grid point evaluation scales exponentially.



Modern Markov Chain Monte Carlo algorithms, such as Hamiltonian Monte Carlo, can
efficiently perform this high-dimensional integration by utilizing gradient
information from the target density betancourt2017. Nevertheless, these
sampling-based methods become prohibitively slow for the number of dimensions
our present inference problem presents. Thus, there is a need to find scalable
methods for the inference problem in Eq. 2.

Variational inference circumvents these technical challenges by proposing an
approximate solution to the problem. Instead of working with the posterior
distribution in its full glory ``\pi(\underline{\theta} \mid \underline{x})``,
let us propose an approximate posterior distribution ``q_\phi`` that belongs to
a distribution family fully parametrized by ``\phi``. For example, let us say
that the distribution ``q_\phi`` belongs to the family of multivariate Normal
distributions such that ``\phi = (\underline{\mu},
\underline{\underline{\Sigma}})``, where ``\underline{\mu}`` is the vector of
means and ``\underline{\underline{\Sigma}}`` is the covariance matrix. If we
replace ``\pi`` by ``q_\phi``, we want ``q_\phi`` to resemble the original
posterior as much as possible. Mathematically, this can be expressed as
minimizing a "*distance metric*"---the Kullback-Leibler (KL) divergence, for
example---between the distributions. Note that we use quotation marks because,
formally, the KL divergence is not a distance metric since it is not symmetric.
Nevertheless, the variational objective is set to find a distribution
``q_\phi^*`` such that

```math
q_\phi^*(\underline{\theta}) =
\min_\phi D_{KL}\left(
    q_\phi(\underline{\theta}) \vert\vert 
    \pi(\underline{\theta} \mid \underline{x})
\right),
\tag{4}
```

where ``D_{KL}`` is the KL divergence. Furthermore, we highlight that the KL
divergence is a strictly positive number, i.e.,

```math
D_{KL}\left(
    q_\phi(\underline{\theta}) \vert\vert 
    \pi(\underline{\theta} \mid \underline{x})
\right) \geq 0,
\tag{5}
```

as this property will become important later on.

At first sight, Eq. 4 does not improve the situation but only introduces further
technical complications. After all, the definition of the KL divergence

```math
D_{KL}\left(
    q_\phi(\underline{\theta}) \vert\vert 
    \pi(\underline{\theta} \mid \underline{x})
\right) \equiv 
\int \cdots \int d^K\underline{\theta}\;
q_\phi(\underline{\theta})
\ln \frac{
    q_\phi(\underline{\theta})
}{
    \pi(\underline{\theta} \mid \underline{x})
},
\tag{6}
```

includes the posterior distribution ``\pi(\underline{\theta} \mid
\underline{x})`` we are trying to get around. However, let us manipulate Eq. 6
to beat it to a more reasonable form. First, we can use the properties of the
logarithms to write

```math
D_{KL}\left(
    q_\phi(\underline{\theta}) \vert\vert 
    \pi(\underline{\theta} \mid \underline{x})
\right) = 
\int d^K\underline{\theta}\; q_\phi(\underline{\theta})
\ln q_\phi(\underline{\theta}) -
\int d^K\underline{\theta}\; q_\phi(\underline{\theta})
\ln \pi(\underline{\theta} \mid \underline{x}),
\tag{7}
```

where, for convenience, we write a single integration sign
(``d^K\underline{\theta}\;`` still represents a multi-dimensional differential).
For the second term in Eq. 7, we can substitute the term inside the logarithm
using Eq. 2. This results in

```math
\begin{aligned}
D_{KL}\left(
    q_\phi(\underline{\theta}) \vert\vert 
    \pi(\underline{\theta} \mid \underline{x})
\right) &= 
\int d^K\underline{\theta}\; q_\phi(\underline{\theta})
\ln q_\phi(\underline{\theta}) \\
&- \int d^K\underline{\theta}\; q_\phi(\underline{\theta})
\ln \left( 
    \frac{
        \pi(\underline{x} \mid \underline{\theta})\pi(\underline{\theta})
    }{
        \pi(\underline{x})
    }
\right).
\end{aligned}
\tag{8}
```

Again, using the properties of logarithms, we can split Eq. 8, obtaining

```math
\begin{aligned}
D_{KL}\left(
    q_\phi(\underline{\theta}) \vert\vert 
    \pi(\underline{\theta} \mid \underline{x})
\right) &= 
\int d^K\underline{\theta}\; q_\phi(\underline{\theta})
\ln q_\phi(\underline{\theta}) \\
&-\int d^K\underline{\theta}\; q_\phi(\underline{\theta})
\ln \pi(\underline{x} \mid \underline{\theta}) \\
&-\int d^K\underline{\theta}\; q_\phi(\underline{\theta})
\ln \pi(\underline{\theta}) \\
&+\int d^K\underline{\theta}\; q_\phi(\underline{\theta})
\ln \pi(\underline{x}).
\end{aligned}
\tag{9}
```

It is convenient to write Eq. 9 as

```math
\begin{aligned}
D_{KL}\left(
    q_\phi(\underline{\theta}) \vert\vert 
    \pi(\underline{\theta} \mid \underline{x})
\right) &= 
\int d^K\underline{\theta}\; q_\phi(\underline{\theta})
\ln \frac{
    q_\phi(\underline{\theta})
    }{
        \pi(\underline{\theta})
    } \\
&-\int d^K\underline{\theta}\; q_\phi(\underline{\theta})
\ln \pi(\underline{x} \mid \underline{\theta}) \\
&+ \ln \pi(\underline{x}) 
\int d^K\underline{\theta}\; q_\phi(\underline{\theta}),
\end{aligned}
\tag{10}
```

where for the last term, we can take ``\ln \pi(\underline{x})`` out of the
integral since it does not depend on ``\underline{\theta}``. Lastly, we utilize
two properties:

- The proposed approximate distribution must be normalized, i.e.,

```math
\int d^K\underline{\theta}\; q_\phi(\underline{\theta}) = 1.
\tag{11}
```

- The law of the unconscious statistician (LOTUS) establishes that for any
probability density function, it must be true that

```math
\int d^K\underline{\theta}\; q_\phi(\underline{\theta})
f(\underline{\theta}) = \left\langle 
    f(\underline{\theta}) 
\right\rangle_{q_\phi},
\tag{12}
```

where ``\left\langle\cdot\right\rangle_{q_\phi}`` is the expected value over the
``q_\phi`` distribution.

Using these two properties, the positivity constraint on the KL divergence in
Eq. 5, and the definition of the KL divergence in Eq. 6 we can rewrite Eq. 10 as

```math
D_{KL}\left( 
    q_\phi(\underline{\theta}) \vert \vert
    \pi(\underline{\theta}) 
\right) -
\left\langle
    \ln \pi(\underline{x} \mid \underline{\theta})
\right\rangle_{q_\phi}
\geq - \ln \pi(\underline{x}).
\tag{13}
```

Multiplying by a minus one, we have the functional form of the so-called
evidence lower bound (ELBO) kingma2014,

```math
\underbrace{
    \ln \pi(\underline{x})
}_{\text{log evidence}} \geq
\underbrace{
    \left\langle
        \ln \pi(\underline{x} \mid \underline{\theta})
    \right\rangle_{q_\phi} -
    D_{KL}\left( 
        q_\phi(\underline{\theta}) \vert \vert
        \pi(\underline{\theta}) 
    \right)
}_{\text{ELBO}}.
\tag{14}
```

Let us recapitulate where we are. We started by presenting the challenge of
working with Bayes' theorem, as it requires a high-dimensional integral of the
form in Eq. 3. As an alternative, variational inference posits to approximate
the posterior distribution ``\pi(\underline{\theta} \mid \underline{x})`` with a
parametric distribution ``q_\phi(\underline{\theta})``. By minimizing the KL
divergence between these distributions, we arrive at the result in Eq. 14, where
the left-hand side---the log marginalized likelihood or log evidence---we cannot
compute for technical/computational reasons. However, the right-hand side is
composed of things we can easily evaluate. We can easily evaluate the
log-likelihood ``\ln \pi(\underline{x} \mid \underline{\theta})`` and the KL
divergence between our proposed approximate distribution
``q_\phi(\underline{\theta})`` and the prior distribution
``\pi(\underline{\theta})``. Moreover, we can compute the gradients of these
functions with respect to the parameters of our proposed distribution. This last
point implies that we can change the parameters of the proposed distribution to
maximize the ELBO. And, although we cannot compute the left-hand side of Eq. 14,
we know that however large we make the ELBO, it will always be smaller than (or
equal) the log-marginal likelihood. Therefore, the larger we can make the ELBO
by modifying the parameters ``\phi``, the closer it gets to the log-marginal
likelihood, and, as a consequence, the better our proposed distribution
``q_\phi(\underline{\theta})`` gets to the true posterior distribution
``\pi(\underline{\theta} \mid \underline{x})``.

In this sense, variational inference turns the intractable numerical integration
problem to an optimization routine, for which there are several algorithms 
available.

### ADVI algorithm

To maximize the right-hand side of Eq. 14, the Automatic Differentiation
Variational Inference (ADVI) algorithm developed in [kucukelbir2016] takes
advantage of advances in probabilistic programming languages to generate a
robust method to perform this optimization. Without going into the details of
the algorithm implementation, for our purposes, it suffices to say that we
define our joint distribution ``\pi(\underline{\theta}, \underline{x})`` as the
product defined in Eq. 1. ADVI then proposes an approximate variational
distribution ``q_\phi`` that can either be a multivariate Normal distribution
with a diagonal covariance matrix, i.e.,

```math
\phi = (\underline{\mu}, \underline{\underline{D}}),
\tag{15}
```

where ``\underline{\underline{D}}`` is the identity matrix, with the diagonal
elements given by the vector of variances ``\underline{\sigma}^2`` for each
variable or a full-rank multivariate Normal distribution

```math
\phi = (\underline{\mu}, \underline{\underline{\Sigma}}).
\tag{16}
```

Then, the parameters are initialized in some value ``\phi_o``. These parameters
are iteratively updated by computing the gradient of the ELBO (right-hand side
of Eq. 14), hereafter defined as ``\mathcal{L}``, with respect to the
parameters, 

```math
\nabla_\phi \mathcal{L} = \nabla_{\underline{\mu}} \mathcal{L} + 
\nabla_{\underline{\sigma}}\mathcal{L},
\tag{17}
```

and then computing

```math
\phi_{t+1} = \phi_{t} + \eta \nabla_\phi \mathcal{L},
\tag{18}
```

where ``\eta`` defines the step size.

This short explanation behind the ADVI algorithm is intended only to gain
intuition on how the optimal variational distribution ``q_\phi`` be computed.
There are many nuances in the implementation of the ADVI algorithm. We invite
the user to look at the original reference for further details.

## vi module

This `vi` module includes (so far) a single function to run variational 
inference using the ADVI algorithm implemented in `Turing.jl`.

```@autodocs
Modules = [BarBay.vi]
Order   = [:function, :type]
```
