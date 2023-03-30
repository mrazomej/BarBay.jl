# Import libraries to define distributions
import Distributions

# Import libraries relevant for MCMC
import Turing

##
# Export functions

# Export mean fitness model
export mean_fitness_neutrals_lognormal

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Population Mean Fitness π(sₜ | Data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    mean_fitness_neutrals_lognormal(r̲ₜ, r̲ₜ₊₁, α̲, sₜ_prior, σₜ_prior)

`Turing.jl` model to sample the posterior for a single population mean fitness
value sₜ, given the raw barcode counts. 

# Model 
For this inference, we can write Bayes theorem as

```math
\pi(
    \bar{s}_t, \sigma_t, \underline{f}_t, \underline{f}_{t+1} \mid
    \underline{r}_t, \underline{r}_{t+1}
) \propto
\prod_{n=1}^N \left[
        \pi(f_t^{(n)} \mid \gamma_t^{(n)}) 
        \pi(\gamma_t^{(n)} \mid \bar{s}_t, \sigma_t)
\right]
\pi(\bar{s}_t) \pi(\sigma_t)
\pi(\underline{f}_t \mid \underline{r}_t)
\pi(\underline{f}_{t+1} \mid \underline{r}_{t+1})
```
where
```math
\gamma_t^{(n)} \equiv \frac{f_{t+1}^{(n)}}{f_t^{n}}.
```

The parametric distributions assumed in this model are of the form
```math
f_t^{(n)} \mid \gamma_t^{(n)} \sim 
\operatorname{Uniform} \left(0, \frac{1}{\gamma_t^{(n)}} \right),
```

```math
\gamma_t^{(n)} \mid \bar{s}_t, \sigma_t \sim 
\log\mathcal{N}(\bar{s}_t, \sigma_t),
```

```math
\bar{s}_t \sim \mathcal{N}(\mu_{\bar{s}_t}, \sigma_{\bar{s}_t}),
```

```math
\sigma_t \sim 
\operatorname{Half}-\mathcal{N}(\mu_{\sigma_t}, \sigma_{\sigma_t}),
```

```math
\underline{f}_t \mid \underline{r}_t \sim 
\operatorname{Dirichlet}(\underline{\alpha}_t + \underline{r}_t),
```
and
```math
\underline{f}_{t+1} \mid \underline{r}_{t+1} \sim 
\operatorname{Dirichlet}(\underline{\alpha}_{t+1} + \underline{r}_{t+1}).
```

For this inference, we enforce all frequencies to be > 0 (even for barcodes
with zero reads) to compute ``\gamma_t^{(n)}``.

The user defines the distribution parameters as:
- ``\underline{\alpha}_t``: `α̲`.
- ``[\mu_{\bar{s}_t}, \sigma_{\bar{s}_t}]``: `sₜ_prior`.
- ``[\mu_{\sigma_t}, \sigma_{\sigma_t}]``: `σₜ_prior`.

# Arguments
- `r̲ₜ::Vector{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t`. NOTE: The last entry of the array must
  be the sum of all of the counts from mutant lineages.
- `r̲ₜ₊₁::Vector{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t + 1`. NOTE: The last entry of the array
  must be the sum of all of the counts from mutant lineages.
- `α̲::Vector{Float64}`: Parameters for Dirichlet prior distribution.

## Optional arguments
- `sₜ_prior::Vector{Real}=[0.0, 2.0]`: Parameters for the mean fitness prior
  distribution π(sₜ).
- `σₜ_prior::Vector{Real}=[0.0, 1.0]`: Parameters for the nuisance standard
  deviation parameter prior distribution π(σₜ).
- `σₜ_trunc::Real=0.0`: Value at which truncate the normal distribution to
  define it as a half-normal.
"""
Turing.@model function mean_fitness_neutrals_lognormal(
    r̲ₜ::Vector{Int64},
    r̲ₜ₊₁::Vector{Int64},
    α̲::Vector{Float64};
    sₜ_prior::Vector{Real}=[0.0, 2.0],
    σₜ_prior::Vector{Real}=[0.0, 1.0],
    σₜ_trunc::Real=0.0
)
    # Prior on mean fitness sₜ
    sₜ ~ Turing.Normal(sₜ_prior...)
    # Prior on LogNormal error σₜ
    σₜ ~ Turing.truncated(Turing.Normal(σₜ_prior...); lower=σₜ_trunc)

    # Frequency distribution from Multinomial-Dirichlet model
    f̲ₜ ~ Turing.Dirichlet(α̲ .+ r̲ₜ)
    f̲ₜ₊₁ ~ Turing.Dirichlet(α̲ .+ r̲ₜ₊₁)

    # Check that all distributions are greater than zero. Although the counts
    # could be zero, we assume that the real frequencies are non-zero always.
    if any(iszero.(f̲ₜ)) | any(iszero.(f̲ₜ₊₁))
        Turing.@addlogprob! -Inf
        return
    end

    # Compute frequency ratio
    γₜ = (f̲ₜ₊₁./f̲ₜ)[1:end-1]

    # Sample posterior for frequency ratio. NOTE: Since the quantity γₜ is not
    # given as input observations to the model, Turing does not add the
    # evaluation of the log likelihood to the overall posterior. This ends up
    # having sₜ sampled directly from the prior. To force Turing to consider γₜ
    # as an "observed" variable, we must force the addition of the log density
    # using the @addlogprob! macro.
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            FillArrays.Fill(-sₜ, length(γₜ)),
            LinearAlgebra.I(length(γₜ)) .* σₜ^2
        ),
        γₜ
    )
end # @model function

@doc raw"""
    mean_fitness_neutrals_lognormal(r̲ₜ, r̲ₜ₊₁, α̲, sₜ_prior, σₜ_prior)

`Turing.jl` model to sample out of the posterior for a single population mean
fitness value sₜ, given the raw barcode counts. Note: this `method` allows the
use of any prior distribution, different from the Normal and Half-Normal priors.

# Model 
For this inference, we can write Bayes theorem as

```math
\pi(
    \bar{s}_t, \sigma_t, \underline{f}_t, \underline{f}_{t+1} \mid
    \underline{r}_t, \underline{r}_{t+1}
) \propto
\prod_{n=1}^N \left[
        \pi(f_t^{(n)} \mid \gamma_t^{(n)}) 
        \pi(\gamma_t^{(n)} \mid \bar{s}_t, \sigma_t)
\right]
\pi(\bar{s}_t) \pi(\sigma_t)
\pi(\underline{f}_t \mid \underline{r}_t)
\pi(\underline{f}_{t+1} \mid \underline{r}_{t+1})
```
where
```math
\gamma_t^{(n)} \equiv \frac{f_{t+1}^{(n)}}{f_t^{n}}.
```

The parametric distributions assumed in this model are of the form
```math
f_t^{(n)} \mid \gamma_t^{(n)} \sim 
\operatorname{Uniform} \left(0, \frac{1}{\gamma_t^{(n)}} \right),
```

```math
\gamma_t^{(n)} \mid \bar{s}_t, \sigma_t \sim 
\log\mathcal{N}(\bar{s}_t, \sigma_t),
```

```math
\bar{s}_t \sim \operatorname{User-defined},
```

```math
\sigma_t \sim  \operatorname{User-defined},
```

```math
\underline{f}_t \mid \underline{r}_t \sim 
\operatorname{Dirichlet}(\underline{\alpha}_t + \underline{r}_t),
```
and
```math
\underline{f}_{t+1} \mid \underline{r}_{t+1} \sim 
\operatorname{Dirichlet}(\underline{\alpha}_{t+1} + \underline{r}_{t+1}).
```

For this inference, we enforce all frequencies to be > 0 (even for barcodes
with zero reads) to compute ``\gamma_t^{(n)}``.

The user defines the distribution parameters as:
- ``\underline{\alpha}_t``: `α̲`.

# Arguments
- `r̲ₜ::Array{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t`. NOTE: The last entry of the array must
  be the sum of all of the counts from mutant lineages.
- `r̲ₜ₊₁::Array{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t + 1`. NOTE: The last entry of the array
  must be the sum of all of the counts from mutant lineages.
- `α̲::Array{Float64}`: Parameters for Dirichlet prior distribution.
- `sₜ_prior::Distributions.ContinuousUnivariateDistribution`: Parametrized
  univariate continuous distribution for the prior on the mean fitness π(sₜ).
- `σₜ_prior:::Distributions.ContinuousUnivariateDistribution`: Parametrized
  univariate continuous distribution for the prior on the nuisance standard
  deviation of the log-normal likelihood π(σₜ).
"""
Turing.@model function mean_fitness_neutrals_lognormal(
    r̲ₜ::Vector{Int64},
    r̲ₜ₊₁::Vector{Int64},
    α̲::Vector{Float64},
    sₜ_prior::Distributions.ContinuousUnivariateDistribution,
    σₜ_prior::Distributions.ContinuousUnivariateDistribution;
)
    # Prior on mean fitness sₜ
    sₜ ~ sₜ_prior
    # Prior on LogNormal error σₜ
    σₜ ~ σₜ_prior

    # Frequency distribution from Multinomial-Dirichlet model
    f̲ₜ ~ Turing.Dirichlet(α̲ .+ r̲ₜ)
    f̲ₜ₊₁ ~ Turing.Dirichlet(α̲ .+ r̲ₜ₊₁)

    # Check that all distributions are greater than zero. Although the counts
    # could be zero, we assume that the real frequencies are non-zero always.
    if any(iszero.(f̲ₜ)) | any(iszero.(f̲ₜ₊₁))
        Turing.@addlogprob! -Inf
        return
    end

    # Compute frequency ratio
    γₜ = (f̲ₜ₊₁./f̲ₜ)[1:end-1]

    # Sample posterior for frequency ratio. NOTE: Since the quantity γₜ is not
    # given as input observations to the model, Turing does not add the
    # evaluation of the log likelihood to the overall posterior. This ends up
    # having sₜ sampled directly from the prior. To force Turing to consider γₜ
    # as an "observed" variable, we must force the addition of the log density
    # using the @addlogprob! macro.
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            FillArrays.Fill(-sₜ, length(γₜ)),
            LinearAlgebra.I(length(γₜ)) .* σₜ^2
        ),
        γₜ
    )
end # @model function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Mutant fitness π(s⁽ᵐ⁾ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #