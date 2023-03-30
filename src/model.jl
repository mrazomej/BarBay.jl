# Import basic math
import LinearAlgebra

# Import libraries to define distributions
import Distributions

# Import libraries relevant for MCMC
import Turing

# Functionality for constructing arrays with identical elements efficiently
import FillArrays
##
# Export functions

# Export mean fitness model
export mean_fitness_neutrals_lognormal
# Export mutant fitness model
export mutant_fitness_lognormal

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Population Mean Fitness π(sₜ | Data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    mean_fitness_neutrals_lognormal(r̲ₜ, r̲ₜ₊₁, α̲, s_prior, σ_prior)

`Turing.jl` model to sample the posterior for a single population mean fitness
value `sₜ`, given the raw barcode counts. 

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
- ``[\mu_{\bar{s}_t}, \sigma_{\bar{s}_t}]``: `s_prior`.
- ``[\mu_{\sigma_t}, \sigma_{\sigma_t}]``: `σ_prior`.

# Arguments
- `r̲ₜ::Vector{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t`. NOTE: The last entry of the array must
  be the sum of all of the counts from mutant lineages.
- `r̲ₜ₊₁::Vector{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t + 1`. NOTE: The last entry of the array
  must be the sum of all of the counts from mutant lineages.
- `α̲::Vector{Float64}`: Parameters for Dirichlet prior distribution.

## Optional arguments
- `s_prior::Vector{Real}=[0.0, 2.0]`: Parameters for the mean fitness prior
  distribution π(sₜ).
- `σ_prior::Vector{Real}=[0.0, 1.0]`: Parameters for the nuisance standard
  deviation parameter prior distribution π(σₜ).
- `σ_trunc::Real=0.0`: Value at which truncate the normal distribution to
  define it as a half-normal.
"""
Turing.@model function mean_fitness_neutrals_lognormal(
    r̲ₜ::Vector{Int64},
    r̲ₜ₊₁::Vector{Int64},
    α̲::Vector{Float64};
    s_prior::Vector{Real}=[0.0, 2.0],
    σ_prior::Vector{Real}=[0.0, 1.0],
    σ_trunc::Real=0.0
)
    # Prior on mean fitness sₜ
    sₜ ~ Turing.Normal(s_prior...)
    # Prior on LogNormal error σₜ
    σₜ ~ Turing.truncated(Turing.Normal(σ_prior...); lower=σ_trunc)

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
    mean_fitness_neutrals_lognormal(r̲ₜ, r̲ₜ₊₁, α̲, s_prior, σ_prior)

`Turing.jl` model to sample out of the posterior for a single population mean
fitness value `sₜ`, given the raw barcode counts. Note: this `method` allows the
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
- `r̲ₜ::Vector{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t`. NOTE: The last entry of the array must
  be the sum of all of the counts from mutant lineages.
- `r̲ₜ₊₁::Vector{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t + 1`. NOTE: The last entry of the array
  must be the sum of all of the counts from mutant lineages.
- `α̲::Vector{Float64}`: Parameters for Dirichlet prior distribution.
- `s_prior::Distributions.ContinuousUnivariateDistribution`: Parametrized
  univariate continuous distribution for the prior on the mean fitness π(sₜ).
- `σ_prior:::Distributions.ContinuousUnivariateDistribution`: Parametrized
  univariate continuous distribution for the prior on the nuisance standard
  deviation of the log-normal likelihood π(σₜ).
"""
Turing.@model function mean_fitness_neutrals_lognormal(
    r̲ₜ::Vector{Int64},
    r̲ₜ₊₁::Vector{Int64},
    α̲::Vector{Float64},
    s_prior::Distributions.ContinuousUnivariateDistribution,
    σ_prior::Distributions.ContinuousUnivariateDistribution;
)
    # Prior on mean fitness sₜ
    sₜ ~ s_prior
    # Prior on LogNormal error σₜ
    σₜ ~ σ_prior

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

@doc raw"""
    mutant_fitness_lognormal(r̲⁽ᵐ⁾, R̲, α̲, μ_sₜ, σ_sₜ; s_prior, σ_prior, σ_trunc)

`Turing.jl` model to sample out of the posterior distribution for a single
mutant fitness value `s⁽ᵐ⁾`, given the raw barcode counts and the
parametrization of the population mean fitness distribution.

# Arguments
- `r̲⁽ᵐ⁾::Vector{Int64}`: Mutant `m` raw barcode counts time-series. Note: this
  vector must be the same length as `r̲⁽ᶜ⁾`. This means that each entry
  `r̲⁽ᵐ⁾[i]` contains the number of reads from barcode `m` at time `i`.
- `R̲::Vector{Int64}`: time-series of Raw **total** reads. This means that entry
  `R̲[i]` contains the total number of reads obtained at time `i`.
- `α̲::Vector{Float64}`: Parameters for Beta prior distribution.
- `μ_sₜ::Vector{Float64}`: Array with the time-series mean values of the
  population mean fitness. This means entry `μ_sₜ[i]` contains the inferred mean
  value of the population mean fitness for time `i`, assuming `sₜ[i] ~
  Normal(μ_sₜ[i], σ_sₜ[i])`.
- `σ_sₜ::Vector{Float64}`: Array with the time-series values of the population
  mean fitness standard deviation. This means entry `σ_sₜ[i]` contains the
  inferred value of the standard deviation of the population mean fitness at
  time `i`, assuming `sₜ[i] ~ Normal(μ_sₜ[i], σ_sₜ[i])`.

## Optional arguments
- `s_prior::Vector{Real}=[0.0, 2.0]`: Parameters for the mutant fitness prior
  distribution π(s⁽ᵐ⁾).
- `σ_prior::Vector{Real}=[0.0, 1.0]`: Parameters for the nuisance standard
  deviation parameter prior distribution π(σ⁽ᵐ⁾).
- `σ_trunc::Real=0.0`: Value at which truncate the normal distribution to define
  it as a half-normal.
"""
Turing.@model function mutant_fitness_lognormal(
    r̲⁽ᵐ⁾::Vector{Int64},
    R̲::Vector{Int64},
    α̲::Vector{Float64},
    μ_sₜ::Vector{Float64},
    σ_sₜ::Vector{Float64};
    s_prior::Vector{Real}=[0.0, 2.0],
    σ_prior::Vector{Real}=[0.0, 1.0],
    σ_trunc::Real=0.0
)
    # Prior on mutant fitness s⁽ᵐ⁾
    s⁽ᵐ⁾ ~ Turing.Normal(s_prior...)
    # Prior on LogNormal error σ⁽ᵐ⁾ 
    σ⁽ᵐ⁾ ~ Turing.truncated(Turing.Normal(σ_prior...); lower=σ_trunc)

    # Population mean fitness values
    s̲ₜ ~ Turing.MvNormal(μ_sₜ, LinearAlgebra.Diagonal(σ_sₜ .^ 2))

    # Initialize array to store frequencies
    f̲⁽ᵐ⁾ = Vector{Float64}(undef, length(r̲⁽ᵐ⁾))

    # Frequency distribution for each time point
    for i in eachindex(r̲⁽ᵐ⁾)
        f̲⁽ᵐ⁾[i] ~ Turing.Beta(α̲[1] + r̲⁽ᵐ⁾[i], α̲[2] + (R̲[i] - r̲⁽ᵐ⁾[i]))
    end # for

    # Compute frequency ratios
    γ̲⁽ᵐ⁾ = f̲⁽ᵐ⁾[2:end] ./ f̲⁽ᵐ⁾[1:end-1]

    # Sample posterior for frequency ratio. Since it is a sample over a
    # generated quantity, we must use the @addlogprob! macro
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            s⁽ᵐ⁾ .- s̲ₜ,
            LinearAlgebra.I(length(s̲ₜ)) .* σ⁽ᵐ⁾^2
        ),
        γ̲⁽ᵐ⁾
    )
end # @model function