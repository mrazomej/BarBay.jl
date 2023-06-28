# Import basic math
import LinearAlgebra

# Import libraries to define distributions
import Distributions
import Random

# Import libraries relevant for MCMC
import Turing

# Functionality for constructing arrays with identical elements efficiently
import FillArrays
##
# Export functions

# Export mean fitness model
export mean_fitness_neutrals_lognormal
export mean_fitness_neutrals_lognormal_priors

# Export mutant fitness model
export mutant_fitness_lognormal, mutant_fitness_lognormal_priors

# Export joint fitness models
export fitness_lognormal

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Population Mean Fitness π(sₜ | Data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    mean_fitness_neutrals_lognormal(r̲ₜ, r̲ₜ₊₁; α, s_prior, σ_prior, σ_trunc)

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
- ``\underline{\alpha}_t``: `α`.
- ``[\mu_{\bar{s}_t}, \sigma_{\bar{s}_t}]``: `s_prior`.
- ``[\mu_{\sigma_t}, \sigma_{\sigma_t}]``: `σ_prior`.

# Arguments
- `r̲ₜ::Vector{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t`. NOTE: The last entry of the array must
  be the sum of all of the counts from mutant lineages.
- `r̲ₜ₊₁::Vector{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t + 1`. NOTE: The last entry of the array
  must be the sum of all of the counts from mutant lineages.

# Keyword Arguments
- `α::Vector{Float64}`: Parameters for Dirichlet prior distribution.

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
    r̲ₜ₊₁::Vector{Int64};
    α::Vector{Float64},
    s_prior::Vector{<:Real}=[0.0, 2.0],
    σ_prior::Vector{<:Real}=[0.0, 1.0],
    σ_trunc::Real=0.0
)
    # Prior on mean fitness sₜ
    sₜ ~ Turing.Normal(s_prior...)
    # Prior on LogNormal error σₜ
    σₜ ~ Turing.truncated(Turing.Normal(σ_prior...); lower=σ_trunc)

    # Frequency distribution from Multinomial-Dirichlet model
    f̲ₜ ~ Turing.Dirichlet(α .+ r̲ₜ)
    f̲ₜ₊₁ ~ Turing.Dirichlet(α .+ r̲ₜ₊₁)

    # Check that all distributions are greater than zero. Although the counts
    # could be zero, we assume that the real frequencies are non-zero always.
    if any(iszero.(f̲ₜ)) | any(iszero.(f̲ₜ₊₁))
        Turing.@addlogprob! -Inf
        # Exit the model evaluation early
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
    return
end # @model function

@doc raw"""
    mean_fitness_neutrals_lognormal_priors(r̲ₜ, r̲ₜ₊₁; α, s_prior, σ_prior)

`Turing.jl` model to sample out of the posterior for a single population mean
fitness value `sₜ`, given the raw barcode counts. Note: this function allows for
the definition of any prior distributions on the mean fitness and the nuisance
standard deviation parameter for the log-likelihood function.

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

For this inference, we enforce all frequencies to be > 0 (even for barcodes with
zero reads) to compute ``\gamma_t^{(n)}``.

The user defines the distribution parameters as:
- ``\underline{\alpha}_t``: `α`.

# Arguments
- `r̲ₜ::Vector{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t`. NOTE: The last entry of the array must
  be the sum of all of the counts from mutant lineages.
- `r̲ₜ₊₁::Vector{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t + 1`. NOTE: The last entry of the array
  must be the sum of all of the counts from mutant lineages.

# Keyword Arguments
- `α::Vector{Float64}`: Parameters for Dirichlet prior distribution.
- `s_prior::Distributions.ContinuousUnivariateDistribution`: Parametrized
  univariate continuous distribution for the prior on the mean fitness π(sₜ).
- `σ_prior:::Distributions.ContinuousUnivariateDistribution`: Parametrized
  univariate continuous distribution for the prior on the nuisance standard
  deviation of the log-normal likelihood π(σₜ).
"""
Turing.@model function mean_fitness_neutrals_lognormal_priors(
    r̲ₜ::Vector{Int64},
    r̲ₜ₊₁::Vector{Int64};
    α::Vector{Float64},
    s_prior::Distributions.ContinuousUnivariateDistribution,
    σ_prior::Distributions.ContinuousUnivariateDistribution
)
    # Prior on mean fitness sₜ
    sₜ ~ s_prior
    # Prior on LogNormal error σₜ
    σₜ ~ σ_prior

    # Frequency distribution from Multinomial-Dirichlet model
    f̲ₜ ~ Turing.Dirichlet(α .+ r̲ₜ)
    f̲ₜ₊₁ ~ Turing.Dirichlet(α .+ r̲ₜ₊₁)

    # Check that all distributions are greater than zero. Although the counts
    # could be zero, we assume that the real frequencies are non-zero always.
    if any(iszero.(f̲ₜ)) | any(iszero.(f̲ₜ₊₁))
        Turing.@addlogprob! -Inf
        # Exit the model evaluation early
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
    return
end # @model function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Mutant fitness π(s⁽ᵐ⁾ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    mutant_fitness_lognormal(r̲⁽ᵐ⁾, R̲; α, μ_sₜ, σ_sₜ, s_prior, σ_prior, σ_trunc)

`Turing.jl` model to sample out of the posterior distribution for a single
mutant fitness value `s⁽ᵐ⁾`, given the raw barcode counts and the
parametrization of the population mean fitness distribution.

# Arguments
- `r̲⁽ᵐ⁾::Vector{Int64}`: Mutant `m` raw barcode counts time-series. Note: this
  vector must be the same length as `r̲⁽ᶜ⁾`. This means that each entry
  `r̲⁽ᵐ⁾[i]` contains the number of reads from barcode `m` at time `i`.
- `R̲::Vector{Int64}`: time-series of Raw **total** reads. This means that entry
  `R̲[i]` contains the total number of reads obtained at time `i`.

# Keyword Arguments
- `α::Vector{Float64}`: Parameters for Beta prior distribution.
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
    R̲::Vector{Int64};
    α::Vector{Float64},
    μ_s̄::Vector{Float64},
    σ_s̄::Vector{Float64},
    s_prior::Vector{<:Real}=[0.0, 2.0],
    σ_prior::Vector{<:Real}=[0.0, 1.0],
    σ_trunc::Real=0.0
)
    # Prior on mutant fitness s⁽ᵐ⁾
    s⁽ᵐ⁾ ~ Turing.Normal(s_prior...)
    # Prior on LogNormal error σ⁽ᵐ⁾ 
    σ⁽ᵐ⁾ ~ Turing.truncated(Turing.Normal(σ_prior...); lower=σ_trunc)

    # Population mean fitness values
    s̲ₜ ~ Turing.MvNormal(μ_s̄, LinearAlgebra.Diagonal(σ_s̄ .^ 2))

    # Initialize array to store frequencies
    f̲⁽ᵐ⁾ = Vector{Float64}(undef, length(r̲⁽ᵐ⁾))

    # Frequency distribution for each time point
    for i in eachindex(r̲⁽ᵐ⁾)
        f̲⁽ᵐ⁾[i] ~ Turing.Beta(α[1] + r̲⁽ᵐ⁾[i], α[2] + (R̲[i] - r̲⁽ᵐ⁾[i]))
    end # for

    # Check that all distributions are greater than zero. Although the counts
    # could be zero, we assume that the real frequencies are non-zero always.
    if any(iszero.(f̲⁽ᵐ⁾))
        Turing.@addlogprob! -Inf
        # Exit the model evaluation early
        return
    end

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
    return
end # @model function

@doc raw"""
    mutant_fitness_lognormal_priors(r̲⁽ᵐ⁾, R̲; α, s_mean_priors, s_prior, σ_prior, σ_trunc)

`Turing.jl` model to sample out of the posterior distribution for a single
mutant fitness value `s⁽ᵐ⁾`, given the raw barcode counts and the
parametrization of the population mean fitness distribution. Note: this function
allows for the definition of any prior distributions on the population mean
fitness, the nuisance standard deviation parameter for the log-likelihood
function, and the mutant mean fitness.

# Arguments
- `r̲⁽ᵐ⁾::Vector{Int64}`: Mutant `m` raw barcode counts time-series. Note: this
  vector must be the same length as `r̲⁽ᶜ⁾`. This means that each entry
  `r̲⁽ᵐ⁾[i]` contains the number of reads from barcode `m` at time `i`.
- `R̲::Vector{Int64}`: time-series of Raw **total** reads. This means that entry
  `R̲[i]` contains the total number of reads obtained at time `i`.

# Keyword Arguments
- `α::Vector{Float64}`: Parameters for Beta prior distribution.
- `s_mean_priors::Vector{<:Distributions.ContinuousUnivariateDistribution}`:
  Vector of univariate distributions defining the prior distribution for each
  population mean fitness value.
- `s_prior::Distributions.ContinuousUnivariateDistribution`: Parametrized
  univariate continuous distribution for the prior on the mean fitness π(sₜ).
- `σ_prior:::Distributions.ContinuousUnivariateDistribution`: Parametrized
  univariate continuous distribution for the prior on the nuisance standard
  deviation of the log-normal likelihood π(σₜ).
"""
Turing.@model function mutant_fitness_lognormal_priors(
    r̲⁽ᵐ⁾::Vector{Int64},
    R̲::Vector{Int64};
    α::Vector{Float64},
    s_mean_priors::Vector{<:Distributions.ContinuousUnivariateDistribution},
    s_prior::Distributions.ContinuousUnivariateDistribution,
    σ_prior::Distributions.ContinuousUnivariateDistribution
)
    # Prior on mutant fitness s⁽ᵐ⁾
    s⁽ᵐ⁾ ~ s_prior
    # Prior on LogNormal error σ⁽ᵐ⁾ 
    σ⁽ᵐ⁾ ~ σ_prior

    # Initialize array to save mean fitness priors
    s̲ₜ = Vector{Float64}(undef, length(s_mean_prior))

    # Loop through population mean fitness priors
    for i in eachindex(s_mean_priors)
        # Sample population mean fitness prior
        s̲ₜ[i] ~ s_mean_priors[i]
    end # for

    # Initialize array to store frequencies
    f̲⁽ᵐ⁾ = Vector{Float64}(undef, length(r̲⁽ᵐ⁾))

    # Frequency distribution for each time point
    for i in eachindex(r̲⁽ᵐ⁾)
        f̲⁽ᵐ⁾[i] ~ Turing.Beta(α[1] + r̲⁽ᵐ⁾[i], α[2] + (R̲[i] - r̲⁽ᵐ⁾[i]))
    end # for

    # Check that all distributions are greater than zero. Although the counts
    # could be zero, we assume that the real frequencies are non-zero always.
    if any(iszero.(f̲⁽ᵐ⁾))
        Turing.@addlogprob! -Inf
        # Exit the model evaluation early
        return
    end

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
    return
end # @model function


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Mutant fitness in fluctuating environemnts π(s1⁽ᵐ⁾, s2⁽ᵐ⁾,.. | data)
# two-environment fluctuations
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    env_mutant_fitness_lognormal(r̲⁽ᵐ⁾, R̲; α, μ_sₜ, σ_sₜ, s_prior, σ_prior, σ_trunc)

`Turing.jl` model to sample out of the posterior distribution for a single
mutant fitness value `s⁽ᵐ⁾`, given the raw barcode counts and the
parametrization of the population mean fitness distribution.

# Arguments
- `r̲⁽ᵐ⁾::Vector{Int64}`: Mutant `m` raw barcode counts time-series. Note: this
  vector must be the same length as `r̲⁽ᶜ⁾`. This means that each entry
  `r̲⁽ᵐ⁾[i]` contains the number of reads from barcode `m` at time `i`.
- `R̲::Vector{Int64}`: time-series of Raw **total** reads. This means that entry
  `R̲[i]` contains the total number of reads obtained at time `i`.

# Keyword arguments
- `envs::Vector{<:Any}`: Vector defining the order of environments. Environments
  can be labeled with numbers (e.g. [1, 2, 2, 3, 1, 3]), strings (e.g. ["env1",
  "env2", "env1"]), or any convenient label. The point being that they should
  follow the order of environments to which strains were exposed during the
  experiment. In the output chain, the order of the inferred fitness will be the
  order obtained from running `unique(envs)`.
- `α::Vector{Float64}`: Parameters for Beta prior distribution.
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
Turing.@model function env_mutant_fitness_lognormal(
    r̲⁽ᵐ⁾::Vector{Int64},
    R̲::Vector{Int64};
    envs::Vector,
    α::Vector{Float64},
    μ_s̄::Vector{Float64},
    σ_s̄::Vector{Float64},
    s_prior::Vector{<:Real}=[0.0, 2.0],
    σ_prior::Vector{<:Real}=[0.0, 1.0],
    σ_trunc::Real=0.0
)
    # Find unique environments
    env_unique = unique(envs)

    # Define environmental indexes
    env_idx = indexin(envs, env_unique)

    # Prior on mutant fitness s⁽ᵐ⁾
    s⁽ᵐ⁾ ~ Turing.filldist(Turing.Normal(s_prior...), length(env_unique))

    # Prior on LogNormal error σ⁽ᵐ⁾ 
    σ⁽ᵐ⁾ ~ Turing.filldist(
        Turing.truncated(
            Turing.Normal(σ_prior...); lower=σ_trunc
        ),
        length(env_unique)
    )

    # Population mean fitness values
    s̲ₜ ~ Turing.MvNormal(μ_s̄, LinearAlgebra.Diagonal(σ_s̄ .^ 2))

    # Initialize array to store frequencies
    f̲⁽ᵐ⁾ = Vector{Float64}(undef, length(r̲⁽ᵐ⁾))

    # Frequency distribution for each time point
    for i in eachindex(r̲⁽ᵐ⁾)
        f̲⁽ᵐ⁾[i] ~ Turing.Beta(α[1] + r̲⁽ᵐ⁾[i], α[2] + (R̲[i] - r̲⁽ᵐ⁾[i]))
    end # for

    # Check that all distributions are greater than zero. Although the counts
    # could be zero, we assume that the real frequencies are non-zero always.
    if any(iszero.(f̲⁽ᵐ⁾))
        Turing.@addlogprob! -Inf
        # Exit the model evaluation early
        return
    end

    # Compute frequency ratios
    γ̲⁽ᵐ⁾ = f̲⁽ᵐ⁾[2:end] ./ f̲⁽ᵐ⁾[1:end-1]

    # Sample posterior for frequency ratio. Since it is a sample over a
    # generated quantity, we must use the @addlogprob! macro
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            s⁽ᵐ⁾[env_idx] .- s̲ₜ,
            LinearAlgebra.Diagonal(σ⁽ᵐ⁾[env_idx] .^ 2)
        ),
        γ̲⁽ᵐ⁾
    )
    return
end # @model function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Joint inference π(s̲⁽ᵐ⁾, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    fitness_lognormal(R̲̲, R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, n̲ₜ; s_pop_prior, σ_pop_prior, s_mut_prior, σ_mut_prior, λ_prior)

`Turing.jl` model to sample the joint posterior distribution for a competitive
fitness experiment.

# Model
`[write model here]`

# Arguments
- `R̲̲⁽ⁿ⁾::Matrix{Int64}`: `T × N` matrix where `T` is the number of time points
  in the data set and `N` is the number of neutral lineage barcodes. Each column
  represents the barcode count trajectory for a single neutral lineage.
  **NOTE**: The model assumes the rows are sorted in order of increasing time.
- `R̲̲⁽ᵐ⁾::Matrix{Int64}`: `T × M` matrix where `T` is the number of time points
  in the data set and `M` is the number of mutant lineage barcodes. Each column
  represents the barcode count trajectory for a single mutant lineage. **NOTE**:
  The model assumes the rows are sorted in order of increasing time.
- `R̲̲::Matrix{Int64}`:: `T × B` matrix, where `T` is the number of time points
  in the data set and `B` is the number of barcodes. Each column represents the
  barcode count trajectory for a single lineage. **NOTE**: This matrix does not
  necessarily need to be equivalent to `hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾)`. This is because
  `R̲̲⁽ᵐ⁾` can exclude mutant barcodes to perform the joint inference only for a
  subgroup, but `R̲̲` must still contain all counts. Usually, if `R̲̲⁽ᵐ⁾`
  excludes mutant barcodes, `R̲̲` must be of the form `hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾,
  R̲̲⁽ᴹ⁾)`, where `R̲̲⁽ᴹ⁾` is a vector that aggregates all excluded mutant barcodes
  into a "super barcode."
- `n̲ₜ::Vector{Int64}`: Vector with the total number of barcode counts for each
  time point. **NOTE**: This vector **must** be equivalent to computing
  `vec(sum(R̲̲, dims=2))`. The reason it is an independent input parameter is to
  avoid the `sum` computation within the `Turing` model.

## Optional Keyword Arguments
- `s_pop_prior::Vector{Float64}=[0.0, 2.0]`: Vector with the correspnding
    parameters (`s_pop_prior[1]` = mean, `s_pop_prior[2]` = standard deviation)
    for a Normal prior on the population mean fitness values. **NOTE**: This
    method assigns the same prior to **all** population mean fitness to be
    inferred.
- `σ_pop_prior::Vector{Float64}=[0.0, 1.0]`: Vector with the correspnding
    parameters (`σ_pop_prior[1]` = mean, `σ_pop_prior[2]` = standard deviation)
    for a Log-Normal prior on the population mean fitness error utilized in the
    log-likelihood function. **NOTE**: This method assigns the same prior to
    **all** population mean fitness errors to be inferred.
- `s_mut_prior::Vector{Float64}=[0.0, 2.0]`: Vector with the correspnding
    parameters (`s_mut_prior[1]` = mean, `s_mut_prior[2]` = standard deviation)
    for a Normal prior on the mutant fitness values. **NOTE**: This method
    assigns the same prior to **all** mutant fitness values to be inferred.
- `σ_mut_prior::Vector{Float64}=[0.0, 1.0]`: Vector with the correspnding
    parameters (`σ_mut_prior[1]` = mean, `σ_mut_prior[2]` = standard deviation)
    for a Log-Normal prior on the mutant fitness error utilized in the
    log-likelihood function. **NOTE**: This method assigns the same prior to
    **all** mutant fitness error values to be inferred.
- `λ_prior::Vector{Float64}=[3.0, 3.0]`: Vector with the corresponding
  parameters (`λ_prior[1]` = mean, `λ_prior[2]` = standard deviation) for a
  Log-Normal prior on the λ parameter in the Poisson distribution. The λ
  parameter can be interpreted as the mean number of barcode counts since we
  assume any barcode count `n⁽ᵇ⁾ ~ Poisson(λ⁽ᵇ⁾)`. **NOTE**: This method assigns
    the same prior to **all** mutant fitness error values to be inferred.
"""
Turing.@model function fitness_lognormal(
    R̲̲⁽ⁿ⁾::Matrix{Int64},
    R̲̲⁽ᵐ⁾::Matrix{Int64},
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64};
    s_pop_prior::Vector{Float64}=[0.0, 2.0],
    σ_pop_prior::Vector{Float64}=[0.0, 1.0],
    s_mut_prior::Vector{Float64}=[0.0, 2.0],
    σ_mut_prior::Vector{Float64}=[0.0, 1.0],
    λ_prior::VecOrMat{Float64}=[3.0, 3.0]
)
    ## %%%%%%%%%%%%%% Population mean fitness  %%%%%%%%%%%%%% ##

    # Prior on population mean fitness π(s̲ₜ) 
    s̲ₜ ~ Turing.MvNormal(
        repeat([s_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* s_pop_prior[2] .^ 2
    )
    # Prior on LogNormal error π(σ̲ₜ)
    σ̲ₜ ~ Turing.MvLogNormal(
        repeat([σ_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* σ_pop_prior[2] .^ 2
    )

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Prior on mutant fitness π(s̲⁽ᵐ⁾)
    s̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([s_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* s_mut_prior[2] .^ 2
    )
    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
        repeat([σ_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* σ_mut_prior[2] .^ 2
    )


    ## %%%%%%%%%%%%%% Barcode frequencies %%%%%%%%%%%%%% ##

    if typeof(λ_prior) <: Vector
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            repeat([λ_prior[1]], length(R̲̲)),
            LinearAlgebra.I(length(R̲̲)) .* λ_prior[2]^2
        )
    elseif typeof(λ_prior) <: Matrix
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            λ_prior[:, 1], LinearAlgebra.Diagonal(λ_prior[:, 2] .^ 2)
        )
    end  # if

    # Reshape λ parameters to fit the matrix format. Note: The Λ̲̲ array is
    # originally sampled as a vector for the `Turing.jl` samplers to deal with
    # it. But reshaping it to a matrix simplifies the computation of frequencies
    # and frequency ratios.
    Λ̲̲ = reshape(Λ̲̲, size(R̲̲)...)

    # Compute barcode frequencies from Poisson parameters
    F̲̲ = Λ̲̲ ./ sum(Λ̲̲, dims=2)

    # Compute frequency ratios between consecutive time points.
    Γ̲̲ = F̲̲[2:end, :] ./ F̲̲[1:end-1, :]

    # Split neutral and mutant frequency ratios. Note: the @view macro means
    # that there is not allocation to memory on this step.
    Γ̲̲⁽ⁿ⁾ = @view Γ̲̲[:, 1:size(R̲̲⁽ⁿ⁾, 2)]
    Γ̲̲⁽ᵐ⁾ = @view Γ̲̲[:, size(R̲̲⁽ⁿ⁾, 2)+1:size(R̲̲⁽ⁿ⁾, 2)+size(R̲̲⁽ᵐ⁾, 2)]

    # Prob of total number of barcodes read given the Poisosn distribution
    # parameters π(nₜ | λ̲ₜ)
    n̲ₜ ~ Turing.arraydist([Turing.Poisson(sum(Λ̲̲[t, :])) for t = 1:size(R̲̲⁽ⁿ⁾, 1)])

    # Loop through time points
    for t = 1:size(R̲̲⁽ⁿ⁾, 1)
        # Prob of reads given parameters π(R̲ₜ | nₜ, f̲ₜ). Note: We add the
        # check_args=false option to avoid the recurrent problem of
        # > Multinomial: p is not a probability vector.
        # due to rounding errors
        R̲̲[t, :] ~ Turing.Multinomial(n̲ₜ[t], F̲̲[t, :]; check_args=false)
    end # for

    ## %%%%%%%%%%%%%% Log-Likelihood functions %%%%%%%%%%%%%% ##

    # Sample posterior for neutral lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ⁿ⁾| sₜ, σₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            repeat(-s̲ₜ, size(Γ̲̲⁽ⁿ⁾, 2)),
            LinearAlgebra.Diagonal(repeat(σ̲ₜ .^ 2, size(Γ̲̲⁽ⁿ⁾, 2)))
        ),
        Γ̲̲⁽ⁿ⁾[:]
    )

    # Sample posterior for nutant lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ᵐ⁾ | s⁽ᵐ⁾, σ⁽ᵐ⁾, s̲ₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            # Build vector for fitness differences
            vcat([s⁽ᵐ⁾ .- s̲ₜ for s⁽ᵐ⁾ in s̲⁽ᵐ⁾]...),
            # Build vector for variances
            LinearAlgebra.Diagonal(
                vcat([repeat([σ], length(s̲ₜ)) for σ in σ̲⁽ᵐ⁾]...) .^ 2
            )
        ),
        Γ̲̲⁽ᵐ⁾[:]
    )
    return
end # @model function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

Turing.@model function fitness_lognormal(
    R̲̲⁽ⁿ⁾::Array{Int64,3},
    R̲̲⁽ᵐ⁾::Array{Int64,3},
    R̲̲::Array{Int64,3},
    n̲ₜ::Matrix{Int64};
    s_pop_prior::Vector{Float64}=[0.0, 2.0],
    σ_pop_prior::Vector{Float64}=[0.0, 1.0],
    s_mut_prior::Vector{Float64}=[0.0, 2.0],
    σ_mut_prior::Vector{Float64}=[0.0, 1.0],
    λ_prior::Vector{Float64}=[3.0, 3.0]
)
    ## %%%%%%%%%%%%%% Population mean fitness  %%%%%%%%%%%%%% ##

    # Prior on population mean fitness π(s̲ₜ) 
    s̲ₜ ~ Turing.arraydist(
        [
            Turing.MvNormal(
                repeat([s_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
                LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* s_pop_prior[2] .^ 2
            )
        ] for rep in size(R̲̲⁽ⁿ⁾, 3)
    )
    # Prior on LogNormal error π(σ̲ₜ)
    σ̲ₜ ~ Turing.arraydist(
        [
            Turing.MvLogNormal(
                repeat([σ_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
                LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* σ_pop_prior[2] .^ 2
            )
        ] for rep in size(R̲̲⁽ⁿ⁾, 3)
    )

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Hyper prior on mutant fitness π(θ̲⁽ᵐ⁾) 
    θ̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([s_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* s_mut_prior[2] .^ 2
    )

    # Hyper prior on mutant variance π(ξ̲⁽ᵐ⁾) 
    ξ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
        repeat([σ_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* σ_mut_prior[2] .^ 2
    )

    # Prior on mutant fitness π(s̲⁽ᵐ⁾ | θ̲⁽ᵐ⁾)
    s̲⁽ᵐ⁾ ~ Turing.arraydist(
        [
            Turing.MvNormal(θ̲⁽ᵐ⁾, LinearAlgebra.Diagonal(ξ̲⁽ᵐ⁾ .^ 2))
        ] for rep in size(R̲̲⁽ᵐ⁾, 3)
    )
    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    σ̲⁽ᵐ⁾ ~ Turing.arraydist(
        [
            Turing.MvLogNormal(
                repeat([σ_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
                LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* σ_mut_prior[2] .^ 2
            )
        ] for rep in size(R̲̲⁽ᵐ⁾, 3)
    )


    ## %%%%%%%%%%%%%% Barcode frequencies %%%%%%%%%%%%%% ##

    # Prior on Poisson distribtion parameters π(λ)
    Λ̲̲ ~ Turing.MvLogNormal(
        repeat([λ_prior[1]], length(R̲̲)),
        LinearAlgebra.I(length(R̲̲)) .* λ_prior[2]^2
    )

    # Reshape λ parameters to fit the matrix format. Note: The Λ̲̲ array is
    # originally sampled as a vector for the `Turing.jl` samplers to deal with
    # it. But reshaping it to a matrix simplifies the computation of frequencies
    # and frequency ratios.
    Λ̲̲ = reshape(Λ̲̲, size(R̲̲)...)

    # Compute barcode frequencies from Poisson parameters
    F̲̲ = Λ̲̲ ./ sum(Λ̲̲, dims=2)

    # Compute frequency ratios between consecutive time points.
    Γ̲̲ = F̲̲[2:end, :, :] ./ F̲̲[1:end-1, :, :]

    # Split neutral and mutant frequency ratios. Note: the @view macro means
    # that there is not allocation to memory on this step.
    Γ̲̲⁽ⁿ⁾ = @view Γ̲̲[:, 1:size(R̲̲⁽ⁿ⁾, 2), :]
    Γ̲̲⁽ᵐ⁾ = @view Γ̲̲[:, size(R̲̲⁽ⁿ⁾, 2)+1:size(R̲̲⁽ⁿ⁾, 2)+size(R̲̲⁽ᵐ⁾, 2), :]

    # Loop through replicates
    for r = 1:size(R̲̲⁽ⁿ⁾, 3)
        # Prob of total number of barcodes read given the Poisosn distribution
        # parameters π(nₜ | λ̲ₜ)
        n̲ₜ[:, r] ~ Turing.arraydist(
            [Turing.Poisson(sum(Λ̲̲[t, :, r])) for t = 1:size(R̲̲⁽ⁿ⁾, 1)]
        )

        # Loop through time points
        for t = 1:size(R̲̲⁽ⁿ⁾, 1)
            # Prob of reads given parameters π(R̲ₜ | nₜ, f̲ₜ). Note: We add the
            # check_args=false option to avoid the recurrent problem of
            # > Multinomial: p is not a probability vector.
            # due to rounding errors
            R̲̲[t, :, r] ~ Turing.Multinomial(
                n̲ₜ[t, r], F̲̲[t, :, r]; check_args=false
            )
        end # for
    end # for

    ## %%%%%%%%%%%%%% Log-Likelihood functions %%%%%%%%%%%%%% ##

    # Loop through replicates
    for r = 1:size(R̲̲⁽ⁿ⁾, 3)
        # Sample posterior for neutral lineage frequency ratio. Since it is a sample
        # over a generated quantity, we must use the @addlogprob! macro
        # π(γₜ⁽ⁿ⁾| sₜ, σₜ)
        Turing.@addlogprob! Turing.logpdf(
            Turing.MvLogNormal(
                repeat(-s̲ₜ[r], size(Γ̲̲⁽ⁿ⁾, 2)),
                LinearAlgebra.Diagonal(repeat(σ̲ₜ .^ 2, size(Γ̲̲⁽ⁿ⁾, 2)))
            ),
            Γ̲̲⁽ⁿ⁾[:, :, r][:]
        )

        # Sample posterior for nutant lineage frequency ratio. Since it is a sample
        # over a generated quantity, we must use the @addlogprob! macro
        # π(γₜ⁽ᵐ⁾ | s⁽ᵐ⁾, σ⁽ᵐ⁾, s̲ₜ)
        Turing.@addlogprob! Turing.logpdf(
            Turing.MvLogNormal(
                # Build vector for fitness differences
                vcat([s⁽ᵐ⁾ .- s̲ₜ[r] for s⁽ᵐ⁾ in s̲⁽ᵐ⁾[r]]...),
                # Build vector for variances
                LinearAlgebra.Diagonal(
                    vcat([repeat([σ], length(s̲ₜ)) for σ in σ̲⁽ᵐ⁾[r]]...) .^ 2
                )
            ),
            Γ̲̲⁽ᵐ⁾[:, :, r][:]
        )
    end # for

    return
end # @model function


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    mean_fitness_lognormal(R̲̲, R̲̲⁽ⁿ⁾, n̲ₜ; s_pop_prior, σ_pop_prior, λ_prior)

`Turing.jl` model to sample the joint posterior distribution of the population
mean fitness for a competitive fitness experiment using only the neutral
lineages.

# Model
`[write model here]`

# Arguments
- `R̲̲⁽ⁿ⁾::Matrix{Int64}`: `T × N` matrix where `T` is the number of time points
  in the data set and `N` is the number of neutral lineage barcodes. Each column
  represents the barcode count trajectory for a single neutral lineage.
  **NOTE**: The model assumes the rows are sorted in order of increasing time.
- `R̲̲::Matrix{Int64}`:: `T × B` matrix, where `T` is the number of time points
  in the data set and `B` is the number of barcodes. Each column represents the
  barcode count trajectory for a single lineage.
- `n̲ₜ::Vector{Int64}`: Vector with the total number of barcode counts for each
  time point. **NOTE**: This vector **must** be equivalent to computing
  `vec(sum(R̲̲, dims=2))`. The reason it is an independent input parameter is to
  avoid the `sum` computation within the `Turing` model.

## Optional Keyword Arguments
- `s_pop_prior::Vector{Float64}=[0.0, 2.0]`: Vector with the correspnding
    parameters (`s_pop_prior[1]` = mean, `s_pop_prior[2]` = standard deviation)
    for a Normal prior on the population mean fitness values. **NOTE**: This
    method assigns the same prior to **all** population mean fitness to be
    inferred.
- `σ_pop_prior::Vector{Float64}=[0.0, 1.0]`: Vector with the correspnding
    parameters (`σ_pop_prior[1]` = mean, `σ_pop_prior[2]` = standard deviation)
    for a Log-Normal prior on the population mean fitness error utilized in the
    log-likelihood function. **NOTE**: This method assigns the same prior to
    **all** population mean fitness errors to be inferred.
- `λ_prior::Vector{Float64}=[3.0, 3.0]`: Vector with the corresponding
  parameters (`λ_prior[1]` = mean, `λ_prior[2]` = standard deviation) for a
  Log-Normal prior on the λ parameter in the Poisson distribution. The λ
  parameter can be interpreted as the mean number of barcode counts since we
  assume any barcode count `n⁽ᵇ⁾ ~ Poisson(λ⁽ᵇ⁾)`. **NOTE**: This method assigns
    the same prior to **all** mutant fitness error values to be inferred.
"""
Turing.@model function mean_fitness_lognormal(
    R̲̲⁽ⁿ⁾::Matrix{Int64},
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64};
    s_pop_prior::Vector{Float64}=[0.0, 2.0],
    σ_pop_prior::Vector{Float64}=[0.0, 1.0],
    λ_prior::VecOrMat{Float64}=[3.0, 3.0]
)
    ## %%%%%%%%%%%%%% Population mean fitness  %%%%%%%%%%%%%% ##

    # Prior on population mean fitness π(s̲ₜ) 
    s̲ₜ ~ Turing.MvNormal(
        repeat([s_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* s_pop_prior[2] .^ 2
    )
    # Prior on LogNormal error π(σ̲ₜ)
    σ̲ₜ ~ Turing.MvLogNormal(
        repeat([σ_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* σ_pop_prior[2] .^ 2
    )

    ## %%%%%%%%%%%%%% Barcode frequencies %%%%%%%%%%%%%% ##

    if typeof(λ_prior) <: Vector
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            repeat([λ_prior[1]], length(R̲̲)),
            LinearAlgebra.I(length(R̲̲)) .* λ_prior[2]^2
        )
    elseif typeof(λ_prior) <: Matrix
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            λ_prior[:, 1], LinearAlgebra.Diagonal(λ_prior[:, 2] .^ 2)
        )
    end  # if

    # Reshape λ parameters to fit the matrix format. Note: The Λ̲̲ array is
    # originally sampled as a vector for the `Turing.jl` samplers to deal with
    # it. But reshaping it to a matrix simplifies the computation of frequencies
    # and frequency ratios.
    Λ̲̲ = reshape(Λ̲̲, size(R̲̲)...)

    # Compute barcode frequencies from Poisson parameters
    F̲̲ = Λ̲̲ ./ sum(Λ̲̲, dims=2)

    # Compute frequency ratios between consecutive time points.
    Γ̲̲ = F̲̲[2:end, :] ./ F̲̲[1:end-1, :]

    # Split neutral and mutant frequency ratios. Note: the @view macro means
    # that there is not allocation to memory on this step.
    Γ̲̲⁽ⁿ⁾ = @view Γ̲̲[:, 1:size(R̲̲⁽ⁿ⁾, 2)]

    # Prob of total number of barcodes read given the Poisosn distribution
    # parameters π(nₜ | λ̲ₜ)
    n̲ₜ ~ Turing.arraydist([Turing.Poisson(sum(Λ̲̲[t, :])) for t = 1:size(R̲̲⁽ⁿ⁾, 1)])

    # Loop through time points
    for t = 1:size(R̲̲⁽ⁿ⁾, 1)
        # Prob of reads given parameters π(R̲ₜ | nₜ, f̲ₜ). Note: We add the
        # check_args=false option to avoid the recurrent problem of
        # > Multinomial: p is not a probability vector.
        # due to rounding errors
        R̲̲[t, :] ~ Turing.Multinomial(n̲ₜ[t], F̲̲[t, :]; check_args=false)
    end # for

    ## %%%%%%%%%%%%%% Log-Likelihood functions %%%%%%%%%%%%%% ##

    # Sample posterior for neutral lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ⁿ⁾| sₜ, σₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            repeat(-s̲ₜ, size(Γ̲̲⁽ⁿ⁾, 2)),
            LinearAlgebra.Diagonal(repeat(σ̲ₜ .^ 2, size(Γ̲̲⁽ⁿ⁾, 2)))
        ),
        Γ̲̲⁽ⁿ⁾[:]
    )
    return
end # @model function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

Turing.@model function singlemutant_fitness_lognormal(
    r̲⁽ᵐ⁾::Vector{Int64},
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64};
    s_pop_prior_mean::Vector{Float64},
    s_pop_prior_std::Vector{Float64},
    s_mut_prior::Vector{<:Real}=[0.0, 2.0],
    σ_mut_prior::Vector{<:Real}=[0.0, 1.0],
    λ_prior::VecOrMat{Float64}=[3.0, 3.0]
)
    ## %%%%%%%%%%%%%% Population mean fitness  %%%%%%%%%%%%%% ##

    # Sample population mean fitness values
    s̲ₜ = Random.rand(
        Turing.MvNormal(
            s_pop_prior_mean, LinearAlgebra.Diagonal(s_pop_prior_std .^ 2)
        )
    )

    # Add "immutable prior" log probability
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvNormal(
            s_pop_prior_mean, LinearAlgebra.Diagonal(s_pop_prior_std .^ 2)
        ),
        s̲ₜ
    )
    # s̲ₜ ~ Turing.MvNormal(
    #     s_pop_prior_mean, LinearAlgebra.Diagonal(s_pop_prior_std .^ 2)
    # )

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Prior on mutant fitness π(s⁽ᵐ⁾)
    s⁽ᵐ⁾ ~ Turing.Normal(s_mut_prior[1], s_mut_prior[2])
    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    σ⁽ᵐ⁾ ~ Turing.LogNormal(σ_mut_prior[1], σ_mut_prior[2])

    ## %%%%%%%%%%%%%% Barcode frequencies %%%%%%%%%%%%%% ##

    if typeof(λ_prior) <: Vector
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            repeat([λ_prior[1]], length(R̲̲)),
            LinearAlgebra.I(length(R̲̲)) .* λ_prior[2]^2
        )
    elseif typeof(λ_prior) <: Matrix
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            λ_prior[:, 1], LinearAlgebra.Diagonal(λ_prior[:, 2] .^ 2)
        )
    end  # if

    # Reshape λ parameters to fit the matrix format. Note: The Λ̲̲ array is
    # originally sampled as a vector for the `Turing.jl` samplers to deal with
    # it. But reshaping it to a matrix simplifies the computation of frequencies
    # and frequency ratios.
    Λ̲̲ = reshape(Λ̲̲, size(R̲̲)...)

    # Compute barcode frequencies from Poisson parameters
    F̲̲ = Λ̲̲ ./ sum(Λ̲̲, dims=2)

    # Compute frequency ratios between consecutive time points.
    Γ̲̲ = F̲̲[2:end, :] ./ F̲̲[1:end-1, :]

    # Extract mutant frequency ratios. Note: the @view macro means
    # that there is not allocation to memory on this step.
    γ̲⁽ᵐ⁾ = @view Γ̲̲[:, 1]

    # Prob of total number of barcodes read given the Poisosn distribution
    # parameters π(nₜ | λ̲ₜ)
    n̲ₜ ~ Turing.arraydist(
        [Turing.Poisson(sum(Λ̲̲[t, :])) for t in eachindex(r̲⁽ᵐ⁾)]
    )

    # Loop through time points
    for t in eachindex(r̲⁽ᵐ⁾)
        # Prob of reads given parameters π(R̲ₜ | nₜ, f̲ₜ). Note: We add the
        # check_args=false option to avoid the recurrent problem of
        # > Multinomial: p is not a probability vector.
        # due to rounding errors
        R̲̲[t, :] ~ Turing.Multinomial(n̲ₜ[t], F̲̲[t, :]; check_args=false)
    end # for

    # Sample posterior for frequency ratio. Since it is a sample over a
    # generated quantity, we must use the @addlogprob! macro
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            s⁽ᵐ⁾ .- s̲ₜ,
            LinearAlgebra.I(length(s̲ₜ)) .* σ⁽ᵐ⁾^2
        ),
        γ̲⁽ᵐ⁾[:]
    )
    return
end # @model function