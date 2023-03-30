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

`Turing.jl` model to sample out of the posterior for a single population mean
fitness value sₜ, given the raw barcode counts.

The sampled probability distribution is of the form

π(sₜ, σₜ | f̲ₜ, f̲ₜ₊₁) π(f̲ₜ, f̲ₜ₊₁ | r̲ₜ, r̲ₜ₊₁) ∝ π(f̲ₜ, f̲ₜ₊₁ | sₜ, σₜ) π(sₜ) π(σₜ) π(f̲ₜ | r̲ₜ) π(f̲ₜ₊₁ |, r̲ₜ₊₁),

where

f̲ₜ / f̲ₜ₊₁ | sₜ, σₜ ~ LogNormal(-sₜ, σₜ),

sₜ ~ Normal(sₜ_prior...),

σₜ ~ Normal(σₜ_prior...),

f̲ₜ | r̲ₜ ~ Dirichlet(α̲ .+ r̲ₜ),

and

f̲ₜ₊₁ | r̲ₜ₊₁ ~ Dirichlet(α̲ .+ r̲ₜ₊₁).


# Arguments
- `r̲ₜ::Array{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t`. NOTE: The last entry of the array must
  be the sum of all of the counts from mutant lineages.
- `r̲ₜ₊₁::Array{Int64}`: Raw counts for **neutral** lineages and the cumulative
  counts for mutant lineages at time `t + 1`. NOTE: The last entry of the array
  must be the sum of all of the counts from mutant lineages.
- `α̲::Array{Float64}`: Parameters for Dirichlet prior distribution.
- `sₜ_prior::Vector{Real}`: Parameters for the mean fitness prior distribution
  π(sₜ).
- `σₜ_prior::Vector{Real}`: Parameters for the nuisance standard deviation
  parameter prior distribution π(σₜ).
"""
Turing.@model function mean_fitness_neutrals_lognormal(
    r̲ₜ::Vector{Int64},
    r̲ₜ₊₁::Vector{Int64},
    α̲::Vector{Float64},
    sₜ_prior::Vector{Real},
    σₜ_prior::Vector{Real};
)
    # Prior on mean fitness sₜ
    sₜ ~ Turing.Normal(sₜ_prior...)
    # Prior on LogNormal error σₜ
    σₜ ~ Turing.truncated(Turing.Normal(σₜ_prior...); lower=0.0)

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

"""
    mean_fitness_neutrals_lognormal(r̲ₜ, r̲ₜ₊₁, α̲, sₜ_prior, σₜ_prior)

`Turing.jl` model to sample out of the posterior for a single population mean
fitness value sₜ, given the raw barcode counts. Note: this `method` allos the
use of any prior distribution, different from the Normal and Half-Normal priors.

The sampled probability distribution is of the form

π(sₜ, σₜ | f̲ₜ, f̲ₜ₊₁) π(f̲ₜ, f̲ₜ₊₁ | r̲ₜ, r̲ₜ₊₁) ∝ π(f̲ₜ, f̲ₜ₊₁ | sₜ, σₜ) π(sₜ)
π(σₜ) π(f̲ₜ | r̲ₜ) π(f̲ₜ₊₁ |, r̲ₜ₊₁),

where

f̲ₜ / f̲ₜ₊₁ | sₜ, σₜ ~ LogNormal(-sₜ, σₜ),

sₜ ~ sₜ_prior,

σₜ ~ σₜ_prior,

f̲ₜ | r̲ₜ ~ Dirichlet(α̲ .+ r̲ₜ),

and

f̲ₜ₊₁ | r̲ₜ₊₁ ~ Dirichlet(α̲ .+ r̲ₜ₊₁).

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