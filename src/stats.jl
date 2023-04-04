##
# Import package to handle dataframes
import DataFrames as DF

# Import statistical libraries
import Distributions
import StatsBase

##

@doc raw"""
    dirichlet_prior_neutral(neutrals)

Function to return the vector α̲ for the equivalent of a uniform Dirichlet prior
when inferring the population mean fitness with the neutral lineages. 

# Arguments
- `neutrals::Vector{Bool}`: Vector indicating which barcodes correspond to
  neutral lineages and wich to mutant lineages.

# Returns
- `α̲::Vector{Float64}`: Parameters for uniform Dirichlet prior. All lineages
  lineages are assigned α = 1. The mutant lineages are grouped together into a
  single term with αᴹ = ∑ₘ α, i.e., the number of non-neutral lineages.
"""
function dirichlet_prior_neutral(neutrals::Vector{Bool})
    # Generate vector of α parameters for uniform Dirichlet prior
    return Float64.([repeat([1], sum(neutrals)); sum(.!(neutrals))])
end # function

@doc raw"""
    beta_prior_mutant(neutrals)

Function to return the vector α̲ for the equivalent of a uniform Dirichlet prior
when inferring the relative fitness of a single mutant. Since we use the Beta
distribution as the prior when inferring the marginal distribution, this
function assigns a `1` to the mutant parameter and a `B̲ - 1` to the complement,
where `B̲` is the total number of unique barcodes.

# Arguments
- `bc_id::Vector{Any}`: Vector with the IDs for each barcode

# Returns
- `α̲::Vector{Float64}`: Parameters for Beta prior. Mutant lineage is assigned
  `α = 1`. The rest of the lineages are grouped together into a single term with
  `αᴮ = B̲ - 1`, i.e., the number of lineages minus the mutant lineage being
  inferred.
"""
function beta_prior_mutant(bc_id::Vector{<:Any})
    # Generate vector of α parameters for uniform Dirichlet prior
    return Float64.([1, length(unique(bc_id)) - 1])
end # function

@doc raw"""
    gaussian_prior_mean_fitness(data)

Function that fits Gaussian (normal) distributions to MCMC traces from the
population mean fitness s̄ₜ. These Gaussians are then used during the mutant
relative fitness inference.

# Arguments
- `data::DataFrames.AbstractDataFrame`: DataFrame containing the MCMC samples
  for each of the inferred population mean fitness values, one inferred mean
  fitness per column.

# Returns
- `µ::Vector{Float64}`: Vector encoding the mean values of the Gaussian
  distributions.
- `σ::Vector{Float64}`: Vector encoding the standard deviation values of the
  Gaussian distributions.
"""
function gaussian_prior_mean_fitness(data::DF.AbstractDataFrame)
    # Initialize arrays to save values
    μ = Vector{Float64}(undef, size(data, 2))
    σ = similar(μ)

    # Loop through each column
    for (i, d) in enumerate(eachcol(data))
        # Fit Gaussian distribution
        dist = Distributions.fit(Distributions.Normal, d)
        # Store mean and standard deviation
        µ[i] = Distributions.mean(dist)
        σ[i] = Distributions.std(dist)
    end # for

    return µ, σ
end # function