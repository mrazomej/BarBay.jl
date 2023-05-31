##
# Import package to handle dataframes
import DataFrames as DF

# Import basic mathematical operations
import LinearAlgebra

# Import statistical libraries
import Random
import Distributions
import StatsBase

# Import library to handle MCMC chains
import MCMCChains

##


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Basic statistical functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    matrix_quantile_range(quantile, matrix; dim=2)

Function to compute the quantile ranges of matrix `mat` over dimension `dim`.
For example, if `quantile[1] = 0.95`, This function returns the `0.025` and
`0.975` quantiles that capture 95 percent of the entires on the matrix.

# Arguments
- `quantile::Vector{<:AbstractFloat}`: List of quantiles to extract from the
  posterior predictive checks.
- `matrix::Matrix{<:Real}`: Array over which to compute quantile ranges.

# Optional arguments
- `dim::Int=2`: Dimension over which to compute quantiles. Defualt = 1, i.e.,
  columns.
"""
function matrix_quantile_range(
    quantile::Vector{<:AbstractFloat}, matrix::Matrix{T}; dims::Int=2
) where {T<:Real}
    # Check that all quantiles are within bounds
    if any(.![0.0 ≤ x ≤ 1 for x in quantile])
        error("All quantiles must be between zero and one")
    end # if

    # Check that dim corresponds to a matrix
    if (dims != 1) & (dims != 2)
        error("Dimensions should match a Matrix dimensiosn, i.e., 1 or 2")
    end # if

    # Get opposite dimension   
    op_dims = first([1, 2][[1, 2].∈Set(dims)])

    # Initialize array to save quantiles
    array_quantile = Array{T}(undef, size(matrix, op_dims), length(quantile), 2)

    # Loop through quantile
    for (i, q) in enumerate(quantile)
        # Lower bound
        array_quantile[:, i, 1] = StatsBase.quantile.(
            eachslice(matrix, dims=dims), (1.0 - q) / 2.0
        )
        # Upper bound
        array_quantile[:, i, 2] = StatsBase.quantile.(
            eachslice(matrix, dims=dims), 1.0 - (1.0 - q) / 2.0
        )
    end # for

    return array_quantile

end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Setting priors
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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

## Optional arguments
- `params::Bool=true`: Boolean indicating whether the distribution parameters
  (mean and variance) or the full distribution should be returned

# Returns
if `params == true`:
    - `µ::Vector{Float64}`: Vector encoding the mean values of the Gaussian
    distributions.
    - `σ::Vector{Float64}`: Vector encoding the standard deviation values of the
    Gaussian distributions.
else
    - `dists::Vector{<:Distributions.ContinuousUnivariateDistribution}`: Vector
    with the `Distributions.jl` fit distributions.
"""
function gaussian_prior_mean_fitness(
    data::DF.AbstractDataFrame; params::Bool=true
)
    # Initialize array to save distributions
    dists = Vector{Distributions.ContinuousUnivariateDistribution}(
        undef, size(data, 2)
    )

    # Initialize arrays to save values
    μ = Vector{Float64}(undef, size(data, 2))
    σ = similar(μ)

    # Loop through each column
    for (i, d) in enumerate(eachcol(data))
        # Fit Gaussian distribution
        dists[i] = Distributions.fit(Distributions.Normal, d)
        # Store mean and standard deviation
        µ[i] = Distributions.mean(dists[i])
        σ[i] = Distributions.std(dists[i])
    end # for

    # Check if parameters should be returned
    if params
        return µ, σ
    else
        return dists
    end # if
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

## Optional arguments
- `params::Bool=true`: Boolean indicating whether the distribution parameters
  (mean and variance) or the full distribution should be returned

# Returns
if `params == true`:
    - `µ::Vector{Float64}`: Vector encoding the mean values of the Gaussian
    distributions.
    - `σ::Vector{Float64}`: Vector encoding the standard deviation values of the
    Gaussian distributions.
else
    - `dists::Vector{<:Distributions.ContinuousUnivariateDistribution}`: Vector
    with the `Distributions.jl` fit distributions.
"""
function gaussian_prior_mean_fitness(
    chains::MCMCChains.Chains; params::Bool=true
)
    # Initialize array to save distributions
    dists = Vector{Distributions.ContinuousUnivariateDistribution}(
        undef, size(chains, 2)
    )

    # Initialize arrays to save values
    μ = Vector{Float64}(undef, size(chains, 2))
    σ = similar(μ)

    # Loop through each variable
    for (i, var) in enumerate(names(chains))
        # Extract data
        d = Array(chains[var])
        # Fit Gaussian distribution
        dists[i] = Distributions.fit(Distributions.Normal, d)
        # Store mean and standard deviation
        µ[i] = Distributions.mean(dists[i])
        σ[i] = Distributions.std(dists[i])
    end # for

    # Check if parameters should be returned
    if params
        return µ, σ
    else
        return dists
    end # if
end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Posterior predictive (retrodictive) checks
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    freq_mutant_ppc(n_ppc, df; param, flatten=true)

Function to compute the **posterior predictive checks** for the barcode
frequency for adaptive mutants. Our model predicts the frequency at time ``t+1``
based on the frequency at time ``t`` as

```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. Our inference model assumes that
```math
    \frac{f_{t+1}^{(m)}}{f_{t}^{(m)}} \sim 
    \log-\mathcal{N}\left( s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
```
where ``\sigma^{(m)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution.

# Arguments
- `df::DataFrames.DataFrame`: Dataframe containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - mutant relative fitness values.
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - log-normal likelihood standard deviation.
  - mutant initial frequency.
- `n_ppc::Int`: Number of samples to generate per set of parameters.

## Optional Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the
  name of the variables in the mcmc chain defining the following variables:
  - `:mutant_mean_fitness`: Variable defining the inferred mutant fitness value
    `s⁽ᵐ⁾`.
  - `:mutant_std_fitness`: Variable defining the standard defining the inferred
    standard deviation on the likelihood function `σ⁽ᵐ⁾`.
  - `mutant_freq`: Variable defining the inferred initial frequency for the
    mutant.
  - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
  multiple chain into a single column.
# Returns
- `fₜ₊₁ = fₜ × exp(s⁽ᵐ⁾ - s̅ₜ)::Array{Float64}`: Evaluation of the frequency
  posterior predictive checks at all times for each MCMC sample.
"""
function freq_mutant_ppc(
    df::DF.AbstractDataFrame,
    n_ppc::Int;
    param::Dict{Symbol,Symbol}=Dict(
        :mutant_mean_fitness => :s⁽ᵐ⁾,
        :mutant_std_fitness => :σ⁽ᵐ⁾,
        :mutant_freq => Symbol("f̲⁽ᵐ⁾[1]"),
        :population_mean_fitness => :s̲ₜ,
    ),
    flatten::Bool=true
)
    # Extract variable names for mean fitness
    mean_vars = sort(
        DF.names(df)[
            occursin.(String(param[:population_mean_fitness]), DF.names(df))
        ]
    )

    # Initialize matrix to save PPC
    f_ppc = Array{Float64}(undef, size(df, 1), length(mean_vars) + 1, n_ppc)

    # Set initial frequency
    f_ppc[:, 1, :] = hcat(repeat([df[:, param[:mutant_freq]]], n_ppc)...)

    # Loop through time points
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        f_ppc[:, i+1, :] = f_ppc[:, i, :] .* Random.rand(
            Distributions.MvLogNormal(
                df[:, param[:mutant_mean_fitness]] .- df[:, var],
                LinearAlgebra.Diagonal(df[:, param[:mutant_std_fitness]] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(f_ppc, dims=3))...)
    else
        # Return raw matrix
        return f_ppc
    end # if
end # function

@doc raw"""
    freq_mutant_ppc(n_ppc, df; param, flatten=true)

Function to compute the **posterior predictive checks** for the barcode
frequency for adaptive mutants. Our model predicts the frequency at time ``t+1``
based on the frequency at time ``t`` as

```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. Our inference model assumes that
```math
    \frac{f_{t+1}^{(m)}}{f_{t}^{(m)}} \sim 
    \log-\mathcal{N}\left( s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
```
where ``\sigma^{(m)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution.

# Arguments
- `chain::MCMCChains.Chains`: Chain containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - mutant relative fitness values.
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - log-normal likelihood standard deviation.
  - mutant initial frequency.
- `n_ppc::Int`: Number of samples to generate per set of parameters.

## Optional Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the
  name of the variables in the mcmc chain defining the following variables:
  - `:mutant_mean_fitness`: Variable defining the inferred mutant fitness value
    `s⁽ᵐ⁾`.
  - `:mutant_std_fitness`: Variable defining the standard defining the inferred
    standard deviation on the likelihood function `σ⁽ᵐ⁾`.
  - `mutant_freq`: Variable defining the inferred initial frequency for the
    mutant.
  - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
  multiple chain into a single column.

# Returns
- `fₜ₊₁ = fₜ × exp(s⁽ᵐ⁾ - s̅ₜ)::Array{Float64}`: Evaluation of the frequency
  posterior predictive checks at all times for each MCMC sample.
"""
function freq_mutant_ppc(
    chain::MCMCChains.Chains,
    n_ppc::Int;
    param::Dict{Symbol,Symbol}=Dict(
        :mutant_mean_fitness => :s⁽ᵐ⁾,
        :mutant_std_fitness => :σ⁽ᵐ⁾,
        :mutant_freq => Symbol("f̲⁽ᵐ⁾[1]"),
        :population_mean_fitness => :s̲ₜ,
    ),
    flatten::Bool=true
)
    # Extract variable names for mean fitness
    mean_vars = sort(
        MCMCChains.namesingroup(chain, param[:population_mean_fitness])
    )

    # Compute number of MCMC samples in chain from number of chains and range of
    # samples
    n_samples = length(MCMCChains.chains(chain)) *
                length(MCMCChains.range(chain))

    # Initialize matrix to save PPC
    f_ppc = Array{Float64}(undef, n_samples, length(mean_vars) + 1, n_ppc)

    # Set initial frequency
    f_ppc[:, 1, :] = hcat(repeat([chain[param[:mutant_freq]][:]], n_ppc)...)

    # Loop through time points
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        f_ppc[:, i+1, :] = f_ppc[:, i, :] .* Random.rand(
            Distributions.MvLogNormal(
                chain[param[:mutant_mean_fitness]][:] .- chain[var][:],
                LinearAlgebra.Diagonal(chain[param[:mutant_std_fitness]][:] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(f_ppc, dims=3))...)
    else
        # Return raw matrix
        return f_ppc
    end # if
end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    logfreq_ratio_mutant_ppc(df, n_ppc, param; flatten=true)

Function to compute the **posterior predictive checks** for the barcode
log-frequency ratio for adaptive mutants. Our model predicts the frequency at
time ``t+1`` based on the frequency at time ``t`` as

```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. Our inference model assumes that
```math
    \frac{f_{t+1}^{(m)}}{f_{t}^{(m)}} \sim 
    \log-\mathcal{N}\left( s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
```
where ``\sigma^{(m)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution.

# Arguments
- `df::DataFrames.DataFrame`: Dataframe containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - mutant relative fitness values.
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - log-normal likelihood standard deviation.
  - mutant initial frequency.
- `n_ppc::Int`: Number of samples to generate per set of parameters.

## Optional Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the
name of the variables in the mcmc chain defining the following variables:
  - `:mutant_mean_fitness`: Variable defining the inferred mutant fitness value
    `s⁽ᵐ⁾`.
  - `:mutant_std_fitness`: Variable defining the standard defining the inferred
    standard deviation on the likelihood function `σ⁽ᵐ⁾`.
  - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
  multiple chain into a single column.

# Returns
- `log(fₜ₊₁ / fₜ) = s⁽ᵐ⁾ - s̅ₜ::Array{Float64}`: Evaluation of the frequency
  posterior predictive checks at all times for each MCMC sample.
"""
function logfreq_ratio_mutant_ppc(
    df::DF.AbstractDataFrame,
    n_ppc::Int;
    param::Dict{Symbol,Symbol}=Dict(
        :mutant_mean_fitness => :s⁽ᵐ⁾,
        :mutant_std_fitness => :σ⁽ᵐ⁾,
        :population_mean_fitness => :s̲ₜ,
    ),
    flatten::Bool=true
)
    # Extract variable names for mean fitness
    mean_vars = sort(
        DF.names(df)[
            occursin.(String(param[:population_mean_fitness]), DF.names(df))
        ]
    )

    # Initialize matrix to save PPC
    γ_ppc = Array{Float64}(undef, size(df, 1), length(mean_vars), n_ppc)

    # Loop through time points
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        γ_ppc[:, i, :] = Random.rand(
            Distributions.MvNormal(
                df[:, param[:mutant_mean_fitness]] .- df[:, var],
                LinearAlgebra.Diagonal(df[:, param[:mutant_std_fitness]] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(γ_ppc, dims=3))...)
    else
        # Return raw matrix
        return γ_ppc
    end # if
end # function

@doc raw"""
    logfreq_ratio_mutant_ppc(df, n_ppc; param, flatten=true)

Function to compute the **posterior predictive checks** for the barcode
log-frequency ratio for adaptive mutants. Our model predicts the frequency at
time ``t+1`` based on the frequency at time ``t`` as

```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. Our inference model assumes that
```math
    \frac{f_{t+1}^{(m)}}{f_{t}^{(m)}} \sim 
    \log-\mathcal{N}\left( s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
```
where ``\sigma^{(m)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution.

# Arguments
- `chain::MCMCChains.Chains`: Chain containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - mutant relative fitness values.
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - log-normal likelihood standard deviation.
  - mutant initial frequency.
- `n_ppc::Int`: Number of samples to generate per set of parameters.

## Optional Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the
name of the variables in the mcmc chain defining the following variables:
  - `:mutant_mean_fitness`: Variable defining the inferred mutant fitness value
    `s⁽ᵐ⁾`.
  - `:mutant_std_fitness`: Variable defining the standard defining the inferred
    standard deviation on the likelihood function `σ⁽ᵐ⁾`.
  - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
  multiple chain into a single column.

# Returns
- `log(fₜ₊₁ / fₜ) = s⁽ᵐ⁾ - s̅ₜ::Array{Float64}`: Evaluation of the frequency
  posterior predictive checks at all times for each MCMC sample.
"""
function logfreq_ratio_mutant_ppc(
    chain::MCMCChains.Chains,
    n_ppc::Int;
    param::Dict{Symbol,Symbol}=Dict(
        :mutant_mean_fitness => :s⁽ᵐ⁾,
        :mutant_std_fitness => :σ⁽ᵐ⁾,
        :population_mean_fitness => :s̲ₜ,
    ),
    flatten::Bool=true
)
    # Extract variable names for mean fitness
    mean_vars = sort(
        MCMCChains.namesingroup(chain, param[:population_mean_fitness])
    )

    # Compute number of MCMC samples in chain from number of chains and range of
    # samples
    n_samples = length(MCMCChains.chains(chain)) *
                length(MCMCChains.range(chain))

    # Initialize matrix to save PPC
    γ_ppc = Array{Float64}(undef, n_samples, length(mean_vars), n_ppc)

    # Loop through time points
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        γ_ppc[:, i, :] = Random.rand(
            Distributions.MvNormal(
                chain[param[:mutant_mean_fitness]][:] .- chain[var][:],
                LinearAlgebra.Diagonal(chain[param[:mutant_std_fitness]][:] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(γ_ppc, dims=3))...)
    else
        # Return raw matrix
        return γ_ppc
    end # if
end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    logfreq_ratio_mean_ppc(df, n_ppc, param; flatten=true)

Function to compute the **posterior predictive checks** (better called the
posterior retrodictive checks) for the barcode log-frequency ratio for neutral
lineages. Our model predicts the frequency at time ``t+1`` based on the
frequency at time ``t`` as

```math
    f_{t+1}^{(n)} = f_{t}^{(n)} 
    \exp\left[ \left( - \bar{s}_t \right) \tau \right],
```
where ``\bar{s}_t`` is the population mean fitness between time ``t`` and
``t+1``, and ``\tau`` is the time interval between time ``t`` and ``t+1``. Our
inference model assumes that
```math
    \frac{f_{t+1}^{(n)}}{f_{t}^{(n)}} \sim 
    \log-\mathcal{N}\left( - \bar{s}_t, \sigma^{(n)} \right),
```
where ``\sigma^{(n)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution.

# Arguments
- `n_ppc::Int`: Number of samples to generate per set of parameters.
- `df::DataFrames.DataFrame`: Dataframe containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - log-normal likelihood standard deviation.

## Optional Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the name of the variables
in the mcmc chain defining the following variables:
    - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
    - `population_std_fitness`: Common pattern in all standard deviations
      estimates for the likelihood.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
  multiple chain into a single column.

# Returns
- `log(fₜ₊₁ / fₜ)= s⁽ᵐ⁾ - s̅ₜ::Array{Float64}`: Evaluation of the log frequency
  ratio posterior predictive checks at all times for each MCMC sample.
"""
function logfreq_ratio_mean_ppc(
    df::DF.AbstractDataFrame,
    n_ppc::Int;
    param::Dict{Symbol,Symbol}=Dict(
        :population_mean_fitness => :sₜ,
        :population_std_fitness => :σₜ
    ),
    flatten::Bool=true
)
    # Extract variable names for mean fitness
    mean_vars = sort(
        DF.names(df)[
            occursin.(String(param[:population_mean_fitness]), DF.names(df))
        ]
    )

    # Extract variable names for standard deviation
    std_vars = sort(
        DF.names(df)[
            occursin.(String(param[:population_std_fitness]), DF.names(df))
        ]
    )

    # Check that number of mean and std variables matches
    if length(mean_vars) != length(std_vars)
        error("The number of mean and standard deviation variables does not match")
    end # if

    # Initialize matrix to save PPC
    γ_ppc = Array{Float64}(undef, size(df, 1), length(mean_vars), n_ppc)

    # Loop through 
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        γ_ppc[:, i, :] = Random.rand(
            Distributions.MvNormal(
                -df[:, var],
                LinearAlgebra.Diagonal(df[:, std_vars[i]] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(γ_ppc, dims=3))...)
    else
        # Return raw matrix
        return γ_ppc
    end # if
end # function

@doc raw"""
    logfreq_ratio_mean_ppc(chain, n_ppc; param, flatten=true)

Function to compute the **posterior predictive checks** (better called the
posterior retrodictive checks) for the barcode log-frequency ratio for neutral
lineages. Our model predicts the frequency at time ``t+1`` based on the
frequency at time ``t`` as

```math
    f_{t+1}^{(n)} = f_{t}^{(n)} 
    \exp\left[ \left( - \bar{s}_t \right) \tau \right],
```
where ``\bar{s}_t`` is the population mean fitness between time ``t`` and
``t+1``, and ``\tau`` is the time interval between time ``t`` and ``t+1``. Our
inference model assumes that
```math
    \frac{f_{t+1}^{(n)}}{f_{t}^{(n)}} \sim 
    \log-\mathcal{N}\left( - \bar{s}_t, \sigma^{(n)} \right),
```
where ``\sigma^{(n)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution.

# Arguments
- `n_ppc::Int`: Number of samples to generate per set of parameters.
- `df::DataFrames.DataFrame`: Dataframe containing the MCMC samples for the
    variables needed to compute the posterior predictive checks. The dataframe
    should have MCMC samples for
    - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
    - log-normal likelihood standard deviation.

## Optional Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the name of the variables
in the mcmc chain defining the following variables:
    - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
    - `population_std_fitness`: Common pattern in all standard deviations
        estimates for the likelihood.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
    multiple chain into a single column.

# Returns
- `log(fₜ₊₁ / fₜ) = s⁽ᵐ⁾ - s̅ₜ::Array{Float64}`: Evaluation of the log frequency
  ratio posterior predictive checks at all times for each MCMC sample.
"""
function logfreq_ratio_mean_ppc(
    chains::MCMCChains.Chains,
    n_ppc::Int;
    param::Dict{Symbol,Symbol}=Dict(
        :population_mean_fitness => :sₜ,
        :population_std_fitness => :σₜ
    ),
    flatten::Bool=true
)
    # Extract variable names for mean fitness
    mean_vars = MCMCChains.namesingroup(chains, param[:population_mean_fitness])

    # Extract variable names for standard deviation
    std_vars = MCMCChains.namesingroup(chains, param[:population_std_fitness])

    # Check that number of mean and std variables matches
    if length(mean_vars) != length(std_vars)
        error("The number of mean and standard deviation variables does not match")
    end # if

    # Compute number of MCMC samples in chain from number of chains and range of
    # samples
    n_samples = length(MCMCChains.chains(chains)) *
                length(MCMCChains.range(chains))

    # Initialize matrix to save PPC
    γ_ppc = Array{Float64}(undef, n_samples, length(mean_vars), n_ppc)

    # Loop through 
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        γ_ppc[:, i, :] = Random.rand(
            Distributions.MvNormal(
                -chains[var][:],
                LinearAlgebra.Diagonal(chains[std_vars[i]][:] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(γ_ppc, dims=3))...)
    else
        # Return raw matrix
        return γ_ppc
    end # if
end # function