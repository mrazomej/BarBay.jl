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
import Turing
import MCMCChains
import DynamicPPL

# Import library to return unconstrained versions of distributions
using Bijectors


# Import libraries for convenient array indexing
import ComponentArrays
import UnPack
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
# Posterior predictive (retrodictive) checks for frequency trajectories
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
# Posterior predictive (retrodictive) checks for log-freq ratio trajectories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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

## Optional Keyword Arguments
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
    logγ_ppc = Array{Float64}(undef, size(df, 1), length(mean_vars), n_ppc)

    # Loop through time points
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        logγ_ppc[:, i, :] = Random.rand(
            Distributions.MvNormal(
                df[:, param[:mutant_mean_fitness]] .- df[:, var],
                LinearAlgebra.Diagonal(df[:, param[:mutant_std_fitness]] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(logγ_ppc, dims=3))...)
    else
        # Return raw matrix
        return logγ_ppc
    end # if
end # function

@doc raw"""
    logfreq_ratio_mutant_ppc(chain, n_ppc; param, flatten=true)

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
    logγ_ppc = Array{Float64}(undef, n_samples, length(mean_vars), n_ppc)

    # Loop through time points
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        logγ_ppc[:, i, :] = Random.rand(
            Distributions.MvNormal(
                chain[param[:mutant_mean_fitness]][:] .- chain[var][:],
                LinearAlgebra.Diagonal(chain[param[:mutant_std_fitness]][:] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(logγ_ppc, dims=3))...)
    else
        # Return raw matrix
        return logγ_ppc
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
    logγ_ppc = Array{Float64}(undef, size(df, 1), length(mean_vars), n_ppc)

    # Loop through 
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        logγ_ppc[:, i, :] = Random.rand(
            Distributions.MvNormal(
                -df[:, var],
                LinearAlgebra.Diagonal(df[:, std_vars[i]] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(logγ_ppc, dims=3))...)
    else
        # Return raw matrix
        return logγ_ppc
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
    logγ_ppc = Array{Float64}(undef, n_samples, length(mean_vars), n_ppc)

    # Loop through 
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        logγ_ppc[:, i, :] = Random.rand(
            Distributions.MvNormal(
                -chains[var][:],
                LinearAlgebra.Diagonal(chains[std_vars[i]][:] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(logγ_ppc, dims=3))...)
    else
        # Return raw matrix
        return logγ_ppc
    end # if
end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Posterior predictive (retrodictive) checks for log-freq ratio trajectories
# in multiple environments
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    logfreq_ratio_mutienv_ppc(df, n_ppc, param; flatten=true)

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
- `envs::Vector{<:Any}`: List of environments in experiment. This is used to
  index the corresponding fitness from the chain. NOTE: The list of environments
  should be the name or corresponding label of the environemnt; the index is
  generated internally.

## Optional Keyword Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the name of the variables
in the mcmc chain defining the following variables:
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
function logfreq_ratio_multienv_ppc(
    df::DF.AbstractDataFrame,
    n_ppc::Int,
    envs::Vector{<:Any};
    param::Dict{Symbol,Symbol}=Dict(
        :mutant_mean_fitness => :s̲⁽ᵐ⁾,
        :mutant_std_fitness => :σ̲⁽ᵐ⁾,
        :population_mean_fitness => :s̲ₜ,
    ),
    flatten::Bool=true
)
    # Find unique environments
    env_unique = unique(envs)
    # Define number of environments
    n_env = length(env_unique)
    # Define environmental indexes
    env_idx = indexin(envs, env_unique)

    # Extract variable names for mean fitness
    mean_vars = sort(
        DF.names(df)[
            occursin.(String(param[:population_mean_fitness]), DF.names(df))
        ]
    )

    # Extract variable names for mutant relative fitness
    s_vars = sort(
        DF.names(df)[
            occursin.(String(param[:mutant_mean_fitness]), DF.names(df))
        ]
    )

    # Extract variable names for mutant relative fitness error
    σ_vars = sort(
        DF.names(df)[
            occursin.(String(param[:mutant_std_fitness]), DF.names(df))
        ]
    )

    # Check that number of environments matches number of variables
    if (length(s_vars) != n_env) | (length(σ_vars) != n_env)
        error(
            "# of mutant-related variables does not match # of environments"
        )
    end # if

    # Check if list of environments matches number of time points
    if (length(envs) != length(mean_vars) + 1)
        error("Number of given environments does not match time points in chain")
    end # if

    # Initialize matrix to save PPC
    logγ_ppc = Array{Float64}(undef, size(df, 1), length(mean_vars), n_ppc)

    # Loop through time points
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        logγ_ppc[:, i, :] = Random.rand(
            Distributions.MvNormal(
                df[:, s_vars[env_idx[i+1]]] .- df[:, var],
                LinearAlgebra.Diagonal(df[:, σ_vars[env_idx[i+1]]] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(logγ_ppc, dims=3))...)
    else
        # Return raw matrix
        return logγ_ppc
    end # if
end # function

function logfreq_ratio_multienv_ppc(
    chain::MCMCChains.Chains,
    n_ppc::Int,
    envs::Vector{<:Any};
    param::Dict{Symbol,Symbol}=Dict(
        :mutant_mean_fitness => :s̲⁽ᵐ⁾,
        :mutant_std_fitness => :σ̲⁽ᵐ⁾,
        :population_mean_fitness => :s̲ₜ,
    ),
    flatten::Bool=true
)
    # Find unique environments
    env_unique = unique(envs)
    # Define number of environments
    n_env = length(env_unique)
    # Define environmental indexes
    env_idx = indexin(envs, env_unique)

    # Extract variable names for mean fitness
    mean_vars = sort(
        MCMCChains.namesingroup(chain, param[:population_mean_fitness])
    )

    # Extract variable names for mutant relative fitness
    s_vars = sort(
        MCMCChains.namesingroup(chain, param[:mutant_mean_fitness])
    )

    # Extract variable names for mutant relative fitness error
    σ_vars = sort(
        MCMCChains.namesingroup(chain, param[:mutant_std_fitness])
    )

    # Check that number of environments matches number of variables
    if (length(s_vars) != n_env) | (length(σ_vars) != n_env)
        error(
            "# of mutant-related variables does not match # of environments"
        )
    end # if

    # Check if list of environments matches number of time points
    if (length(envs) != length(mean_vars) + 1)
        error("Number of given environments does not match time points in chain")
    end # if

    # Compute number of MCMC samples in chain from number of chains and range of
    # samples
    n_samples = length(MCMCChains.chains(chain)) *
                length(MCMCChains.range(chain))

    # Initialize matrix to save PPC
    logγ_ppc = Array{Float64}(undef, n_samples, length(mean_vars), n_ppc)

    # Loop through time points
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        logγ_ppc[:, i, :] = Random.rand(
            Distributions.MvNormal(
                chain[s_vars[env_idx[i+1]]][:] .- chain[var][:],
                LinearAlgebra.Diagonal(chain[σ_vars[env_idx[i+1]]][:] .^ 2)
            ),
            n_ppc
        )
    end # for

    if flatten
        # Return flatten matrix
        return vcat(collect(eachslice(logγ_ppc, dims=3))...)
    else
        # Return raw matrix
        return logγ_ppc
    end # if
end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Naive fitness estimate
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    naive_fitness(data; id_col, time_col, count_col, neutral_col, pseudo_count)

Function to compute a naive estimate of mutant fitness data based on counts. The
fitness estimate is computed as

```math
\left\langle
\log\frac{f^{(m)}_{t+1}}{f^{(m)}_{t}} - \log\frac{f^{(n)}_{t+1}}{f^{(n)}_{t}}
\right\rangle = s^{(m)}
```

# Arguments
- `data::DataFrames.AbstractDataFrame`: **Tidy dataframe** with the data to be
used to infer the fitness values on mutants. The `DataFrame` must contain at
least the following columns:
    - `id_col`: Column identifying the ID of the barcode. This can the barcode
    sequence, for example.
    - `time_col`: Column defining the measurement time point.
    - `count_col`: Column with the raw barcode count.
    - `neutral_col`: Column indicating whether the barcode is from a neutral
    lineage or not.

## Optional Keyword Arguments
- `id_col::Symbol=:barcode`: Name of the column in `data` containing the barcode
    identifier. The column may contain any type of entry.
- `time_col::Symbol=:time`: Name of the column in `data` defining the time point
  at which measurements were done. The column may contain any type of entry as
  long as `sort` will resulted in time-ordered names.
- `count_col::Symbol=:count`: Name of the column in `data` containing the raw
  barcode count. The column must contain entries of type `Int64`.
- `neutral_col::Symbol=:neutral`: Name of the column in `data` defining whether
  the barcode belongs to a neutral lineage or not. The column must contain
  entries of type `Bool`.
- `rm_T0::Bool=false`: Optional argument to remove the first time point from the
inference. Commonly, the data from this first time point is of much lower
quality. Therefore, removing this first time point might result in a better
inference.
- `pseudo_count::Int=1`: Pseudo count number to add to all counts. This is
  useful to avoid divisions by zero.

# Returns
- `DataFrames.DataFrame`: Data frame with two columns:
    - `id_col`: Column indicating the strain ID.
    - `fitness`: Naive fitness estimate.
"""
function naive_fitness(
    data::DF.AbstractDataFrame;
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rm_T0::Bool=false,
    pseudo_count::Int=1
)
    # Keep only the needed data to work with
    data = data[:, [id_col, time_col, count_col, neutral_col]]

    # Extract unique time points
    timepoints = sort(unique(data[:, time_col]))

    # Remove T0 if indicated
    if rm_T0
        data = data[.!(data[:, time_col] .== first(timepoints)), :]
    end # if

    # Add pseudo-count to each barcode to avoid division by zero
    data[:, count_col] .+= pseudo_count

    # Extract total counts per barcode
    data_total = DF.combine(DF.groupby(data, time_col), count_col => sum)
    # Add total count column to dataframe
    DF.leftjoin!(data, data_total; on=time_col)

    # Add frequency column
    DF.insertcols!(data, :freq => data[:, count_col] ./ data.count_sum)

    # Initialize dataframe to save the log freq changes
    data_log = DF.DataFrame()

    # Group data by barcode
    data_group = DF.groupby(data, id_col)

    # Loop through each group
    for d in data_group
        # Compute log change
        DF.append!(
            data_log,
            DF.DataFrame(
                id_col .=> first(d[:, id_col]),
                time_col => d[2:end, time_col],
                :logf => diff(log.(d.freq)),
                neutral_col .=> first(d[:, neutral_col])
            )
        )
    end # for

    # Compute the mean population fitness s̄ₜ for all timepoints
    data_st = DF.combine(
        DF.groupby(data_log[data_log[:, neutral_col], :], time_col),
        :logf => StatsBase.mean
    )

    # Locate index to extract the corresponding mean population fitness
    idx_st = [
        findfirst(x .== data_st[:, time_col]) for x in data_log[:, time_col]
    ]

    # Add normalized column to dataframe
    DF.insertcols!(
        data_log, :logf_norm => data_log.logf .- data_st[idx_st, :logf_mean]
    )

    # Compute mean fitness and return it as dataframe
    return DF.rename!(
        DF.combine(
            DF.groupby(data_log[.!data_log[:, neutral_col], :], id_col),
            :logf_norm => StatsBase.mean
        ),
        :logf_norm_mean => :fitness
    )
end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define full-rank normal distribution for variational inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    Function to build a full-rank distribution to be used for ADVI optimization.
    The code in this function comes from (`Turing.jl
    tutorial`)[https://turinglang.org/v0.28/tutorials/09-variational-inference/]

# Arguments
- `dim::Int`: Dimensionality of parameter space.
- `model::DynamicPPL.model`: Turing model to be fit using ADVI.

# Returns
Initialized distribution to be used when fitting a full-rank variational model.
"""
function build_getq(dim, model)
    # Define base distribution as standard normal.
    base_dist = Turing.DistributionsAD.TuringDiagMvNormal(zeros(dim), ones(dim))

    # From Turing.jl:
    # > bijector(model::Turing.Model) is defined by Turing, and will return a
    # bijector which takes you from the space of the latent variables to the
    # real space. We're interested in using a normal distribution as a
    # base-distribution and transform samples to the latent space, thus we need
    # the inverse mapping from the reals to the latent space.
    constrained_dist = Bijectors.inverse(Bijectors.bijector(model))

    # Define proto array with parameters for full-rank normal distribution.
    # Note: Using the ComponentArray makes things much simpler to work with.
    proto_arr = ComponentArrays.ComponentArray(;
        L=zeros(dim, dim), b=zeros(dim)
    )

    # Get Axes for proto-array. This basically returns the shape of each element
    # in proto_arr
    proto_axes = ComponentArrays.getaxes(proto_arr)
    # Define number of parameters
    num_params = length(proto_arr)

    # Define getq function to be returned with specific dimensions for full-rank
    # variational problem.
    function getq(θ)
        # Unpack parameters. This is where the combination of
        # `ComponentArrays.jl` and `UnPack.jl` become handy.
        L, b = begin
            # Unpack covariance matrix and mean array
            UnPack.@unpack L, b = ComponentArrays.ComponentArray(θ, proto_axes)
            # Convert covariance matrix to lower diagonal covariance matrix to
            # use Cholesky decomposition.
            LinearAlgebra.LowerTriangular(L), b
        end

        # From Turing.jl:
        # > For this to represent a covariance matrix we need to ensure that the
        # diagonal is positive. We can enforce this by zeroing out the diagonal
        # and then adding back the diagonal exponentiated.

        # 1. Extract diagonal elements of matrix L
        D = LinearAlgebra.Diagonal(LinearAlgebra.diag(L))
        # 2. Subtract diagonal elements to make the L diagonal be all zeros,
        #    then, add the exponential of the diagonal to ensure positivit.
        A = L - D + exp(D)

        # Define unconstrained parameters by composing the constrained
        # distribution with the bijectors Shift and Scale. The ∘ operator means
        # to compose functions (f∘g)(x) = f(g(x)). NOTE: I do not fully
        # follow how this works, I am using Turing.jl's example.

        b = constrained_dist ∘ Bijectors.Shift(b) ∘ Bijectors.Scale(A)

        return Turing.transformed(base_dist, b)
    end

    # Return resulting getq function
    return getq

end # function