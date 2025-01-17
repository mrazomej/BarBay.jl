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

# Import needed function from the utils module
using ..utils: data_to_arrays

##


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Basic statistical functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
matrix_quantile_range(quantile, matrix; dim=2) 

Function to compute the quantile ranges of matrix `matrix` over dimension `dim`.

For example, if `quantile[1] = 0.95`, this function returns the `0.025` and 
`0.975` quantiles that capture 95 percent of the entries in the matrix.

# Arguments
- `quantile::Vector{<:AbstractFloat}`: List of quantiles to extract from the
  posterior predictive checks.  
- `matrix::Matrix{<:Real}`: Array over which to compute quantile ranges.

# Keyword Arguments
- `dim::Int=2`: Dimension over which to compute quantiles. Default is 2, i.e. 
  columns.

# Returns
- `qs`: Matrix with requested quantiles over specified dimension.
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
    freq_bc_ppc(df, n_ppc; kwargs)

Function to compute the **posterior predictive checks** for the barcode
frequency for adaptive mutants. 
    
# Model
    
The functional form that connects the barcode frequency at time ``t+1`` with the
frequency at time ``t`` is of the form
```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. The statistical models in this package
assume that
```math
    \log\left(\frac{f_{t+1}^{(m)}}{f_{t}^{(m)}}\right) \sim 
    \mathcal{N}\left( s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
```
where ``\sigma^{(m)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution to produce the posterior
predictive checks.

# Arguments
- `df::DataFrames.DataFrame`: Dataframe containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - mutant relative fitness values.
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - (log)-normal likelihood standard deviation.
  - mutant initial frequency.
- `n_ppc::Int`: Number of samples to generate per set of parameters.

## Optional Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the name of the variables
  in the mcmc chain defining the following variables:
  - `:bc_mean_fitness`: Variable defining the inferred mutant fitness value
    `s⁽ᵐ⁾`.
  - `:bc_std_fitness`: Variable defining the standard defining the inferred
    standard deviation on the likelihood function `σ⁽ᵐ⁾`.
  - `bc_freq`: Variable defining the inferred initial frequency for the mutant.
  - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
- `model::Symbol=:lognormal`: Either `:normal` or `:lognormal` to indicate if
    the model used a normal or lognormal distribution for the likelihood. This
    is because when using a normal distribution, the nuisance parameters are
    sampled in log scale and need to be exponentiated.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
  multiple chain into a single column.
# Returns
- `fₜ₊₁ = fₜ × exp(s⁽ᵐ⁾ - s̅ₜ)::Array{Float64}`: Evaluation of the frequency
  posterior predictive checks at all times for each MCMC sample.
"""
function freq_bc_ppc(
    df::DF.AbstractDataFrame,
    n_ppc::Int;
    param::Dict{Symbol,Symbol}=Dict(
        :bc_mean_fitness => :s⁽ᵐ⁾,
        :bc_std_fitness => :σ⁽ᵐ⁾,
        :bc_freq => Symbol("f̲⁽ᵐ⁾[1]"),
        :population_mean_fitness => :s̲ₜ,
    ),
    model::Symbol=:lognormal,
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
    f_ppc[:, 1, :] = hcat(repeat([df[:, param[:bc_freq]]], n_ppc)...)

    # Loop through time points
    for (i, var) in enumerate(mean_vars)
        if model == :lognormal
            # Sample out of posterior distribution
            f_ppc[:, i+1, :] = f_ppc[:, i, :] .* Random.rand(
                Distributions.MvLogNormal(
                    df[:, param[:bc_mean_fitness]] .- df[:, var],
                    LinearAlgebra.Diagonal(
                        df[:, param[:bc_std_fitness]] .^ 2
                    )
                ),
                n_ppc
            )
        elseif model == :normal
            # Sample out of posterior distribution
            f_ppc[:, i+1, :] = f_ppc[:, i, :] .* Random.rand(
                Distributions.MvLogNormal(
                    df[:, param[:bc_mean_fitness]] .- df[:, var],
                    LinearAlgebra.Diagonal(
                        exp.(df[:, param[:bc_std_fitness]]) .^ 2
                    )
                ),
                n_ppc
            )
        else
            error("model must be :normal or :lognormal")
        end # if
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
    freq_bc_ppc(chain, n_ppc; kwargs)

Function to compute the **posterior predictive checks** for the barcode
frequency for adaptive mutants. 
    
# Model

The functional form that connects the barcode frequency at time ``t+1`` with the
frequency at time ``t`` is of the form
```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. The statistical models in this package
assume that
```math
    \log\left(\frac{f_{t+1}^{(m)}}{f_{t}^{(m)}}\right) \sim 
    \mathcal{N}\left( s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
```
where ``\sigma^{(m)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution to produce the posterior
predictive checks.

# Arguments
- `chain::MCMCChains.Chains`: Chain containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - mutant relative fitness values.
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - (log)-normal likelihood standard deviation.
  - mutant initial frequency.
- `n_ppc::Int`: Number of samples to generate per set of parameters.

## Optional Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the name of the variables
  in the mcmc chain defining the following variables:
  - `:bc_mean_fitness`: Variable defining the inferred mutant fitness value
    `s⁽ᵐ⁾`.
  - `:bc_std_fitness`: Variable defining the standard defining the inferred
    standard deviation on the likelihood function `σ⁽ᵐ⁾`.
  - `bc_freq`: Variable defining the inferred initial frequency for the
    mutant.
  - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
  multiple chain into a single column.

# Returns
- `fₜ₊₁ = fₜ × exp(s⁽ᵐ⁾ - s̅ₜ)::Array{Float64}`: Evaluation of the frequency
  posterior predictive checks at all times for each MCMC sample.
"""
function freq_bc_ppc(
    chain::MCMCChains.Chains,
    n_ppc::Int;
    param::Dict{Symbol,Symbol}=Dict(
        :bc_mean_fitness => :s⁽ᵐ⁾,
        :bc_std_fitness => :σ⁽ᵐ⁾,
        :bc_freq => Symbol("f̲⁽ᵐ⁾[1]"),
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
    f_ppc[:, 1, :] = hcat(repeat([chain[param[:bc_freq]][:]], n_ppc)...)

    # Loop through time points
    for (i, var) in enumerate(mean_vars)
        # Sample out of posterior distribution
        f_ppc[:, i+1, :] = f_ppc[:, i, :] .* Random.rand(
            Distributions.MvLogNormal(
                chain[param[:bc_mean_fitness]][:] .- chain[var][:],
                LinearAlgebra.Diagonal(
                    exp.(chain[param[:bc_std_fitness]][:]) .^ 2
                )
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
    logfreq_ratio_bc_ppc(df, n_ppc; kwargs)

Function to compute the **posterior predictive checks** for the barcode
log-frequency ratio for adaptive mutants. 

# Model
    
The functional form that connects the barcode frequency at time ``t+1`` with the
frequency at time ``t`` is of the form
```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. The statistical models in this package
assume that
```math
    \log\left(\frac{f_{t+1}^{(m)}}{f_{t}^{(m)}}\right) \sim 
    \mathcal{N}\left( s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
```
where ``\sigma^{(m)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution to produce the posterior
predictive checks.

# Arguments
- `df::DataFrames.DataFrame`: Dataframe containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - mutant relative fitness values.
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - (log)-normal likelihood standard deviation.
- `n_ppc::Int`: Number of samples to generate per set of parameters.

## Optional Keyword Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the name of the variables
in the mcmc chain defining the following variables:
  - `:bc_mean_fitness`: Variable defining the inferred mutant fitness value
    `s⁽ᵐ⁾`.
  - `:bc_std_fitness`: Variable defining the standard defining the inferred
    standard deviation on the likelihood function `σ⁽ᵐ⁾`.
  - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
  multiple chain into a single column.

# Returns
- `log(fₜ₊₁ / fₜ) = s⁽ᵐ⁾ - s̅ₜ::Array{Float64}`: Evaluation of the frequency
  posterior predictive checks at all times for each MCMC sample.
"""
function logfreq_ratio_bc_ppc(
    df::DF.AbstractDataFrame,
    n_ppc::Int;
    param::Dict{Symbol,Symbol}=Dict(
        :bc_mean_fitness => :s⁽ᵐ⁾,
        :bc_std_fitness => :σ⁽ᵐ⁾,
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
                df[:, param[:bc_mean_fitness]] .- df[:, var],
                LinearAlgebra.Diagonal(
                    exp.(df[:, param[:bc_std_fitness]]) .^ 2
                )
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
    logfreq_ratio_bc_ppc(chain, n_ppc; kwargs)

Function to compute the **posterior predictive checks** for the barcode
log-frequency ratio for adaptive mutants. 

# Model

The functional form that connects the barcode frequency at time ``t+1`` with the
frequency at time ``t`` is of the form
```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. The statistical models in this package
assume that
```math
    \log\left(\frac{f_{t+1}^{(m)}}{f_{t}^{(m)}}\right) \sim 
    \mathcal{N}\left( s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
```
where ``\sigma^{(m)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution to produce the posterior
predictive checks.

# Arguments
- `chain::MCMCChains.Chains`: Chain containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - mutant relative fitness values.
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - (log)-normal likelihood standard deviation.
- `n_ppc::Int`: Number of samples to generate per set of parameters.

## Optional Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the name of the variables
in the mcmc chain defining the following variables:
  - `:bc_mean_fitness`: Variable defining the inferred mutant fitness value
    `s⁽ᵐ⁾`.
  - `:bc_std_fitness`: Variable defining the standard defining the inferred
    standard deviation on the likelihood function `σ⁽ᵐ⁾`.
  - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
  multiple chain into a single column.

# Returns
- `log(fₜ₊₁ / fₜ) = s⁽ᵐ⁾ - s̅ₜ::Array{Float64}`: Evaluation of the frequency
  posterior predictive checks at all times for each MCMC sample.
"""
function logfreq_ratio_bc_ppc(
    chain::MCMCChains.Chains,
    n_ppc::Int;
    param::Dict{Symbol,Symbol}=Dict(
        :bc_mean_fitness => :s⁽ᵐ⁾,
        :bc_std_fitness => :σ⁽ᵐ⁾,
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
                chain[param[:bc_mean_fitness]][:] .- chain[var][:],
                LinearAlgebra.Diagonal(
                    exp.(chain[param[:bc_std_fitness]][:]) .^ 2
                )
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
    logfreq_ratio_popmean_ppc(df, n_ppc; kwargs)

Function to compute the **posterior predictive checks** (better called the
posterior retrodictive checks) for the barcode log-frequency ratio for neutral
lineages. 

# Model

The functional form that connects the barcode frequency at tie ``t+1`` based on
the frequency at time ``t`` for **neutral barcodes** is of the form

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
- `df::DataFrames.DataFrame`: Dataframe containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - (log)-normal likelihood standard deviation.
- `n_ppc::Int`: Number of samples to generate per set of parameters.

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
function logfreq_ratio_popmean_ppc(
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
                LinearAlgebra.Diagonal(exp.(df[:, std_vars[i]]) .^ 2)
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
    logfreq_ratio_popmean_ppc(chain, n_ppc; kwargs)

Function to compute the **posterior predictive checks** (better called the
posterior retrodictive checks) for the barcode log-frequency ratio for neutral
lineages. 

# Model

The functional form that connects the barcode frequency at tie ``t+1`` based on
the frequency at time ``t`` for **neutral barcodes** is of the form

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
- `chain::MCMCChains.Chains`: Chain containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - mutant relative fitness values.
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - (log)-normal likelihood standard deviation.
- `n_ppc::Int`: Number of samples to generate per set of parameters.

## Optional Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the name of the variables
in the mcmc chain defining the following variables:
    - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
    - `population_std_fitness`: Common pattern in all standard deviations
        estimates for the likelihood.
- `model::Symbol=:lognormal`: Either `:normal` or `:lognormal` to indicate if
  the model used a normal or lognormal distribution for the likelihood. This is
  because when using a normal distribution, the nuisance parameters are sampled
  in log scale and need to be exponentiated.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
    multiple chain into a single column.

# Returns
- `log(fₜ₊₁ / fₜ) = s⁽ᵐ⁾ - s̅ₜ::Array{Float64}`: Evaluation of the log frequency
  ratio posterior predictive checks at all times for each MCMC sample.
"""
function logfreq_ratio_popmean_ppc(
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
                LinearAlgebra.Diagonal(exp.(chains[std_vars[i]][:]) .^ 2)
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
    logfreq_ratio_mutienv_ppc(df, n_ppc; kwargs)

Function to compute the **posterior predictive checks** for the barcode
log-frequency ratio for adaptive mutants. 

# Model

The functional form that connects the barcode frequency at time ``t+1`` with the
frequency at time ``t`` is of the form
```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. The statistical models in this package
assume that
```math
    \log\left(\frac{f_{t+1}^{(m)}}{f_{t}^{(m)}}\right) \sim 
    \mathcal{N}\left( s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
```
where ``\sigma^{(m)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution to produce the posterior
predictive checks.

# Arguments
- `df::DataFrames.DataFrame`: Dataframe containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - mutant relative fitness values.
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - (log)-normal likelihood standard deviation.
- `n_ppc::Int`: Number of samples to generate per set of parameters.
- `envs::Vector{<:Any}`: List of environments in experiment. This is used to
  index the corresponding fitness from the chain. NOTE: The list of environments
  should be the name or corresponding label of the environemnt; the index is
  generated internally.

## Optional Keyword Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the name of the variables
in the mcmc chain defining the following variables:
  - `:bc_mean_fitness`: Variable defining the inferred mutant fitness value
    `s⁽ᵐ⁾`.
  - `:bc_std_fitness`: Variable defining the standard defining the inferred
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
        :bc_mean_fitness => :s̲⁽ᵐ⁾,
        :bc_std_fitness => :σ̲⁽ᵐ⁾,
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
            occursin.(String(param[:bc_mean_fitness]), DF.names(df))
        ]
    )

    # Extract variable names for mutant relative fitness error
    σ_vars = sort(
        DF.names(df)[
            occursin.(String(param[:bc_std_fitness]), DF.names(df))
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
                LinearAlgebra.Diagonal(
                    exp.(df[:, σ_vars[env_idx[i+1]]]) .^ 2
                )
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
logfreq_ratio_mutienv_ppc(chain, n_ppc; kwargs)

Function to compute the **posterior predictive checks** (better called the
posterior retrodictive checks) for the barcode log-frequency ratio for neutral
lineages. 

# Model

The functional form that connects the barcode frequency at time ``t+1`` with the
frequency at time ``t`` is of the form
```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. The statistical models in this package
assume that
```math
    \log\left(\frac{f_{t+1}^{(m)}}{f_{t}^{(m)}}\right) \sim 
    \mathcal{N}\left( s^{(m)} - \bar{s}_t, \sigma^{(m)} \right),
```
where ``\sigma^{(m)}`` is the inferred standard deviation for the model. This
function generates samples out of this distribution to produce the posterior
predictive checks.

# Arguments
- `chain::MCMCChains.Chains`: Chain containing the MCMC samples for the
  variables needed to compute the posterior predictive checks. The dataframe
  should have MCMC samples for
  - mutant relative fitness values.
  - population mean fitness values. NOTE: The number of columns containing
    population mean fitness values determines the number of datapoints where the
    ppc are evaluated.
  - (log)-normal likelihood standard deviation.
- `n_ppc::Int`: Number of samples to generate per set of parameters.

## Optional Arguments
- `param::Dict{Symbol, Symbol}`: Dictionary indicating the name of the variables
in the mcmc chain defining the following variables:
    - `population_mean_fitness`: Common pattern in all population mean fitness
    variables.
    - `population_std_fitness`: Common pattern in all standard deviations
        estimates for the likelihood.
- `model::Symbol=:lognormal`: Either `:normal` or `:lognormal` to indicate if
  the model used a normal or lognormal distribution for the likelihood. This is
  because when using a normal distribution, the nuisance parameters are sampled
  in log scale and need to be exponentiated.
- `flatten::Bool=true`: Boolean indicating whether to flatten the output of
    multiple chain into a single column.

# Returns
- `log(fₜ₊₁ / fₜ) = s⁽ᵐ⁾ - s̅ₜ::Array{Float64}`: Evaluation of the log frequency
  ratio posterior predictive checks at all times for each MCMC sample.
"""
function logfreq_ratio_multienv_ppc(
    chain::MCMCChains.Chains,
    n_ppc::Int,
    envs::Vector{<:Any};
    param::Dict{Symbol,Symbol}=Dict(
        :bc_mean_fitness => :s̲⁽ᵐ⁾,
        :bc_std_fitness => :σ̲⁽ᵐ⁾,
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
        MCMCChains.namesingroup(chain, param[:bc_mean_fitness])
    )

    # Extract variable names for mutant relative fitness error
    σ_vars = sort(
        MCMCChains.namesingroup(chain, param[:bc_std_fitness])
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
                LinearAlgebra.Diagonal(
                    exp.(chain[σ_vars[env_idx[i+1]]][:]) .^ 2
                )
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
    naive_fitness(data; id_col, time_col, count_col, neutral_col, pseudocount)

Function to compute a naive estimate of mutant fitness data based on counts. The
fitness estimate is computed as

⟨log⁡(f⁽ᵐ⁾ₜ₊₁ / f⁽ᵐ⁾ₜ) - log⁡(f⁽ⁿ⁾ₜ₊₁ / f⁽ⁿ⁾ₜ))⟩ = s⁽ᵐ⁾

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
- `pseudocount::Int=1`: Pseudo count number to add to all counts. This is
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
    pseudocount::Int=1
)
    # Keep only the needed data to work with
    data = data[:, [id_col, time_col, count_col, neutral_col]]

    # Add pseudo-count to each barcode to avoid division by zero
    data[:, count_col] .+= pseudocount

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
# Computing naive parameter priors based on neutrals data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    naive_prior(data; kwargs)

Function to compute a naive set of parameters for the prior distributions of the
population mean fitness `s̲ₜ` values, the nuisance parameters in the
log-likelihood functions for the frequency ratios `logσ̲ₜ`, and the log of the
Poisson parameters for the observation model `logΛ̲̲`

This function expects the data in a **tidy** format. This means that every row
represents **a single observation**. For example, if we measure barcode `i` in 4
different time points, each of these four measurements gets an individual row.
Furthermore, measurements of barcode `j` over time also get their own individual
rows.
        
The `DataFrame` must contain at least the following columns:
- `id_col`: Column identifying the ID of the barcode. This can the barcode
    sequence, for example.
- `time_col`: Column defining the measurement time point.
- `count_col`: Column with the raw barcode count.
- `neutral_col`: Column indicating whether the barcode is from a neutral lineage
or not.
- `rep_col`: (Optional) For hierarchical models to be build with multiple
  experimental replicates, this column defines which observations belong to
  which replicate.

# Arguments
- `data::DataFrames.AbstractDataFrame`: **Tidy dataframe** with the data to be
used to sample from the population mean fitness posterior distribution.

## Optional Keyword Arguments
- `id_col::Symbol=:barcode`: Name of the column in `data` containing the barcode
    identifier. The column may contain any type of entry.
- `time_col::Symbol=:time`: Name of the column in `data` defining the time point
    at which measurements were done. The column may contain any type of entry as
    long as `sort` will resulted in time-ordered names.
- `count_col::Symbol=:count`: Name of the column in `data` containing raw counts
    per barcode. The column must contain entries of type `<: Int`.
- `neutral_col::Symbol=:neutral`: Name of the column in `data` defining whether
    the barcode belongs to a neutral lineage or not. The column must contain
    entries of type `Bool`.
- `rep_col::Union{Nothing,Symbol}=nothing`: (Optional) Column indicating the
  experimental replicates each point belongs to.
- `pseudocount::Int=1`: Pseudo counts to add to raw counts to avoid dividing by
  zero. This is useful if some of the barcodes go extinct.

# Returns
- `prior_params::Dict`: Dictionary with **two** entries:
    - `s_pop_prior`: **Mean** value of the population mean fitness. **NOTE**:
      This naive empirical method cannot make statements about the expected
      standard deviation of the population mean fitness. It is up to the
      researcher to determine this value.
    - `logσ_pop_prior`: **Mean** value on the nuisance parameter for the
      log-likelihood functions on the log-frequency ratios. In other words, the
      mean for the (log)-Normal distribution on the frequency ratios. **NOTE**:
      This naive empirical method cannot make statements about the expected
      standard deviation of the population mean fitness. It is up to the
      researcher to determine this value. **NOTE**: Typically, one can use the
      same estimate for both the neutral and the mutant lineages.
    - `logλ_prior`: **Mean** value of the nuisance parameter for the Poisson
      observation model parameter. **NOTE**: This naive empirical method cannot
      make statements about the expected standard deviation of the population
      mean fitness. It is up to the researcher to determine this value.
"""
function naive_prior(
    data::DF.AbstractDataFrame;
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rep_col::Union{Nothing,Symbol}=nothing,
    pseudocount::Int=1,
)
    # Add pseudocount to count column
    data[:, count_col] = data[:, count_col] .+ pseudocount

    # Convert data to arrays
    data_mats = data_to_arrays(
        data;
        id_col=id_col,
        time_col=time_col,
        count_col=count_col,
        neutral_col=neutral_col,
        rep_col=rep_col,
    )

    if typeof(rep_col) <: Nothing
        # Compute frequencies
        bc_freq = data_mats.bc_count ./ data_mats.bc_total

        # Extract neutral lineages frequencies
        neutral_freq = bc_freq[:, 1:data_mats.n_neutral]

        # Compute log-frequency ratios
        neutral_logfreq = log.(
            neutral_freq[2:end, :] ./ neutral_freq[1:end-1, :]
        )
    elseif typeof(rep_col) <: Symbol
        # Check if all replicates had the same number of time points
        if typeof(data_mats.bc_count) <: Array{<:Int,3}
            # Initialize array to save frequencies
            freqs = Array{Float64}(undef, size(data_mats.bc_count)...)

            # Compute frequencies
            freqlist = [
                x ./ data_mats.bc_total
                for x in eachslice(data_mats.bc_count; dims=2)
            ]

            # Loop through each slice of freqs
            for (i, freq) in enumerate(freqlist)
                freqs[:, i, :] = freq
            end # for

            # Assign frequencies
            bc_freq = freqs

            # Extract neutral lineages frequencies
            neutral_freq = bc_freq[:, 1:data_mats.n_neutral, :]
            # Compute log-frequency ratios
            neutral_logfreq = log.(
                neutral_freq[2:end, :, :] ./ neutral_freq[1:end-1, :, :]
            )
        elseif typeof(data_mats.bc_count) <: Vector{<:Matrix{<:Int}}
            # Define number of replicates
            global n_rep = data_mats.n_rep

            # Compute frequencies
            bc_freq = [
                data_mats.bc_count[rep] ./ data_mats.bc_total[rep]
                for rep in 1:n_rep
            ]

            # Extract neutral lineages frequencies
            neutral_freq = [
                bc_freq[rep][:, 1:data_mats.n_neutral]
                for rep = 1:n_rep
            ]
            # Compute log-frequency ratios
            neutral_logfreq = [
                log.(
                    neutral_freq[rep][2:end, :] ./
                    neutral_freq[rep][1:end-1, :]
                )
                for rep = 1:n_rep
            ]
        end # if
    end # if

    # ========== Population mean fitness prior ========== #  

    # Compute mean per time point for approximate mean fitness making sure we to
    # remove infinities.
    if typeof(rep_col) <: Nothing
        logfreq_mean = StatsBase.mean.(
            [x[.!isinf.(x)] for x in eachrow(neutral_logfreq)]
        )
    elseif typeof(rep_col) <: Symbol
        # Check if all replicates have the same number of time points
        if typeof(data_mats.bc_count) <: Array{<:Int,3}
            # Initialize array to save means
            logfreq_mean = Matrix{Float64}(
                undef, size(neutral_logfreq)[1], size(neutral_logfreq)[3]
            )

            # Loop through time points
            for i = 1:size(neutral_logfreq)[1]
                # Loop through replicates
                for k = 1:size(neutral_logfreq)[3]
                    logfreq_mean[i, k] = StatsBase.mean(
                        neutral_logfreq[i, :, k][
                            .!isinf.(neutral_logfreq[i, :, k])
                        ]
                    )
                end # for
            end # for
        elseif typeof(data_mats.bc_count) <: Vector{<:Matrix{<:Int}}
            # Compute mean per time point per replicate
            logfreq_mean = vcat([
                StatsBase.mean.(
                    [x[.!isinf.(x)] for x in eachrow(nlf)]
                ) for nlf in neutral_logfreq
            ]...)
        end # if
    end # if

    # Define prior for population mean fitness.
    s_pop_prior = -logfreq_mean[:]

    # ========== Nuisance log-likelihood parameter priors ========== #  

    # Compute mean per time point for approximate mean fitness making sure we to
    # remove infinities.
    if typeof(rep_col) <: Nothing
        logfreq_std = StatsBase.std.(
            [x[.!isinf.(x)] for x in eachrow(neutral_logfreq)]
        )
    elseif typeof(rep_col) <: Symbol
        # Check if all replicates have the same number of time points
        if typeof(data_mats.bc_count) <: Array{<:Int,3}
            # Initialize array to save means
            logfreq_std = Matrix{Float64}(
                undef, size(neutral_logfreq)[1], size(neutral_logfreq)[3]
            )

            # Loop through time points
            for i = 1:size(neutral_logfreq)[1]
                # Loop through replicates
                for k = 1:size(neutral_logfreq)[3]
                    logfreq_std[i, k] = StatsBase.std(
                        neutral_logfreq[i, :, k][
                            .!isinf.(neutral_logfreq[i, :, k])
                        ]
                    )
                end # for
            end # for
        elseif typeof(data_mats.bc_count) <: Vector{<:Matrix{<:Int}}
            # Compute mean per time point per replicate
            logfreq_std = vcat([
                StatsBase.std.(
                    [x[.!isinf.(x)] for x in eachrow(nlf)]
                ) for nlf in neutral_logfreq
            ]...)
        end # if
    end # if

    # Define prior for population mean fitness.
    logσ_pop_prior = -logfreq_std[:]


    #== Nuisance parameter for the Poisson–distribution observational model ==#

    # Compute mean per time point for approximate mean fitness making sure we to
    # remove infinities.
    if (typeof(rep_col) <: Nothing) |
       (typeof(data_mats.bc_count) <: Array{Int64,3})
        logλ_prior = log.(data_mats.bc_count)[:]
    elseif typeof(data_mats.bc_count) <: Vector{<:Matrix{<:Int}}
        logλ_prior = vcat(
            [log.(data_mats.bc_count[rep])[:] for rep = 1:n_rep]...
        )
    end # if

    return Dict(
        :s_pop_prior => s_pop_prior,
        :logσ_pop_prior => logσ_pop_prior,
        :logλ_prior => logλ_prior,
    )

end # function