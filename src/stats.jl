##
# Import package to handle dataframes
import DataFrames as DF

# Import statistical libraries
import Distributions
import StatsBase

# Import library to handle MCMC chains
import MCMCChains

##

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Posterior predictive checks
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    freq_mutant_ppc(chain, varname_mut, varname_mean)

Function to compute the **posterior predictive checks** for the barcode
frequency for adaptive mutants. Our model predicts the frequency at time ``t+1``
based on the frequency at time ``t`` as

```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. This funciton computes the frequency
for each of the MCMC samples in the `chain` object.

# Arguments
- `chain::MCMCChains.Chains`: `Turing.jl` MCMC chain for the fitness of a single
  mutant.

## Optional arguments
- `varname_mut::Symbol=Symbol("s⁽ᵐ⁾")`: Variable name for the mutant relative fitness
    in the `chain` object.
- `varname_mean::Symbol=Symbol("s̲ₜ")`: Variable name for *all* population mean
  fitness.
- `freq_mut::Symbol=Symbol("f̲⁽ᵐ⁾")`: Variable name for *all* mutant barcode
  frequencies.

# Returns
- `fₜ₊₁ = fₜ × exp(s⁽ᵐ⁾ - s̅ₜ)::Array{Float64}`: Evaluation of the frequency
  posterior predictive checks at all times for each MCMC sample. The dimensions
  of the output are (n_samples × n_time × n_chains)
"""
function freq_mutant_ppc(
    chain::MCMCChains.Chains;
    varname_mut::Symbol=Symbol("s⁽ᵐ⁾"),
    varname_mean::Symbol=Symbol("s̲ₜ"),
    freq_mut::Symbol=Symbol("f̲⁽ᵐ⁾")
)
    # Extract number of chains
    n_chains = length(MCMCChains.chains(chain))
    # Extract number of steps per chain
    n_samples = length(chain)
    # Extract number of timepoints
    n_time = size(MCMCChains.group(chain, varname_mean), 2)

    # Extract relevant parameters
    # 1. Mutant fitness
    s⁽ᵐ⁾_chain = first(MCMCChains.get(chain, varname_mut))

    # 2. Population mean fitness chain. This is collected into a Tensor where
    #    each face are the samples corresponding to a time point.
    # Initialize array to save chain   
    sₜ_chain = Array{Float64}(undef, n_samples, n_chains, n_time)
    # Extract and sort chain variable names for population mean fitness
    sₜ_names = sort(
        collect(
            keys(MCMCChains.get(chain, MCMCChains.namesingroup(chain, varname_mean)))
        )
    )
    # Loop through time points
    for (i, s) in enumerate(sₜ_names)
        # Store chain
        sₜ_chain[:, :, i] = first(MCMCChains.get(chain, s))
    end # for

    # 3. Fitnes at time zero
    f₀⁽ᵐ⁾ = first(MCMCChains.get(chain, Symbol(String(freq_mut) * "[1]")))

    # Compute frequency ratios according to model
    γ⁽ᵐ⁾ = exp.(s⁽ᵐ⁾_chain .- sₜ_chain)

    # Initialize array to save frequencies
    f̲⁽ᵐ⁾ = Array{Float64}(undef, n_samples, n_chains, n_time + 1)
    # Save first column as the time zero frequency
    f̲⁽ᵐ⁾[:, :, 1] = f₀⁽ᵐ⁾

    # Loop through time points
    for t = 2:(n_time+1)
        # Compute fitness
        f̲⁽ᵐ⁾[:, :, t] = f̲⁽ᵐ⁾[:, :, t-1] .* γ⁽ᵐ⁾[:, :, t-1]
    end # for

    return permutedims(f̲⁽ᵐ⁾, [1, 3, 2])
end # function

@doc raw"""
    freq_mut_ppc_quantile(quantile, chain, varname_mut, varname_mean, freq_mut)

Function to compute the **posterior predictive checks** quantiles for the
barcode frequency for adaptive mutants. Our model predicts the frequency at time
``t+1`` based on the frequency at time ``t`` as
    
```math
    f_{t+1}^{(m)} = f_{t}^{(m)} 
    \exp\left[ \left( s^{(m)} - \bar{s}_t \right) \tau \right],
```
where ``s^{(m)}`` is the mutant relative fitness, ``\bar{s}_t`` is the
population mean fitness between time ``t`` and ``t+1``, and ``\tau`` is the time
interval between time ``t`` and ``t+1``. This funciton computes the frequency
for each of the MCMC samples in the `chain` object, and then extracts the
quantiles from these posterior predictive checks.

# Arguments
- `quantile::Vector{<:AbstractFloat}`: List of quantiles to extract from the
  posterior predictive checks.
- `chain::MCMCChains.Chains`: `Turing.jl` MCMC chain for the fitness of a single
    mutant.

## Optional arguments
- `varname_mut::Symbol=Symbol("s⁽ᵐ⁾")`: Variable name for the mutant relative fitness
    in the `chain` object.
- `varname_mean::Symbol=Symbol("s̲ₜ")`: Variable name for *all* population mean
    fitness.
- `freq_mut::Symbol=Symbol("f̲⁽ᵐ⁾")`: Variable name for *all* mutant barcode
    frequencies.

# Returns
- `fₜ₊₁ = fₜ × exp(s⁽ᵐ⁾ - s̅ₜ)::Array{Float64}`: Evaluation of the frequency
  posterior predictive check quantiles at all times for each MCMC sample.
"""
function freq_mutant_ppc_quantile(
    quantile::Vector{<:AbstractFloat},
    chain::MCMCChains.Chains;
    varname_mut::Symbol=Symbol("s⁽ᵐ⁾"),
    varname_mean::Symbol=Symbol("s̲ₜ"),
    freq_mut::Symbol=Symbol("f̲⁽ᵐ⁾")
)
    # Check that all quantiles are within bounds
    if any(.![0.0 ≤ x ≤ 1 for x in quantile])
        error("All quantiles must be between zero and one")
    end # if

    # Compute posterior predictive checks for a particular chain
    f_ppc = freq_mutant_ppc(chain; varname_mut=varname_mut, varname_mean=varname_mean, freq_mut=freq_mut)

    # Compact multiple chains into single long chain
    f_ppc = vcat([f_ppc[:, :, i] for i = 1:size(f_ppc, 3)]...)

    # Initialize matrix to save quantiles
    f_quant = Array{Float64}(undef, size(f_ppc, 2), length(quantile), 2)

    # Loop through quantile
    for (i, q) in enumerate(quantile)
        # Lower bound
        f_quant[:, i, 1] = StatsBase.quantile.(
            eachcol(f_ppc), (1.0 - q) / 2.0
        )
        # Upper bound
        f_quant[:, i, 2] = StatsBase.quantile.(
            eachcol(f_ppc), 1.0 - (1.0 - q) / 2.0
        )
    end # for

    return f_quant
end # function

@doc raw"""
    logfreqratio_neutral_ppc_quantile(quantile, df)

Function to compute the **posterior predictive checks** for the barcode log
frequency ratio for neutral lineages. Our model predicts the frequency for
neutral lineages at time ``t+1`` based on the frequency at time ``t`` as

```math
    f_{t+1}^{(n)} = f_{t}^{(n)} 
    \exp\left[  - \bar{s}_t \tau \right],
```
where ``\bar{s}_t`` is the population mean fitness between time ``t`` and
``t+1``, and ``\tau`` is the time interval between time ``t`` and ``t+1``.
Solving for the mean fitness results in
```math
    \frac{1}{\tau} \log \frac{f_{t+1}^{(n)}}{f_{t}^{(n)}} = - \bar{s}_t.
```

This function computes the quantiles of the log frequency ration for the neutral
lineages, given the MCMC samples of the population mean fitness.

# Arguments


# Returns
- `log(fₜ₊₁ / fₜ) = - s̅ₜ::Array{Float64}`: Evaluation of the log frequency ratio
  posterior predictive checks at all times for each MCMC sample. The dimensions
  of the output are (n_samples × n_time × n_chains)
"""
function logfreqratio_neutral_ppc_quantile(
    quantile::Vector{<:AbstractFloat},
    chain::MCMCChains.Chains;
    varname_mean::Symbol=Symbol("sₜ")
)
    # Check that all quantiles are within bounds
    if any(.![0.0 ≤ x ≤ 1 for x in quantile])
        error("All quantiles must be between zero and one")
    end # if

    # Extract variable names
    varnames = MCMCChains.names(chain)

    # Check that names match pattern given by varname_mean
    if any(.!occursin.(String(varname_mean), String.(varnames)))
        error("The name of the variables in the chain do not match the pattern in varname_mean")
    end # if

    # Extract values
    logfreqratio = [vec(Matrix(chain[n])) for n in varnames]




    # Extract number of chains
    n_chains = length(MCMCChains.chains(chain))
    # Extract number of steps per chain
    n_samples = length(chain)


end # function