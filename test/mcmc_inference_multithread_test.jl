#

# Activate environment
@load_pkg(".")

# Import package to revise package
import Revise

# Import library package
import BarBay

# Import basic mathematical functions
import LinearAlgebra
import Distributions
import Random

# Import MCMC-related packages
import Turing
using ReverseDiff
import MCMCChains

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import library to save and load native julia objects
import JLD2

# Import library to list files
import Glob

# Import plotting libraries
using CairoMakie
import ColorSchemes

Random.seed!(42)
##

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
  barcode count trajectory for a single lineage. **NOTE**: This matrix **must**
  be equivalent to `hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾)`. The reason it is an independent input
  parameter is to avoid the `hcat` computation within the `Turing` model.
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
    λ_prior::Vector{Float64}=[3.0, 3.0]
)
    ## %%%%%%%%%%%%%% Population mean fitness  %%%%%%%%%%%%%% ##

    # Prior on population mean fitness P(s̲ₜ)
    s̲ₜ ~ Turing.MvNormal(
        repeat([s_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* s_pop_prior[2] .^ 2
    )
    # Prior on LogNormal error P(σ̲ₜ)
    σ̲ₜ ~ Turing.MvLogNormal(
        repeat([σ_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* σ_pop_prior[2] .^ 2
    )

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Prior on mutant fitness P(s̲⁽ᵐ⁾)
    s̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([s_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* s_mut_prior[2] .^ 2
    )
    # Prior on LogNormal error P(σ̲⁽ᵐ⁾)
    σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
        repeat([σ_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* σ_mut_prior[2] .^ 2
    )


    ## %%%%%%%%%%%%%% Barcode frequencies %%%%%%%%%%%%%% ##

    # Prior on Poisson distribtion parameters P(λ)
    Λ̲̲ ~ Turing.MvLogNormal(
        repeat([λ_prior[1]], length(R̲̲)),
        LinearAlgebra.I(length(R̲̲)) .*
        λ_prior[2]^2
    )

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
    Γ̲̲⁽ᵐ⁾ = @view Γ̲̲[:, size(R̲̲⁽ⁿ⁾, 2)+1:end]

    # Prob of total number of barcodes read given the Poisosn distribution
    # parameters P(nₜ | λ̲ₜ)
    n̲ₜ ~ Turing.arraydist([Turing.Poisson(sum(Λ̲̲[t, :])) for t = 1:size(R̲̲⁽ⁿ⁾, 1)])

    # Loop through time points
    for t = 1:size(R̲̲⁽ⁿ⁾, 1)
        # Prob of reads given parameters P(R̲ₜ | nₜ, f̲ₜ). Note: We add the
        # check_args=false option to avoid the recurrent problem of
        # > Multinomial: p is not a probability vector.
        # due to rounding errors
        R̲̲[t, :] ~ Turing.Multinomial(n̲ₜ[t], F̲̲[t, :]; check_args=false)
    end # for

    ## %%%%%%%%%%%%%% Log-Likelihood functions %%%%%%%%%%%%%% ##

    # Sample posterior for neutral lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # P(γₜ⁽ⁿ⁾| sₜ, σₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            -1.0 .* repeat(s̲ₜ, size(Γ̲̲⁽ⁿ⁾, 2)),
            LinearAlgebra.Diagonal(repeat(σ̲ₜ .^ 2, size(Γ̲̲⁽ⁿ⁾, 2)))
        ),
        Γ̲̲⁽ⁿ⁾[:]
    )

    # Sample posterior for nutant lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # P(γₜ⁽ᵐ⁾ | s⁽ᵐ⁾, σ⁽ᵐ⁾, s̲ₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            # Build vector for fitness differences
            vcat([s⁽ᵐ⁾ .- s̲ₜ for s⁽ᵐ⁾ in s̲⁽ᵐ⁾]...),
            # Build vector for variances
            vcat([repeat([σ], length(s̲ₜ)) for σ in σ̲⁽ᵐ⁾]...)
        ),
        Γ̲̲⁽ᵐ⁾[:]
    )
    return
end # @model function


##

println("Loading data...\n")
# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

##

# Define relevant column names
time_col = :time
id_col = :barcode
count_col = :count
neutral_col = :neutral

# %%%%%%%%%%% Neutral barcodes data %%%%%%%%%%% ##

# Re-extract unique time points
timepoints = sort(unique(data[:, time_col]))

# Group data by unique mutant barcode
data_group = DF.groupby(data[data[:, neutral_col], :], id_col)

# Check that all barcodes were measured at all points
if any([size(d, 1) for d in data_group] .!= length(timepoints))
    error("Not all neutral barcodes have reported counts in all time points")
end # if

# Extract keys
data_keys = [k[String(id_col)] for k in keys(data_group)]

# Initialize array to save counts for each mutant at time t
R̲̲⁽ⁿ⁾ = Matrix{Int64}(
    undef, length(timepoints), length(data_group)
)

# Loop through each unique barcode
for (i, d) in enumerate(data_group)
    # Sort data by timepoint
    DF.sort!(d, time_col)
    # Extract data
    R̲̲⁽ⁿ⁾[:, i] = d[:, count_col]
end # for

## %%%%%%%%%%% Mutant barcodes data %%%%%%%%%%% ##

# Group data by unique mutant barcode
data_group = DF.groupby(data[.!data[:, neutral_col], :], id_col)

# Check that all barcodes were measured at all points
if any([size(d, 1) for d in data_group] .!= length(timepoints))
    error("Not all mutant barcodes have reported counts in all time points")
end # if

# Extract keys
data_keys = [k[String(id_col)] for k in keys(data_group)]

# Initialize array to save counts for each mutant at time t
R̲̲⁽ᵐ⁾ = Matrix{Int64}(
    undef, length(timepoints), length(data_group)
)

# Loop through each unique barcode
for (i, d) in enumerate(data_group)
    # Sort data by timepoint
    DF.sort!(d, time_col)
    # Extract data
    R̲̲⁽ᵐ⁾[:, i] = d[:, count_col]
end # for

## %%%%%%%%%%% Total barcodes data %%%%%%%%%%% ##

# Concatenate neutral and mutant data matrices
R̲̲ = hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾)

# Compute total counts for each run
n̲ₜ = vec(sum(R̲̲, dims=2))

##

println("Initializing MCMC sampling...\n")

# Set AutoDiff backend to ReverseDiff.jl for faster computation
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

multithread = false

if multithread
    # Sample chain
    chain = Turing.sample(
        fitness_lognormal(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲, n̲ₜ; Dict()...),
        Turing.NUTS(0.65),
        Turing.MCMCThreads(),
        10,
        4,
        progress=true
    )
else
    chain = mapreduce(
        c -> Turing.sample(
            fitness_lognormal(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲, n̲ₜ; Dict()...),
            Turing.NUTS(0.65),
            15,
            progress=true
        ),
        Turing.chainscat,
        1:1
    )
end # if

##

# Write output into memory
JLD2.jldsave(
    "./output/data_example_01_1000steps_4walkers.jld2", chain=chain
)

##

# Load chain
chain = JLD2.load("./output/data_example_01_1000steps_4walkers.jld2")["chain"]

##

# Name variables to be extracted from chains
chain_vars = [:s̲ₜ, :σ̲ₜ]

# Locate variable names to extract from chain
chain_names = reduce(
    vcat, [MCMCChains.namesingroup(chain, var) for var in chain_vars]
)

# Extract chain variables
chn = chain[chain_names]

# Define number of posterior predictive check samples
n_ppc = 5_000

# Define dictionary with corresponding parameters for variables needed for the
# posterior predictive checks
param = Dict(
    :population_mean_fitness => :s̲ₜ,
    :population_std_fitness => :σ̲ₜ,
)

# Compute posterior predictive checks
ppc_mat = BarBay.stats.logfreq_ratio_mean_ppc(
    chain, n_ppc; param=param
)

##

##

# Initialize figure
fig = Figure(resolution=(450, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="ln(fₜ₊₁/fₜ)",
    title="log-frequency ratio PPC"
)

# Define quantiles to compute
qs = [0.68, 0.95, 0.997]

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 0.75, length(qs)))

# Plot posterior predictive checks
BarBay.viz.ppc_time_series!(
    ax, qs, ppc_mat; colors=colors
)

# Add plot for median (we use the 5 percentile to have a "thicker" line showing
# the median)
BarBay.viz.ppc_time_series!(
    ax, [0.05], ppc_mat; colors=ColorSchemes.Blues_9[end:end]
)

# Plot log-frequency ratio of neutrals
BarBay.viz.logfreq_ratio_time_series!(
    ax,
    data[data.neutral, :];
    freq_col=:freq,
    color=:black,
    alpha=1.0,
    linewidth=2
)

fig

##

# Find barcode with maximum count
bc = data[first(argmax(data.count, dims=1)), :barcode]

# Extract data for barcode example
data_bc = data[data.barcode.==bc, :]

# Sort data by time
DF.sort!(data_bc, :time)

# Group data by unique mutant barcode
data_group = DF.groupby(data[.!data[:, :neutral], :], id_col)

# List group keys
bc_group = first.(values.(keys(data_group)))

# Find barcode index to extract chain
bc_idx = findfirst(bc_group .== bc)

##

# Name variables to be extracted from chains
chain_vars = [Symbol("s̲⁽ᵐ⁾[$(bc_idx)]"), Symbol("σ̲⁽ᵐ⁾[$(bc_idx)]"), :s̲ₜ]

# Locate variable names to extract from chain
chain_names = reduce(
    vcat, [MCMCChains.namesingroup(chain, var) for var in chain_vars]
)

# Extract chain variables
chn = chain[chain_names]

##

# Define number of posterior predictive check samples
n_ppc = 5_000

# Define dictionary with corresponding parameters for variables needed for the
# posterior predictive checks
param = Dict(
    :mutant_mean_fitness => Symbol("s̲⁽ᵐ⁾[$(bc_idx)]"),
    :mutant_std_fitness => Symbol("σ̲⁽ᵐ⁾[$(bc_idx)]"),
    :population_mean_fitness => :s̲ₜ,
)

# Compute posterior predictive checks
ppc_mat = BarBay.stats.logfreq_ratio_mutant_ppc(
    chn, n_ppc; param=param
)

##

# Initialize figure
fig = Figure(resolution=(450, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="ln(fₜ₊₁/fₜ)",
    title="log-frequency ratio PPC"
)

# Define quantiles to compute
qs = [0.95, 0.675]

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.5, 0.75, length(qs)))

# Plot posterior predictive checks
BarBay.viz.ppc_time_series!(
    ax, qs, ppc_mat; colors=colors
)

# Add plot for median (we use the 5 percentile to have a "thicker" line showing
# the median)
BarBay.viz.ppc_time_series!(
    ax, [0.05], ppc_mat; colors=ColorSchemes.Blues_9[end:end]
)

# Add scatter of data
scatterlines!(ax, diff(log.(data_bc.freq)), color=:black)

fig