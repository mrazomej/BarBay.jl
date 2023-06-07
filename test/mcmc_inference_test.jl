#

# Activate environment
@load_pkg(".")

# Import package to revise package
import Revise

# Import library package
import BayesFitness

# Import basic mathematical functions
import LinearAlgebra
import Distributions
import Random

# Import MCMC-related packages
import Turing
using ReverseDiff
using Zygote
import MCMCChains
using DynamicHMC

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

Turing.@model function fitness_lognormal(
    R̲̲::Matrix{Int64},
    R̲̲⁽ⁿ⁾::Matrix{Int64},
    R̲̲⁽ᵐ⁾::Matrix{Int64},
    n̲ₜ::Vector{Int64};
    s_pop_prior::Vector{Float64}=[0.0, 2.0],
    σ_pop_prior::Vector{Float64}=[0.0, 1.0],
    s_mut_prior::Vector{Float64}=[0.0, 2.0],
    σ_mut_prior::Vector{Float64}=[0.0, 1.0],
    λ_prior::Vector{Float64}=[3.0, 3.0]
)
    # Prior on population mean fitness
    s̲ₜ ~ Turing.MvNormal(
        repeat([s_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* s_pop_prior[2] .^ 2
    )
    # Prior on LogNormal error σ̲ₜ
    σ̲ₜ ~ Turing.MvLogNormal(
        repeat([σ_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* σ_pop_prior[2] .^ 2
    )

    # Prior on mutant fitness s̲⁽ᵐ⁾
    s̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([s_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* s_mut_prior[2] .^ 2
    )
    # Prior on LogNormal error σ̲⁽ᵐ⁾ 
    σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
        repeat([σ_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* σ_mut_prior[2] .^ 2
    )

    # Prior on λ parameters
    Λ̲̲ ~ Turing.MvLogNormal(
        repeat([λ_prior[1]], size(R̲̲⁽ⁿ⁾, 1) * (size(R̲̲⁽ⁿ⁾, 2) + size(R̲̲⁽ᵐ⁾, 2))),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) * (size(R̲̲⁽ⁿ⁾, 2) + size(R̲̲⁽ᵐ⁾, 2))) .*
        λ_prior[2]^2
    )

    # Reshape λ parameters to fit the matrix format
    Λ̲̲ = reshape(Λ̲̲, size(R̲̲⁽ⁿ⁾, 1), (size(R̲̲⁽ⁿ⁾, 2) + size(R̲̲⁽ᵐ⁾, 2)))

    # Compute frequencies πₜ
    F̲̲ = Λ̲̲ ./ sum(Λ̲̲, dims=2)

    # Compute frequency ratios γₜ
    Γ̲̲ = F̲̲[2:end, :] ./ F̲̲[1:end-1, :]
    # Split neutral and mutant frequency ratios
    Γ̲̲⁽ⁿ⁾ = @view Γ̲̲[:, 1:size(R̲̲⁽ⁿ⁾, 2)]
    Γ̲̲⁽ᵐ⁾ = @view Γ̲̲[:, size(R̲̲⁽ⁿ⁾, 2)+1:end]

    # Prob nₜ | λ̲ₜ
    n̲ₜ ~ Turing.arraydist([Turing.Poisson(sum(Λ̲̲[t, :])) for t = 1:size(R̲̲⁽ⁿ⁾, 1)])

    # Loop through time
    for t = 1:size(R̲̲⁽ⁿ⁾, 1)
        # Prob of reads given parameters R̲ₜ | nₜ, f̲ₜ.
        R̲̲[t, :] ~ Turing.Multinomial(n̲ₜ[t], F̲̲[t, :])
    end # for


    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            -1 .* repeat(s̲ₜ, size(Γ̲̲⁽ⁿ⁾, 2)),
            LinearAlgebra.Diagonal(repeat(σ̲ₜ .^ 2, size(Γ̲̲⁽ⁿ⁾, 2)))
        ),
        Γ̲̲⁽ⁿ⁾[:]
    )

    # Sample posterior for frequency ratio. Since it is a sample over a
    # generated quantity, we must use the @addlogprob! macro
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

time_col = :time
id_col = :barcode
count_col = :count

# Re-extract unique time points
timepoints = sort(unique(data[:, time_col]))

# Group data by unique mutant barcode
data_group = DF.groupby(data[.!data[:, :neutral], :], id_col)

# Check that all barcodes were measured at all points
if any([size(d, 1) for d in data_group] .!= length(timepoints))
    error("Not all barcodes have reported counts in all time points")
end # if

# Extract keys
data_keys = [k[String(id_col)] for k in keys(data_group)]

# Extract total number of barcodes per timepoint
R_tot = DF.combine(DF.groupby(data, time_col), count_col => sum)
# Sort dataframe by time
DF.sort!(R_tot, time_col)
# Extract sorted counts
R̲ = R_tot[:, Symbol(String(count_col) * "_sum")]

# Initialize array to save counts for each mutant at time t
R_mut = Matrix{Int64}(
    undef, length(timepoints), length(data_group)
)

# Loop through each unique barcode
for (i, d) in enumerate(data_group)
    # Sort data by timepoint
    DF.sort!(d, time_col)
    # Extract data
    R_mut[:, i] = d[:, count_col]
end # for

# R_mut = R_mut[:, 1:10]


##

# Group data by unique mutant barcode
data_group = DF.groupby(data[data[:, :neutral], :], id_col)

# Check that all barcodes were measured at all points
if any([size(d, 1) for d in data_group] .!= length(timepoints))
    error("Not all barcodes have reported counts in all time points")
end # if

# Extract keys
data_keys = [k[String(id_col)] for k in keys(data_group)]

# Extract total number of barcodes per timepoint
R_tot = DF.combine(DF.groupby(data, time_col), count_col => sum)
# Sort dataframe by time
DF.sort!(R_tot, time_col)
# Extract sorted counts
R̲ = R_tot[:, Symbol(String(count_col) * "_sum")]

# Initialize array to save counts for each mutant at time t
R_neut = Matrix{Int64}(
    undef, length(timepoints), length(data_group)
)

# Loop through each unique barcode
for (i, d) in enumerate(data_group)
    # Sort data by timepoint
    DF.sort!(d, time_col)
    # Extract data
    R_neut[:, i] = d[:, count_col]
end # for

##

# Concatenate data
R_tot = hcat(R_neut, R_mut)

# Compute total counts for each run
nₜ = vec(sum(R_tot, dims=2))

##

println("Initializing MCMC sampling...\n")

# Set AutoDiff backend to ReverseDiff.jl for faster computation
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

# Sample chain
@time chain = Turing.sample(
    fitness_lognormal(R_tot, R_neut, R_mut, nₜ),
    Turing.NUTS(0.65),
    Turing.MCMCThreads(),
    1_000,
    4
)

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
ppc_mat = BayesFitness.stats.logfreq_ratio_mean_ppc(
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
BayesFitness.viz.ppc_time_series!(
    ax, qs, ppc_mat; colors=colors
)

# Add plot for median (we use the 5 percentile to have a "thicker" line showing
# the median)
BayesFitness.viz.ppc_time_series!(
    ax, [0.05], ppc_mat; colors=ColorSchemes.Blues_9[end:end]
)

# Plot log-frequency ratio of neutrals
BayesFitness.viz.logfreq_ratio_time_series!(
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
ppc_mat = BayesFitness.stats.logfreq_ratio_mutant_ppc(
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
BayesFitness.viz.ppc_time_series!(
    ax, qs, ppc_mat; colors=colors
)

# Add plot for median (we use the 5 percentile to have a "thicker" line showing
# the median)
BayesFitness.viz.ppc_time_series!(
    ax, [0.05], ppc_mat; colors=ColorSchemes.Blues_9[end:end]
)

# Add scatter of data
scatterlines!(ax, diff(log.(data_bc.freq)), color=:black)

fig