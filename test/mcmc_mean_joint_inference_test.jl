# Activate environment
@load_pkg(".")

# Import package to revise package
import Revise

# Import library package
import BarBay

# Import libraries to manipulate data
import DataFrames as DF
import CSV
import MCMCChains

# Import library to set random seed
import Random

# Import library to save and load native julia objects
import JLD2

# Import plotting libraries
using CairoMakie
import ColorSchemes

# Import library to perform Bayesian inference
import Turing
# Import AutoDiff backend
using ReverseDiff

Random.seed!(42)

##

# Set AutoDiff backend
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)


##

println("Loading data...\n")
# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

# Count number of unique neutral barcodes
n_barcode = length(unique(data[data.neutral, :barcode]))
# Define number of time points
n_time = length(unique(data.time))

# Initialize matrix to save λ_prior
λ_prior = ones((n_barcode + 1) * n_time, 2)
# Set λ_prior for barcodes
λ_prior[1:end-n_time, 1] .*= 3
λ_prior[1:end-n_time, 2] .*= 3
# Set λ_prior for grouped barcodes
λ_prior[end-n_time+1:end, 1] *= 6
λ_prior[end-n_time+1:end, 2] *= 3

# Define function parameters
param = Dict(
    :data => data,
    :n_walkers => 4,
    :n_steps => 1_000,
    :outputname => "./output/data_example_01_meanfitness_1000steps_04walkers",
    :model => BarBay.model.mean_fitness_lognormal,
    :model_kwargs => Dict(
        :λ_prior => λ_prior
    ),
    :sampler => Turing.NUTS(1000, 0.65),
    :ensemble => Turing.MCMCThreads(),
)

##

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Run inference
println("Running Inference...")
@time BarBay.mcmc.mcmc_joint_mean_fitness(; param...)

##

println("Plotting trances and densities...\n")
# Read MCMC chain with population mean fitness
chains = JLD2.load(
    "./output/data_example_01_meanfitness_1000steps_04walkers.jld2"
)["chain"]

# Locate parameter names
var_names = MCMCChains.namesingroup(chains, :s̲ₜ)

# Initialize figure
fig = Figure(resolution=(600, 600))

# Generate mcmc_trace_density! plot
BarBay.viz.mcmc_trace_density!(fig, chains[var_names]; alpha=0.5)

# save("../docs/src/figs/fig03.svg", fig)

fig

##

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
    chains, n_ppc; param=param
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

# save("../docs/src/figs/fig04.svg", fig)

fig

##