##

# Activate environment
@load_pkg(".")

# Import package to revise package
import Revise

# Import library package
import BayesFitness

# Import libraries to manipulate data
import DataFrames as DF
import CSV
import MCMCChains

# Import library to save and load native julia objects
import JLD2

# Import library to list files
import Glob

# Import plotting libraries
using CairoMakie
import ColorSchemes
##

println("Loading data...\n")
# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

##

println("Fitting Gaussian distributions to inferred mean fitness...\n")

# Extract mean fitness MCMC chains
mean_fitness_chains = BayesFitness.utils.jld2_concat_chains(
    "./output/", "data_01_meanfitness", [:sₜ]; id_str=""
)

# Infer mean fitness distribution parameters by fitting a Gaussian
mean_fitness_dist = BayesFitness.stats.gaussian_prior_mean_fitness(
    mean_fitness_chains
)
##

# Infer mean fitness distributions by fitting a Gaussian
fit_dists = BayesFitness.stats.gaussian_prior_mean_fitness(
    mean_fitness_chains,
    params=false
)

# Initialize figure
fig = Figure(resolution=(600, 600))

# Add axis objects for each timepoint
axes = [
    Axis(
        fig[i, j],
        xlabel="population mean fitness (s̄ₜ)",
        ylabel="ecdf",
    ) for i = 1:2 for j = 1:2
]

# Loop through time points
for (i, var) in enumerate(names(mean_fitness_chains))
    # Plot ECDF
    BayesFitness.viz.mcmc_fitdist_cdf!(
        axes[i],
        Array(mean_fitness_chains[var])[:],
        fit_dists[i]
    )

    axes[i].title = "timepoint $(i)"
end # for

save("../docs/src/figs/fig05.svg", fig)

fig

##

# Define function parameters
param = Dict(
    :data => data,
    :n_walkers => 4,
    :n_steps => 4_000,
    :outputdir => "./output/",
    :outputname => "data_01_mutantfitness",
    :model => BayesFitness.model.mutant_fitness_lognormal,
    :model_kwargs => Dict(
        :α => BayesFitness.stats.beta_prior_mutant(
            data[data.time.==0, :barcode],
        ),
        :μ_s̄ => mean_fitness_dist[1],
        :σ_s̄ => mean_fitness_dist[2],
    ),
    :multithread_mutant => true,
)

##

println("Running inference...\n")
# Run inference
BayesFitness.mcmc.mcmc_mutant_fitness(; param...)

##

# Find barcode with maximum count
bc = data[first(argmax(data.count, dims=1)), :barcode]

# Select file to process
file = first(
    Glob.glob("$(param[:outputdir])/$(param[:outputname])*$(bc).jld2")
)

# Load one of the files as an example
chain = JLD2.load(file)["chain"]

##

# Name variables to be extracted from chains
chain_vars = [Symbol("s⁽ᵐ⁾"), Symbol("σ⁽ᵐ⁾")]

# Extract variables from chain
chn = chain[chain_vars]

# Initialize figure
fig = Figure(resolution=(600, 350))

# Generate mcmc_trace_density! plot
BayesFitness.viz.mcmc_trace_density!(fig, chn; alpha=0.5)

save("../docs/src/figs/fig06.svg", fig)

fig

##
# Name variables to be extracted from chains
chain_vars = [Symbol("s⁽ᵐ⁾"), Symbol("σ⁽ᵐ⁾"), Symbol("f̲⁽ᵐ⁾[1]"), :s̲ₜ]

# Locate variable names to extract from chain
chain_names = reduce(
    vcat, [MCMCChains.namesingroup(chain, var) for var in chain_vars]
)

# Extract chain variables
chn = chain[chain_names]

##

# Extract data for barcode example
data_bc = data[data.barcode.==bc, :]

# Sort data by time
DF.sort!(data_bc, :time)

##

# Define number of posterior predictive check samples
n_ppc = 5_000

# Define dictionary with corresponding parameters for variables needed for the
# posterior predictive checks
param = Dict(
    :mutant_mean_fitness => :s⁽ᵐ⁾,
    :mutant_std_fitness => :σ⁽ᵐ⁾,
    :mutant_freq => Symbol("f̲⁽ᵐ⁾[1]"),
    :population_mean_fitness => :s̲ₜ,
)

# Compute posterior predictive checks
ppc_mat = BayesFitness.stats.freq_mutant_ppc(
    chn,
    n_ppc;
    param=param
)

##

# Initialize figure
fig = Figure(resolution=(450, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="barcode frequency",
    title="frequency trajectories",
    yscale=log10,
)

# Define quantiles to compute
qs = [0.95, 0.675]

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.5, 0.75, length(qs)))

# Plot posterior predictive checks
BayesFitness.viz.ppc_time_series!(
    ax, qs, ppc_mat; colors=colors
)

# Add plot for median
BayesFitness.viz.ppc_time_series!(
    ax, [0.03], ppc_mat; colors=ColorSchemes.Blues_9[end:end]
)

# Add scatter of data
scatterlines!(ax, data_bc.freq, color=:black)

save("../docs/src/figs/fig07.svg", fig)

fig

##

# Name variables to be extracted from chains
chain_vars = [Symbol("s⁽ᵐ⁾"), Symbol("σ⁽ᵐ⁾"), :s̲ₜ]

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
    :mutant_mean_fitness => :s⁽ᵐ⁾,
    :mutant_std_fitness => :σ⁽ᵐ⁾,
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

save("../docs/src/figs/fig08.svg", fig)
fig

##