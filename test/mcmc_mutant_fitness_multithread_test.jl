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

# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

##

# Infer mean fitness distributions
mean_fitness_dist = BayesFitness.stats.gaussian_prior_mean_fitness(
    BayesFitness.utils.var_jld2_to_df("./output/", "data_01_meanfitness", :sₜ)
)

##

# Define function parameters
param = Dict(
    :data => data,
    :n_walkers => 3,
    :n_steps => 1_000,
    :outputdir => "./output/",
    :outputname => "data_01_mutantfitness",
    :model => BayesFitness.model.mutant_fitness_lognormal,
    :model_kwargs => Dict(
        :α => BayesFitness.stats.beta_prior_mutant(
            data[data.time.==0, :barcode],
        ),
        :μ_s̄ => mean_fitness_dist[1],
        :σ_s̄ => mean_fitness_dist[2],
    )
)

##

# Run inference
BayesFitness.mcmc.mcmc_mutant_fitness_multithread(; param...)

##

# Find barcode with maximum count
bc = data[first(argmax(data.count, dims=1)), :barcode]

# Select file to process
file = first(Glob.glob("$(param[:outputdir])/$(param[:outputname])*$(bc)*"))

# Extract data
data_bc = data[data.barcode.==bc, :]

# Sort data by time
DF.sort!(data_bc, :time)

# Load one of the files as an example
mcmc_chain = JLD2.load(file)["chain"]

##

# Define quantiles to compute
qs = [0.95, 0.675, 0.02]

# Define colors
colors = reverse(get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs))))

# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="barcode frequency",
    yscale=log10
)

# Plot posterior predictive checks 
BayesFitness.viz.freq_mutant_ppc!(ax, qs, mcmc_chain; colors=colors)

# Add scatter of data
scatterlines!(ax, data_bc.count ./ data_bc.count_sum, color=:black)

fig

##
