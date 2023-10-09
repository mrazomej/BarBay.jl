# Activate environment
@load_pkg(".")

# Import package to revise package
import Revise

# Import library package
import BarBay

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import library to set random seed
import Random
import Distributions
import StatsBase

# Import library to save and load native julia objects
import JLD2

# Import plotting libraries
using CairoMakie
import ColorSchemes

# Import library to perform Bayesian inference
import Turing
import MCMCChains
import DynamicHMC

# Import AutoDiff backend
using ReverseDiff

# Import Memoization
using Memoization

Random.seed!(42)

##

# Set AutoDiff backend
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

##

# Define sampling hyperparameters
n_steps = 100
n_walkers = 4

# Define whether or not to generate diagnostic plots
gen_plots = true

##

println("Loading data...\n")
# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

##

# Define function parameters

param = Dict(
    :data => data,
    :n_walkers => n_walkers,
    :n_steps => n_steps,
    :outputname => "./output/chain_freq_$(n_steps)steps_$(lpad(n_walkers, 2, "0"))walkers",
    :model => BarBay.model.freq_lognormal,
    :model_kwargs => Dict(
        :λ_prior => [3.0, 3.0]
    ),
    :sampler => Turing.DynamicNUTS(),
    :ensemble => Turing.MCMCThreads(),
)

##

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Run inference
println("Running Inference...")
@time BarBay.mcmc.mcmc_joint_fitness(; param...)

##

# Load chain and mutant order into memory
ids, chn = values(JLD2.load("$(param[:outputname]).jld2"))

##

if gen_plots
    println("Fitting parametric distributions to relevant parameters")

    # Select variables for population mean fitness and associated variance
    var_name = MCMCChains.namesingroup(chn, :Λ̲̲)

    # Fit normal distributions to population mean fitness
    freq_dist = Distributions.fit.(
        Ref(Distributions.LogNormal), [vec(chn[x]) for x in var_name]
    )
end # if

##

Random.seed!(42)

if gen_plots
    println("Plotting ECDF plots for population mean fitness")

    # Define number of rows and columns
    n_row, n_col = [4, 4]

    # Sample random set of observations to plot
    idx = StatsBase.sample(eachindex(var_name), n_col * n_row; replace=false)

    # Initialize figure
    fig = Figure(resolution=(300 * n_col, 300 * n_row))

    # Add axis objects for each timepoint
    axes = [
        Axis(
            fig[i, j],
            xlabel="λ parameter",
            ylabel="ecdf",
        ) for i = 1:n_row for j = 1:n_col
    ]

    # Loop through time points
    for (i, j) in enumerate(idx)
        # Plot ECDF
        BarBay.viz.mcmc_fitdist_cdf!(
            axes[i],
            vec(chn[var_name[j]])[:],
            freq_dist[j]
        )
    end # for

    fig
end # if