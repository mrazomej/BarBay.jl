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

# Import library to set random seed
import Random

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

# Define number of steps
n_steps = 1_000
# Define number of walkers
n_walkers = 4

# Define whether plots should be generated or not
gen_plots = false

##

println("Loading data...\n")
# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

##

if gen_plots
    # Plot trajectories

    println("Plotting frequency trajectories...\n")
    # Initialize figure
    fig = Figure(resolution=(450, 350))

    # Add axis
    ax = Axis(
        fig[1, 1],
        xlabel="time point",
        ylabel="barcode frequency",
        yscale=log10,
        title="frequency trajectories"
    )

    # Plot Mutant barcode trajectories
    BayesFitness.viz.bc_time_series!(
        ax,
        data[.!data.neutral, :];
        quant_col=:freq,
        zero_lim=1E-7,
        zero_label="extinct",
        alpha=0.25,
        linewidth=2
    )

    # Plot Neutral barcode trajectories
    BayesFitness.viz.bc_time_series!(
        ax,
        data[data.neutral, :];
        quant_col=:freq,
        zero_lim=1E-7,
        color=ColorSchemes.Blues_9[end],
        alpha=0.9,
        linewidth=2
    )

    # save("../docs/src/figs/fig01.svg", fig)

    fig

end # if
##

if gen_plots
    println("Plotting log-frequency ratio trajectories...\n")
    # Initialize figure
    fig = Figure(resolution=(450, 350))

    # Add axis
    ax = Axis(
        fig[1, 1],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
        title="log-frequency ratio"
    )

    # Plot log-frequency ratio of mutants
    BayesFitness.viz.logfreq_ratio_time_series!(
        ax,
        data[.!data.neutral, :];
        freq_col=:freq,
        alpha=0.25,
        linewidth=2
    )

    # Plot log-frequency ratio of neutrals
    BayesFitness.viz.logfreq_ratio_time_series!(
        ax,
        data[data.neutral, :];
        freq_col=:freq,
        color=ColorSchemes.Blues_9[end],
        alpha=1.0,
        linewidth=2
    )

    # save("../docs/src/figs/fig02.svg", fig)

    fig
end # if

##

# Define function parameters

param = Dict(
    :data => data,
    :n_walkers => n_walkers,
    :n_steps => n_steps,
    :outputname => "./output/chain_popmean_fitness_$(n_steps)steps_$(lpad(n_walkers, 2, "0"))walkers",
    :model => BayesFitness.model.neutrals_lognormal,
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
BayesFitness.mcmc.mcmc_popmean_fitness(; param...)

##

# println("Plotting trances and densities...\n")
# # Concatenate population mean fitness chains into single chain
# chains = BayesFitness.utils.jld2_concat_chains(
#     param[:outputdir], param[:outputname], [:sₜ]; id_str=""
# )

# # Initialize figure
# fig = Figure(resolution=(600, 600))

# # Generate mcmc_trace_density! plot
# BayesFitness.viz.mcmc_trace_density!(fig, chains; alpha=0.5)

# save("../docs/src/figs/fig03.svg", fig)

# fig

# ##

# # Name variables to be extracted from chains
# chain_vars = [:sₜ, :σₜ]

# # Extract variables into single chain object
# chains = BayesFitness.utils.jld2_concat_chains(
#     param[:outputdir], param[:outputname], chain_vars; id_str=""
# )

# # Define number of posterior predictive check samples
# n_ppc = 5_000

# # Define dictionary with corresponding parameters for variables needed for the
# # posterior predictive checks
# param = Dict(
#     :population_mean_fitness => :sₜ,
#     :population_std_fitness => :σₜ,
# )

# # Compute posterior predictive checks
# ppc_mat = BayesFitness.stats.logfreq_ratio_mean_ppc(
#     chains, n_ppc; param=param
# )

# ##

# # Initialize figure
# fig = Figure(resolution=(450, 350))

# # Add axis
# ax = Axis(
#     fig[1, 1],
#     xlabel="time point",
#     ylabel="ln(fₜ₊₁/fₜ)",
#     title="log-frequency ratio PPC"
# )

# # Define quantiles to compute
# qs = [0.68, 0.95, 0.997]

# # Define colors
# colors = get(ColorSchemes.Blues_9, LinRange(0.25, 0.75, length(qs)))

# # Plot posterior predictive checks
# BayesFitness.viz.ppc_time_series!(
#     ax, qs, ppc_mat; colors=colors
# )

# # Add plot for median (we use the 5 percentile to have a "thicker" line showing
# # the median)
# BayesFitness.viz.ppc_time_series!(
#     ax, [0.05], ppc_mat; colors=ColorSchemes.Blues_9[end:end]
# )

# # Plot log-frequency ratio of neutrals
# BayesFitness.viz.logfreq_ratio_time_series!(
#     ax,
#     data[data.neutral, :];
#     freq_col=:freq,
#     color=:black,
#     alpha=1.0,
#     linewidth=2
# )

# save("../docs/src/figs/fig04.svg", fig)

# fig

# ##