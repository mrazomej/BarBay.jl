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

# Import basic math
import StatsBase
import Random

# Import library to save and load native julia objects
import JLD2

# Import library to perform Bayesian inference
import Turing
# Import AutoDiff backend
using ReverseDiff

# Import library to list files
import Glob

# Import plotting libraries
using CairoMakie
import ColorSchemes

Random.seed!(42)
##

# Set AutoDiff backend
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

##

# Define sampling hyperparameters
n_steps = 1000
n_walkers = 4

# Define boolean to plot trajectories
plot_trajectories = false

##
println("Loading data...\n")
# Import data
data = CSV.read(
    "$(git_root())/test/data/data_example_multi-env.csv", DF.DataFrame
)

##


if plot_trajectories
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Plot trajectories
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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

    # List unique environments
    envs = unique(data.env)

    # Define environment-color dictionary
    colors = Dict(envs .=> ColorSchemes.Accent_3[1:length(envs)])

    # Define time-environment relation
    time_env = Matrix(unique(data[:, [:time, :env]]))

    # Loop through each time point
    for t = 1:size(time_env, 1)
        # Color plot background
        vspan!(
            ax,
            time_env[t, 1] - 1,
            time_env[t, 1],
            color=(colors[time_env[t, 2]], 0.25)
        )
    end # for

    # Plot mutant barcode frequency trajectories
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

    # Set axis limits
    xlims!(ax, -0.5, 5)

    fig

    ##

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Plot log-freq ratio trajectories
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    println("Plotting log-frequency ratio trajectories...\n")
    # Initialize figure
    fig = Figure(resolution=(450, 350))

    # Add axis
    ax = Axis(
        fig[1, 1],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
        title="log-frequency ratio trajectories"
    )

    # List unique environments
    envs = unique(data.env)

    # Define environment-color dictionary
    colors = Dict(envs .=> ColorSchemes.Accent_3)

    # Define time-environment relation
    time_env = Matrix(unique(data[:, [:time, :env]]))

    # Loop through each time point
    for t = 2:size(time_env, 1)
        # Color plot background
        vspan!(
            ax,
            time_env[t, 1] - 0.5,
            time_env[t, 1] + 0.5,
            color=(colors[time_env[t, 2]], 0.25)
        )
    end # for

    # Plot mutant barcode frequency trajectories
    BayesFitness.viz.logfreq_ratio_time_series!(
        ax,
        data[.!data.neutral, :];
        freq_col=:freq,
        alpha=0.25,
        linewidth=2
    )

    # Plot Neutral barcode trajectories
    BayesFitness.viz.logfreq_ratio_time_series!(
        ax,
        data[data.neutral, :];
        freq_col=:freq,
        color=ColorSchemes.Blues_9[end],
        alpha=0.9,
        linewidth=2
    )

    # Set axis limits
    xlims!(ax, 1, 5)
    ylims!(ax, -4, 4)

    fig

end # if
##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Setting sampling parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of barcodes
n_barcode = length(unique(data[data.neutral, :barcode])) + 1

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
    :n_walkers => n_walkers,
    :n_steps => n_steps,
    :outputname => "./output/single_mutant_multi-env_inference/" *
                   "chain_multi-env_$(lpad(n_steps, 2, "0"))steps_" *
                   "$(lpad(n_walkers, 2, "0"))walkers_bc",
    :model => BayesFitness.model.env_fitness_lognormal,
    :model_kwargs => Dict(
        :envs => ["G", "H", "N", "G", "H", "N"],
        :λ_prior => λ_prior,
    ),
    :sampler => Turing.NUTS(500, 0.65),
    :ensemble => Turing.MCMCSerial(),
    :multithread => true
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Sample posterior distribution
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Create output directory
if !isdir("./output/single_mutant_multi-env_inference/")
    mkdir("./output/single_mutant_multi-env_inference/")
end # if


# Run inference
println("Running Inference...")
@time BayesFitness.mcmc.mcmc_single_fitness(; param...)

