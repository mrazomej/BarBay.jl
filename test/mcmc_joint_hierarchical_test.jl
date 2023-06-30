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

##
println("Loading data...\n")
# Import data
data = CSV.read(
    "$(git_root())/test/data/data_hierarchical_example_01.csv", DF.DataFrame
)

##

# Plot trajectories

println("Plotting frequency trajectories...\n")
# Initialize figure
fig = Figure(resolution=(450 * 2, 350))

# Add axis
ax1 = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="barcode frequency",
    yscale=log10,
    title="frequency trajectories | R1"
)

ax2 = Axis(
    fig[1, 2],
    xlabel="time point",
    ylabel="barcode frequency",
    yscale=log10,
    title="frequency trajectories | R2"
)

# compile axes into single object
axes = [ax1, ax2]
# Loop through repeats
for (i, rep) in enumerate(unique(data.rep))
    # Plot Mutant barcode trajectories
    BayesFitness.viz.bc_time_series!(
        axes[i],
        data[(.!data.neutral).&(data.rep.==rep), :];
        quant_col=:freq,
        zero_lim=1E-7,
        zero_label="extinct",
        alpha=0.25,
        linewidth=2
    )

    # Plot Neutral barcode trajectories for R1
    BayesFitness.viz.bc_time_series!(
        axes[i],
        data[(data.neutral).&(data.rep.==rep), :];
        quant_col=:freq,
        zero_lim=1E-7,
        color=ColorSchemes.Blues_9[end],
        alpha=0.9,
        linewidth=2
    )
end # for

fig

##

param = Dict(
    :data => data,
    :n_walkers => n_walkers,
    :n_steps => n_steps,
    :outputname => "./output/data_example_01_hierarchical_$(n_steps)steps_$(lpad(n_walkers, 2, "0"))walkers",
    :model => BayesFitness.model.fitness_hierarchical_replicates,
    :model_kwargs => Dict(
        :λ_prior => [3.0, 3.0]
    ),
    :sampler => Turing.NUTS(0.65),
    :ensemble => Turing.MCMCThreads(),
)

##

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Run inference
println("Running Inference...")
@time BayesFitness.mcmc.mcmc_joint_fitness_hierarchical_replicates(; param...)
