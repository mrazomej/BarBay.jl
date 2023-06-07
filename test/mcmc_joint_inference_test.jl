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
data = CSV.read("$(git_root())/test/data/data_example_02.csv", DF.DataFrame)
# Add frequency column
data[!, :freq] = data.count ./ data.count_sum

##

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
    zero_lim=1E-9,
    zero_label="extinct",
    alpha=0.25,
    linewidth=2
)

# Plot Neutral barcode trajectories
BayesFitness.viz.bc_time_series!(
    ax,
    data[data.neutral, :];
    quant_col=:freq,
    zero_lim=1E-9,
    color=ColorSchemes.Blues_9[end],
    alpha=0.9,
    linewidth=2
)

fig

##

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

fig

##

# Define function parameters

param = Dict(
    :data => data,
    :n_walkers => 1,
    :n_steps => 10,
    :outputname => "./output/data_example_02_joint_fitness",
    :model => BayesFitness.model.fitness_lognormal,
    :multithread => false,
    :sampler => Turing.HMC(0.05, 10)
)

##

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Run inference
println("Running Inference...")
@time BayesFitness.mcmc.mcmc_joint_fitness(; param...)

##