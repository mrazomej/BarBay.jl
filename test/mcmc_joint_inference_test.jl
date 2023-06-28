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

# Define sampling hyperparameters
n_steps = 1000
n_walkers = 4

##

println("Loading data...\n")
# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

# Define function parameters

param = Dict(
    :data => data,
    :n_walkers => n_walkers,
    :n_steps => n_steps,
    :outputname => "./output/data_example_01_$(n_steps)steps_$(lpad(n_walkers, 2, "0"))walkers.jld2",
    :model => BayesFitness.model.fitness_lognormal,
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
@time BayesFitness.mcmc.mcmc_joint_fitness(; param...)

##