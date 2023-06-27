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
    :outputname => "./output/data_example_01_meanfitness_1000steps_04walkers.jld2",
    :model => BayesFitness.model.mean_fitness_lognormal,
    :model_kwargs => Dict(
        :λ_prior => λ_prior
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
@time BayesFitness.mcmc.mcmc_joint_mean_fitness(; param...)

##