##
println("hello, world!")
##

# Import package to revise package
import Revise

# Import library package
import BayesFitness

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import library for Bayesian inference
import Turing
##

# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

##

# Define function parameters

param = Dict(
    :data => data,
    :n_walkers => 1,
    :n_steps => 10,
    :outputdir => "./",
    :outputname => "test_mcmc",
    :model => BayesFitness.model.mean_fitness_neutrals_lognormal,
    :model_kwargs => Dict(
        :α => BayesFitness.stats.dirichlet_uniform_prior_neutral(
            data[data.time.==0, :neutral],
        )
    )
)
##

# Run inference
BayesFitness.mcmc.mcmc_mean_fitness(; param...)


##

data::DF.AbstractDataFrame,
n_walkers::Int,
n_steps::Int,
outputdir::String,
outputname::String,
model::Function,
model_kwargs::Dict = Dict(),
id_col::Symbol = :barcode,
time_col::Symbol = :time,
neutral_col::Symbol = :neutral,
rm_T0::Bool = false,
suppress_output::Bool = false,
sampler::Turing.Inference.InferenceAlgorithm = Turing.NUTS(0.65),
verbose::Bool = true