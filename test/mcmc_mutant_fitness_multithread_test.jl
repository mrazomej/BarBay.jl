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