module BayesFitness

# Export mean fitness model
export mean_fitness_neutrals_lognormal in model

module model
include("model.jl")
end # model submodule

end # BayesFitness module
