# Import basic math
import LinearAlgebra

# Import libraries to define distributions
import Distributions
import Random

# Import libraries relevant for MCMC
import Turing

# ============================================================================ # 
# Include functions
# ============================================================================ # 

# fitness inference π(s̲⁽ᵐ⁾, s̲ₜ | data)
include("./model_fitness_normal.jl")

# fitness inference in multiple environemnts π(s1⁽ᵐ⁾, s2⁽ᵐ⁾,.. | data)
include("./model_multienv_fitness_normal.jl")

# Hierarchical model for multiple experimental replicates π(θ̲ᴹ, s̲ᴹ, s̲ₜ | data)
include("./model_fitness_normal_hierarchical_replicates.jl")

# Hierarchical model for multiple environments and multiple experimental
# replicates π(θ̲ᴹ₁, θ̲ᴹ₂,…, s̲ᴹ, s̲ₜ | data)
include("./model_multienv_fitness_normal_hierarchical_replicates.jl")

# Hierarchical model for genotypes on a single experiment π(θ̲ᴹ, s̲ᴹ, s̲ₜ | data)
include("./model_fitness_normal_hierarchical_genotypes.jl")