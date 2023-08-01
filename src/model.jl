# Import basic math
import LinearAlgebra

# Import libraries to define distributions
import Distributions
import Random

# Import libraries relevant for MCMC
import Turing

# Functionality for constructing arrays with identical elements efficiently
import FillArrays
##
# Export functions

# Export joint fitness models
export fitness_lognormal

##

# ============================================================================ # 
# Include functions
# ============================================================================ # 

# Population mean fitness inference from neutrals data π(s̲ₜ | data)
include("./model_freq_lognormal.jl")

# Population mean fitness inference from neutrals data π(s̲ₜ | data)
include("./model_neutrals_lognormal.jl")

# fitness inference π(s̲⁽ᵐ⁾, s̲ₜ | data)
include("./model_fitness_lognormal.jl")

# fitness inference π(s̲⁽ᵐ⁾, s̲ₜ | data)
include("./model_fitness_normal.jl")

# fitness inference in multiple environemnts π(s1⁽ᵐ⁾, s2⁽ᵐ⁾,.. | data)
include("./model_multienv_fitness_lognormal.jl")

# fitness inference in multiple environemnts π(s1⁽ᵐ⁾, s2⁽ᵐ⁾,.. | data)
include("./model_multienv_fitness_normal.jl")

# Hierarchical model for multiple experimental replicates π(θ̲ᴹ, s̲ᴹ, s̲ₜ | data)
include("./model_fitness_lognormal_hierarchical_replicates.jl")

# Hierarchical model for multiple experimental replicates π(θ̲ᴹ, s̲ᴹ, s̲ₜ | data)
include("./model_fitness_normal_hierarchical_replicates.jl")

# Hierarchical model for genotypes on a single experiment π(θ̲ᴹ, s̲ᴹ, s̲ₜ | data)
include("./model_fitness_normal_hierarchical_genotypes.jl")