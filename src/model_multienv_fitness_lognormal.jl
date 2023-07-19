# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# joint fitness inference in multiple environemnts π(s1⁽ᵐ⁾, s2⁽ᵐ⁾,.. | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    multienv_fitness_lognormal(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲, n̲ₜ; kwargs)

`Turing.jl` model to sample the joint posterior distribution for a competitive
fitness experiment with different environments on each growth-dilution cycle.

# Model
`[write model here]`

# Arguments
- `R̲̲⁽ⁿ⁾::Matrix{Int64}`: `T × N` matrix where `T` is the number of time points
  in the data set and `N` is the number of neutral lineage barcodes. Each column
  represents the barcode count trajectory for a single neutral lineage.
  **NOTE**: The model assumes the rows are sorted in order of increasing time.
- `R̲̲⁽ᵐ⁾::Matrix{Int64}`: `T × M` matrix where `T` is the number of time points
  in the data set and `M` is the number of mutant lineage barcodes. Each column
  represents the barcode count trajectory for a single mutant lineage. **NOTE**:
  The model assumes the rows are sorted in order of increasing time.
- `R̲̲::Vector{Vector{Int64}}`:: `T × B` matrix--split into a vector of vectors
  for computational efficiency--where `T` is the number of time points in the
  data set and `B` is the number of barcodes. Each column represents the barcode
  count trajectory for a single lineage. **NOTE**: This matrix does not
  necessarily need to be equivalent to `hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾)`. This is because
  `R̲̲⁽ᵐ⁾` can exclude mutant barcodes to perform the joint inference only for a
  subgroup, but `R̲̲` must still contain all counts. Usually, if `R̲̲⁽ᵐ⁾`
  excludes mutant barcodes, `R̲̲` must be of the form `hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾,
  R̲̲⁽ᴹ⁾)`, where `R̲̲⁽ᴹ⁾` is a vector that aggregates all excluded mutant
  barcodes into a "super barcode."
- `n̲ₜ::Vector{Int64}`: Vector with the total number of barcode counts for each
  time point. **NOTE**: This vector **must** be equivalent to computing
  `vec(sum(R̲̲, dims=2))`. The reason it is an independent input parameter is to
  avoid the `sum` computation within the `Turing` model.

## Keyword Arguments
- `envs::Vector{<:Any}`: List of environments for each time point in dataset.
  NOTE: The length must be equal to that of `n̲ₜ` to have one environment per
  time point.

## Optional Keyword Arguments
- `s_pop_prior::VecOrMat{Float64}=[0.0, 2.0]`: Vector or Matrix with the
    correspnding parameters (Vector: `s_pop_prior[1]` = mean, `s_pop_prior[2]` =
    standard deviation, Matrix: `s_pop_prior[:, 1] = mean`, `s_pop_prior[:, 2] =
    standard deviation`) for a Normal prior on the population mean fitness
    values. If `typeof(s_pop_prior) <: Matrix`, there should be as many rows in
    the matrix as pairs of time adjacent time points in dataset.
- `σ_pop_prior::VecOrMat{Float64}=[0.0, 1.0]`: Vector or Matrix with the
    correspnding parameters (Vector: `σ_pop_prior[1]` = mean, `σ_pop_prior[2]` =
    standard deviation, Matrix: `σ_pop_prior[:, 1] = mean`, `σ_pop_prior[:, 2] =
    standard deviation`) for a Log-Normal prior on the population mean fitness
    error utilized in the log-likelihood function. If `typeof(σ_pop_prior) <:
    Matrix`, there should be as many rows in the matrix as pairs of time
    adjacent time points in dataset.
- `s_mut_prior::VecOrMat{Float64}=[0.0, 2.0]`: Vector or Matrix with the
    correspnding parameters (Vector: `s_mut_prior[1]` = mean, `s_mut_prior[2]` =
    standard deviation, Matrix: `s_mut_prior[:, 1] = mean`, `s_mut_prior[:, 2] =
    standard deviation`) for a Normal prior on the mutant fitness values. If
    `typeof(s_mut_prior) <: Matrix`, there should be as many rows in the matrix
    as mutant lineages × number of unique environments in the dataset.
- `σ_mut_prior::VecOrMat{Float64}=[0.0, 1.0]`: Vector or Matrix with the
  correspnding parameters (Vector: `s_mut_prior[1]` = mean, `s_mut_prior[2]` =
  standard deviation, Matrix: `s_mut_prior[:, 1] = mean`, `s_mut_prior[:, 2] =
  standard deviation`) for a Log-Normal prior on the mutant fitness error
  utilized in the log-likelihood function. If `typeof(σ_mut_prior) <: Matrix`,
  there should be as many rows in the matrix as mutant lineages × number of
  unique environments in the dataset.
- `λ_prior::VecOrMat{Float64}=[3.0, 3.0]`: Vector or Matrix with the
  correspnding parameters (Vector: `λ_prior[1]` = mean, `λ_prior[2]` = standard
  deviation, Matrix: `λ_prior[:, 1] = mean`, `λ_prior[:, 2] = standard
  deviation`) for a Log-Normal prior on the λ parameter in the Poisson
  distribution. The λ parameter can be interpreted as the mean number of barcode
  counts since we assume any barcode count `n⁽ᵇ⁾ ~ Poisson(λ⁽ᵇ⁾)`. If
  `typeof(λ_prior) <: Matrix`, there should be as many rows in the matrix as
  number of barcodes × number of time points in the dataset.
"""
Turing.@model function multienv_fitness_lognormal(
    R̲̲⁽ⁿ⁾::Matrix{Int64},
    R̲̲⁽ᵐ⁾::Matrix{Int64},
    R̲̲::Vector{Vector{Int64}},
    n̲ₜ::Vector{Int64};
    envs::Vector{<:Any},
    s_pop_prior::VecOrMat{Float64}=[0.0, 2.0],
    σ_pop_prior::VecOrMat{Float64}=[0.0, 1.0],
    s_mut_prior::VecOrMat{Float64}=[0.0, 2.0],
    σ_mut_prior::VecOrMat{Float64}=[0.0, 1.0],
    λ_prior::VecOrMat{Float64}=[3.0, 3.0]
)
    # Check that the number of time points and environments matches
    if length(n̲ₜ) != length(envs)
        error("Number of time points must match list of of environments")
    end # if

    # Find unique environments
    env_unique = unique(envs)
    # Define number of environments
    n_env = length(env_unique)
    # Define environmental indexes
    env_idx = indexin(envs, env_unique)

    # Define number of time points
    n_time = length(n̲ₜ)
    # Define number of neutrals
    n_neutral = size(R̲̲⁽ⁿ⁾, 2)
    # Define numbero f mutants
    n_mut = size(R̲̲⁽ᵐ⁾, 2)

    ## %%%%%%%%%%%%%% Population mean fitness  %%%%%%%%%%%%%% ##

    # Prior on population mean fitness π(s̲ₜ) 
    if typeof(s_pop_prior) <: Vector
        s̲ₜ ~ Turing.MvNormal(
            repeat([s_pop_prior[1]], n_time - 1),
            LinearAlgebra.I(n_time - 1) .* s_pop_prior[2] .^ 2
        )
    elseif typeof(s_pop_prior) <: Matrix
        s̲ₜ ~ Turing.MvNormal(
            s_pop_prior[:, 1], LinearAlgebra.Diagonal(s_pop_prior[:, 2] .^ 2)
        )
    end # if

    # Prior on LogNormal error π(σ̲ₜ)
    if typeof(σ_pop_prior) <: Vector
        σ̲ₜ ~ Turing.MvLogNormal(
            repeat([σ_pop_prior[1]], n_time - 1),
            LinearAlgebra.I(n_time - 1) .* σ_pop_prior[2] .^ 2
        )
    elseif typeof(σ_pop_prior) <: Matrix
        σ̲ₜ ~ Turing.MvLogNormal(
            σ_pop_prior[:, 1], LinearAlgebra.Diagonal(σ_pop_prior[:, 2] .^ 2)
        )
    end # if

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Prior on mutant fitness π(s̲⁽ᵐ⁾)
    if typeof(s_mut_prior) <: Vector
        s̲⁽ᵐ⁾ ~ Turing.MvNormal(
            repeat([s_mut_prior[1]], n_mut * n_env),
            LinearAlgebra.I(n_mut * n_env) .* s_mut_prior[2] .^ 2
        )
    elseif typeof(s_mut_prior) <: Matrix
        s̲⁽ᵐ⁾ ~ Turing.MvNormal(
            s_mut_prior[:, 1], LinearAlgebra.Diagonal(s_mut_prior[:, 2] .^ 2)
        )
    end # if

    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    if typeof(σ_mut_prior) <: Vector
        σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
            repeat([σ_mut_prior[1]], n_mut * n_env),
            LinearAlgebra.I(n_mut * n_env) .* σ_mut_prior[2] .^ 2
        )
    elseif typeof(σ_mut_prior) <: Matrix
        σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
            σ_mut_prior[:, 1], LinearAlgebra.Diagonal(σ_mut_prior[:, 2] .^ 2)
        )
    end # if


    ## %%%%%%%%%%%%%% Barcode frequencies %%%%%%%%%%%%%% ##

    if typeof(λ_prior) <: Vector
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            repeat([λ_prior[1]], sum(length.(R̲̲))),
            LinearAlgebra.I(sum(length.(R̲̲))) .* λ_prior[2]^2
        )
    elseif typeof(λ_prior) <: Matrix
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            λ_prior[:, 1], LinearAlgebra.Diagonal(λ_prior[:, 2] .^ 2)
        )
    end  # if

    # Reshape λ parameters to fit the matrix format. Note: The Λ̲̲ array is
    # originally sampled as a vector for the `Turing.jl` samplers to deal with
    # it. But reshaping it to a matrix simplifies the computation of frequencies
    # and frequency ratios.
    Λ̲̲ = reshape(Λ̲̲, length(R̲̲), length(first(R̲̲)))

    # Compute barcode frequencies from Poisson parameters
    F̲̲ = Λ̲̲ ./ sum(Λ̲̲, dims=2)

    # Compute frequency ratios between consecutive time points.
    Γ̲̲ = F̲̲[2:end, :] ./ F̲̲[1:end-1, :]

    # Split neutral and mutant frequency ratios. Note: the @view macro means
    # that there is not allocation to memory on this step.
    Γ̲̲⁽ⁿ⁾ = vec(Γ̲̲[:, 1:n_neutral])
    Γ̲̲⁽ᵐ⁾ = vec(Γ̲̲[:, n_neutral+1:n_neutral+n_mut])

    # Prob of total number of barcodes read given the Poisosn distribution
    # parameters π(nₜ | λ̲ₜ)
    n̲ₜ ~ Turing.arraydist(Turing.Poisson.(vec(sum(Λ̲̲, dims=2))))

    # Prob of reads given parameters π(R̲ₜ | nₜ, f̲ₜ). 
    # Note # 1: We add the check_args=false option to avoid the recurrent
    # problem of
    # > Multinomial: p is not a probability vector. 
    # due to rounding errors 
    # Note # 2: We use @addlogprob! rather than a broadcasting function of the
    # form
    # R̲̲ .~ Turing.Multinomial.(n̲ₜ, eachrow(F̲̲); check_args=false)
    # because according to this discussion
    # (https://discourse.julialang.org/t/making-turing-fast-with-large-numbers-of-parameters/69072/78?u=dlakelan)
    # broadcasting does not work well when using ReverseDiff.jl
    Turing.@addlogprob! sum(
        Turing.logpdf.(
            Turing.Multinomial.(n̲ₜ, eachrow(F̲̲); check_args=false),
            R̲̲
        ),
    )

    ## %%%%%%%%%%%% Reshape arrays to split replicate variables %%%%%%%%%%%% ##
    s̲⁽ᵐ⁾ = reshape(s̲⁽ᵐ⁾, n_env, :)  # n_env × n_mut
    σ̲⁽ᵐ⁾ = reshape(σ̲⁽ᵐ⁾, n_env, :)  # n_env × n_mut

    ## %%%%%%%%%%%%%% Log-Likelihood functions %%%%%%%%%%%%%% ##

    # Sample posterior for neutral lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ⁿ⁾| sₜ, σₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            repeat(-s̲ₜ, n_neutral),
            LinearAlgebra.Diagonal(repeat(σ̲ₜ .^ 2, n_neutral))
        ),
        Γ̲̲⁽ⁿ⁾
    )

    # Sample posterior for nutant lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ᵐ⁾ | s⁽ᵐ⁾, σ⁽ᵐ⁾, s̲ₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            # Build vector for fitness differences
            vcat([s⁽ᵐ⁾[env_idx[2:end]] .- s̲ₜ for s⁽ᵐ⁾ in eachcol(s̲⁽ᵐ⁾)]...),
            # Build vector for variances
            LinearAlgebra.Diagonal(
                vcat([σ⁽ᵐ⁾[env_idx[2:end]] for σ⁽ᵐ⁾ in eachcol(σ̲⁽ᵐ⁾)]...) .^ 2
            )
        ),
        Γ̲̲⁽ᵐ⁾
    )
    return F̲̲
end # @model function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# single mutant fitness inference in multiple environemnts 
# π(s1⁽ᵐ⁾, s2⁽ᵐ⁾,.. | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    multienv_fitness_lognormal(R̲̲⁽ⁿ⁾, r̲⁽ᵐ⁾, R̲̲, n̲ₜ; kwargs)

`Turing.jl` model to sample the posterior distribution for a competitive fitness
experiment with different environments on each growth-dilution cycle using data
from a single mutant barcode and all available neutral barcodes.

# Model
`[write model here]`

# Arguments
- `R̲̲⁽ⁿ⁾::Matrix{Int64}`: `T × N` matrix where `T` is the number of time points
  in the data set and `N` is the number of neutral lineage barcodes. Each column
  represents the barcode count trajectory for a single neutral lineage.
  **NOTE**: The model assumes the rows are sorted in order of increasing time.
- `r̲⁽ᵐ⁾::Vector{Int64}`: `T` dimensional vector where `T` is the number of time
  points in the data set. **NOTE**: The model assumes the rows are sorted in
  order of increasing time.
- `R̲̲::Matrix{Int64}`:: `T × (N+2)` matrix, where `T` is the number of time
  points in the data set and `N` is the number of neutral barcodes. Each of the
  first `N` columns represent the barcode count trajectory for a single neutral
  lineage. The `N+1` column represents the count trajectory for the relevant
  mutant barcode. The `N+2` column represents the trajectory of all other
  ignored barcodes.
- `n̲ₜ::Vector{Int64}`: Vector with the total number of barcode counts for each
  time point. **NOTE**: This vector **must** be equivalent to computing
  `vec(sum(R̲̲, dims=2))`. The reason it is an independent input parameter is to
  avoid the `sum` computation within the `Turing` model.

## Keyword Arguments
- `envs::Vector{<:Any}`: List of environments for each time point in dataset.
  NOTE: The length must be equal to that of `n̲ₜ` to have one environment per
  time point.

## Optional Keyword Arguments
- `s_pop_prior::VecOrMat{Float64}=[0.0, 2.0]`: Vector or Matrix with the
    correspnding parameters (Vector: `s_pop_prior[1]` = mean, `s_pop_prior[2]` =
    standard deviation, Matrix: `s_pop_prior[:, 1] = mean`, `s_pop_prior[:, 2] =
    standard deviation`) for a Normal prior on the population mean fitness
    values. If `typeof(s_pop_prior) <: Matrix`, there should be as many rows in
    the matrix as pairs of time adjacent time points in dataset.
- `σ_pop_prior::VecOrMat{Float64}=[0.0, 1.0]`: Vector or Matrix with the
    correspnding parameters (Vector: `σ_pop_prior[1]` = mean, `σ_pop_prior[2]` =
    standard deviation, Matrix: `σ_pop_prior[:, 1] = mean`, `σ_pop_prior[:, 2] =
    standard deviation`) for a Log-Normal prior on the population mean fitness
    error utilized in the log-likelihood function. If `typeof(σ_pop_prior) <:
    Matrix`, there should be as many rows in the matrix as pairs of time
    adjacent time points in dataset.
- `s_mut_prior::VecOrMat{Float64}=[0.0, 2.0]`: Vector or Matrix with the
  correspnding parameters (Vector: `s_mut_prior[1]` = mean, `s_mut_prior[2]` =
  standard deviation, Matrix: `s_mut_prior[:, 1] = mean`, `s_mut_prior[:, 2] =
  standard deviation`) for a Normal prior on the mutant fitness values. If
  `typeof(s_mut_prior) <: Matrix`, there should be as many rows in the matrix as
  unique environments.
- `σ_mut_prior::VecOrMat{Float64}=[0.0, 1.0]`: Vector or Matrix with the
  correspnding parameters (Vector: `s_mut_prior[1]` = mean, `s_mut_prior[2]` =
  standard deviation, Matrix: `s_mut_prior[:, 1] = mean`, `s_mut_prior[:, 2] =
  standard deviation`) for a Log-Normal prior on the mutant fitness error
  utilized in the log-likelihood function. If `typeof(σ_mut_prior) <: Matrix`,
  there should be as many rows in the matrix as unique environments.
- `λ_prior::Vector{Float64}=[3.0, 3.0]`: Vector with the corresponding
  parameters (`λ_prior[1]` = mean, `λ_prior[2]` = standard deviation) for a
  Log-Normal prior on the λ parameter in the Poisson distribution. The λ
  parameter can be interpreted as the mean number of barcode counts since we
  assume any barcode count `n⁽ᵇ⁾ ~ Poisson(λ⁽ᵇ⁾)`. **NOTE**: This method assigns
    the same prior to **all** mutant fitness error values to be inferred.
"""
Turing.@model function multienv_fitness_lognormal(
    R̲̲⁽ⁿ⁾::Matrix{Int64},
    r̲⁽ᵐ⁾::Vector{Int64},
    R̲̲::Vector{Vector{Int64}},
    n̲ₜ::Vector{Int64};
    envs::Vector{<:Any},
    s_pop_prior::VecOrMat{Float64}=[0.0, 2.0],
    σ_pop_prior::VecOrMat{Float64}=[0.0, 1.0],
    s_mut_prior::VecOrMat{Float64}=[0.0, 2.0],
    σ_mut_prior::VecOrMat{Float64}=[0.0, 1.0],
    λ_prior::VecOrMat{Float64}=[3.0, 3.0]
)
    # Check that the number of time points and environments matches
    if length(n̲ₜ) != length(envs)
        error("Number of time points must match list of of environments")
    end # if

    # Find unique environments
    env_unique = unique(envs)
    # Define number of environments
    n_env = length(env_unique)
    # Define environmental indexes
    env_idx = indexin(envs, env_unique)

    # Define number of time points
    n_time = length(n̲ₜ)
    # Define number of neutrals
    n_neutral = size(R̲̲⁽ⁿ⁾, 2)

    ## %%%%%%%%%%%%%% Population mean fitness  %%%%%%%%%%%%%% ##

    # Prior on population mean fitness π(s̲ₜ) 
    if typeof(s_pop_prior) <: Vector
        s̲ₜ ~ Turing.MvNormal(
            repeat([s_pop_prior[1]], n_time - 1),
            LinearAlgebra.I(n_time - 1) .* s_pop_prior[2] .^ 2
        )
    elseif typeof(s_pop_prior) <: Matrix
        s̲ₜ ~ Turing.MvNormal(
            s_pop_prior[:, 1], LinearAlgebra.Diagonal(s_pop_prior[:, 2] .^ 2)
        )
    end # if

    # Prior on LogNormal error π(σ̲ₜ)
    if typeof(σ_pop_prior) <: Vector
        σ̲ₜ ~ Turing.MvLogNormal(
            repeat([σ_pop_prior[1]], n_time - 1),
            LinearAlgebra.I(n_time - 1) .* σ_pop_prior[2] .^ 2
        )
    elseif typeof(σ_pop_prior) <: Matrix
        σ̲ₜ ~ Turing.MvNormal(
            σ_pop_prior[:, 1], LinearAlgebra.Diagonal(σ_pop_prior[:, 2] .^ 2)
        )
    end # if

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Prior on mutant fitness π(s̲⁽ᵐ⁾)
    if typeof(s_mut_prior) <: Vector
        s̲⁽ᵐ⁾ ~ Turing.MvNormal(
            repeat([s_mut_prior[1]], n_mut),
            LinearAlgebra.I(n_mut) .* s_mut_prior[2] .^ 2
        )
    elseif typeof(s_mut_prior) <: Matrix
        s̲⁽ᵐ⁾ ~ Turing.MvNormal(
            s_mut_prior[:, 1], LinearAlgebra.Diagonal(s_mut_prior[:, 2] .^ 2)
        )
    end # if


    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    if typeof(σ_mut_prior) <: Vector
        σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
            repeat([σ_mut_prior[1]], n_mut),
            LinearAlgebra.I(n_mut) .* σ_mut_prior[2] .^ 2
        )
    elseif typeof(σ_mut_prior) <: Matrix
        σ̲⁽ᵐ⁾ ~ Turing.MvNormal(
            σ_mut_prior[:, 1], LinearAlgebra.Diagonal(σ_mut_prior[:, 2] .^ 2)
        )
    end # if

    ## %%%%%%%%%%%%%% Barcode frequencies %%%%%%%%%%%%%% ##

    if typeof(λ_prior) <: Vector
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            repeat([λ_prior[1]], sum(length.(R̲̲))),
            LinearAlgebra.I(sum(length.(R̲̲))) .* λ_prior[2]^2
        )
    elseif typeof(λ_prior) <: Matrix
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            λ_prior[:, 1], LinearAlgebra.Diagonal(λ_prior[:, 2] .^ 2)
        )
    end  # if

    # Reshape λ parameters to fit the matrix format. Note: The Λ̲̲ array is
    # originally sampled as a vector for the `Turing.jl` samplers to deal with
    # it. But reshaping it to a matrix simplifies the computation of frequencies
    # and frequency ratios.
    Λ̲̲ = reshape(Λ̲̲, size(R̲̲)...)

    # Compute barcode frequencies from Poisson parameters
    F̲̲ = Λ̲̲ ./ sum(Λ̲̲, dims=2)

    # Compute frequency ratios between consecutive time points.
    Γ̲̲ = F̲̲[2:end, :] ./ F̲̲[1:end-1, :]

    # Split neutral and mutant frequency ratios. Note: the @view macro means
    # that there is not allocation to memory on this step.
    Γ̲̲⁽ⁿ⁾ = vec(Γ̲̲[:, 1:n_neutral])
    γ̲⁽ᵐ⁾ = vec(Γ̲̲[:, n_neutral+1])

    # Prob of total number of barcodes read given the Poisosn distribution
    # parameters π(nₜ | λ̲ₜ)
    n̲ₜ ~ Turing.arraydist(Turing.Poisson.(vec(sum(Λ̲̲, dims=2))))

    # Prob of reads given parameters π(R̲ₜ | nₜ, f̲ₜ). 
    # Note # 1: We add the check_args=false option to avoid the recurrent
    # problem of
    # > Multinomial: p is not a probability vector. 
    # due to rounding errors 
    # Note # 2: We use @addlogprob! rather than a broadcasting function of the
    # form
    # R̲̲ .~ Turing.Multinomial.(n̲ₜ, eachrow(F̲̲); check_args=false)
    # because according to this discussion
    # (https://discourse.julialang.org/t/making-turing-fast-with-large-numbers-of-parameters/69072/78?u=dlakelan)
    # broadcasting does not work well when using ReverseDiff.jl
    Turing.@addlogprob! sum(
        Turing.logpdf.(
            Turing.Multinomial.(n̲ₜ, eachrow(F̲̲); check_args=false),
            R̲̲
        ),
    )

    ## %%%%%%%%%%%%%% Log-Likelihood functions %%%%%%%%%%%%%% ##

    # Sample posterior for neutral lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ⁿ⁾| sₜ, σₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            repeat(-s̲ₜ, n_neutral),
            LinearAlgebra.Diagonal(repeat(σ̲ₜ .^ 2, n_neutral))
        ),
        Γ̲̲⁽ⁿ⁾
    )

    # Sample posterior for nutant lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ᵐ⁾ | s⁽ᵐ⁾, σ⁽ᵐ⁾, s̲ₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            # Build vector for fitness differences
            s̲⁽ᵐ⁾[env_idx[2:end]] .- s̲ₜ,
            # Build vector for variances
            LinearAlgebra.Diagonal(σ̲⁽ᵐ⁾[env_idx[2:end]] .^ 2)
        ),
        γ̲⁽ᵐ⁾
    )
    return F̲̲
end # @model function