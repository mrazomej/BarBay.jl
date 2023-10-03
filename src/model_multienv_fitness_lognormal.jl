# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# joint fitness inference in multiple environemnts π(s1⁽ᵐ⁾, s2⁽ᵐ⁾,.. | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    multienv_fitness_lognormal(R̲̲, n̲ₜ, n_neutral, n_bc; kwargs)

Defines a model to estimate fitness effects in a competitive fitness experiment
with different environments across growth-dilution cycles.

# Model summary

Note: All multivariate normal distributions listed below have diagonal
covariance matrices. This is equivalent to independent normal random variables,
but evaluation and sampling is much more computationally efficient.

- Prior on population mean fitness `π(s̲ₜ)`

`s̲ₜ ~ MvNormal(params=s_pop_prior)`

- Prior on population mean fitness associated error `π(σ̲ₜ)`

`σ̲ₜ ~ MvLogNormal(params=σ_pop_prior)`

- Prior on non-neutral relative fitness `π(sᵢ⁽ᵐ⁾)` (subindex `i` indicates the
  environment).

`sᵢ⁽ᵐ⁾ ~ Normal(params=s_bc_prior)`

- Prior on non-neutral relative fitness associated error `π(σᵢ⁽ᵐ⁾)` (subindex
  `i` indicates the environment)

`σᵢ⁽ᵐ⁾ ~ LogNormal(params=σ_bc_prior)`

- Prior on Poisson distribtion parameters `π(λ)` (sampled as a `T × B` matrix
  for each of the `B` barcodes over `T` time points)

`Λ̲̲ ~ MvLogNormal(params=logλ_prior)`

- Probability of total number of barcodes read given the Poisson distribution
  parameters `π(nₜ | λ̲ₜ)`

`nₜ ~ Poisson(∑ₜ λₜ)`

- Barcode frequencies (deterministic relationship from the Poisson parameters)

`fₜ⁽ⁱ⁾ = λₜ⁽ⁱ⁾ / ∑ⱼ λₜ⁽ʲ⁾`

- frequency ratios (deterministic relationship from barcode frequencies)

`γₜ⁽ⁱ⁾ = fₜ₊₁⁽ⁱ⁾ / fₜ⁽ⁱ⁾`

- Probability of number of reads at time t for all barcodes given the total
  number of reads and the barcode frequencies `π(r̲ₜ | nₜ, f̲ₜ)`

`r̲ₜ ~ Multinomial(nₜ, f̲ₜ)`

- Probability of neutral barcodes frequency ratios `π(γₜ⁽ⁿ⁾| sₜ, σₜ)`

`logγₜ⁽ⁿ⁾ ~ Normal(μ = -sₜ, σ = exp(logσₜ))`

- Probability of non-neutral barcodes frequency ratios `π(logγₜ⁽ᵐ⁾| s⁽ᵐ⁾, σ⁽ᵐ⁾,
    sₜ)`. **Note**: This is done grouping by corresponding environment such that
    if time `t` is associated with environment `i`, sᵢ⁽ᵐ⁾ is used as the fitness
    value.

`logγₜ⁽ᵐ⁾ ~ Normal(μ = sᵢ⁽ᵐ⁾ - sₜ, σ = exp(σ⁽ᵐ⁾))`

# Arguments
- `R̲̲::Matrix{Int64}`:: `T × B` matrix--split into a vector of vectors for
  computational efficiency--where `T` is the number of time points in the data
  set and `B` is the number of barcodes. Each column represents the barcode
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
- `n_neutral::Int`: Number of neutral lineages in dataset.
- `n_bc::Int`: Number of mutant lineages in datset.

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
- `s_bc_prior::VecOrMat{Float64}=[0.0, 2.0]`: Vector or Matrix with the
    correspnding parameters (Vector: `s_bc_prior[1]` = mean, `s_bc_prior[2]` =
    standard deviation, Matrix: `s_bc_prior[:, 1] = mean`, `s_bc_prior[:, 2] =
    standard deviation`) for a Normal prior on the mutant fitness values. If
    `typeof(s_bc_prior) <: Matrix`, there should be as many rows in the matrix
    as mutant lineages × number of unique environments in the dataset.
- `σ_bc_prior::VecOrMat{Float64}=[0.0, 1.0]`: Vector or Matrix with the
  correspnding parameters (Vector: `s_bc_prior[1]` = mean, `s_bc_prior[2]` =
  standard deviation, Matrix: `s_bc_prior[:, 1] = mean`, `s_bc_prior[:, 2] =
  standard deviation`) for a Log-Normal prior on the mutant fitness error
  utilized in the log-likelihood function. If `typeof(σ_bc_prior) <: Matrix`,
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
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64},
    n_neutral::Int,
    n_bc::Int;
    envs::Vector{<:Any},
    s_pop_prior::VecOrMat{Float64}=[0.0, 2.0],
    σ_pop_prior::VecOrMat{Float64}=[0.0, 1.0],
    s_bc_prior::VecOrMat{Float64}=[0.0, 2.0],
    σ_bc_prior::VecOrMat{Float64}=[0.0, 1.0],
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
    if typeof(s_bc_prior) <: Vector
        s̲⁽ᵐ⁾ ~ Turing.MvNormal(
            repeat([s_bc_prior[1]], n_bc * n_env),
            LinearAlgebra.I(n_bc * n_env) .* s_bc_prior[2] .^ 2
        )
    elseif typeof(s_bc_prior) <: Matrix
        s̲⁽ᵐ⁾ ~ Turing.MvNormal(
            s_bc_prior[:, 1], LinearAlgebra.Diagonal(s_bc_prior[:, 2] .^ 2)
        )
    end # if

    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    if typeof(σ_bc_prior) <: Vector
        σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
            repeat([σ_bc_prior[1]], n_bc * n_env),
            LinearAlgebra.I(n_bc * n_env) .* σ_bc_prior[2] .^ 2
        )
    elseif typeof(σ_bc_prior) <: Matrix
        σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
            σ_bc_prior[:, 1], LinearAlgebra.Diagonal(σ_bc_prior[:, 2] .^ 2)
        )
    end # if


    ## %%%%%%%%%%%%%% Barcode frequencies %%%%%%%%%%%%%% ##

    if typeof(λ_prior) <: Vector
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            repeat([λ_prior[1]], length(R̲̲)),
            LinearAlgebra.I(length(R̲̲)) .* λ_prior[2]^2
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
    Γ̲̲⁽ᵐ⁾ = vec(Γ̲̲[:, n_neutral+1:n_neutral+n_bc])

    # Prob of total number of barcodes read given the Poisosn distribution
    # parameters π(nₜ | λ̲ₜ)
    n̲ₜ ~ Turing.arraydist(
        Turing.Poisson.(vec(sum(Λ̲̲, dims=2)), check_args=false)
    )

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
            eachrow(R̲̲)
        ),
    )

    ## %%%%%%%%%%%% Reshape arrays to split replicate variables %%%%%%%%%%%% ##
    s̲⁽ᵐ⁾ = reshape(s̲⁽ᵐ⁾, n_env, :)  # n_env × n_bc
    σ̲⁽ᵐ⁾ = reshape(σ̲⁽ᵐ⁾, n_env, :)  # n_env × n_bc

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