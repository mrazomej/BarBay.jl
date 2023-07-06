# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# joint fitness inference in multiple environemnts π(s1⁽ᵐ⁾, s2⁽ᵐ⁾,.. | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    env_fitness_lognormal(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲, n̲ₜ; kwargs)

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
- `R̲̲::Matrix{Int64}`:: `T × B` matrix, where `T` is the number of time points
  in the data set and `B` is the number of barcodes. Each column represents the
  barcode count trajectory for a single lineage. **NOTE**: This matrix does not
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
- `s_pop_prior::Vector{Float64}=[0.0, 2.0]`: Vector with the correspnding
    parameters (`s_pop_prior[1]` = mean, `s_pop_prior[2]` = standard deviation)
    for a Normal prior on the population mean fitness values. **NOTE**: This
    method assigns the same prior to **all** population mean fitness to be
    inferred.
- `σ_pop_prior::Vector{Float64}=[0.0, 1.0]`: Vector with the correspnding
    parameters (`σ_pop_prior[1]` = mean, `σ_pop_prior[2]` = standard deviation)
    for a Log-Normal prior on the population mean fitness error utilized in the
    log-likelihood function. **NOTE**: This method assigns the same prior to
    **all** population mean fitness errors to be inferred.
- `s_mut_prior::Vector{Float64}=[0.0, 2.0]`: Vector with the correspnding
    parameters (`s_mut_prior[1]` = mean, `s_mut_prior[2]` = standard deviation)
    for a Normal prior on the mutant fitness values. **NOTE**: This method
    assigns the same prior to **all** mutant fitness values to be inferred.
- `σ_mut_prior::Vector{Float64}=[0.0, 1.0]`: Vector with the correspnding
    parameters (`σ_mut_prior[1]` = mean, `σ_mut_prior[2]` = standard deviation)
    for a Log-Normal prior on the mutant fitness error utilized in the
    log-likelihood function. **NOTE**: This method assigns the same prior to
    **all** mutant fitness error values to be inferred.
- `λ_prior::Vector{Float64}=[3.0, 3.0]`: Vector with the corresponding
  parameters (`λ_prior[1]` = mean, `λ_prior[2]` = standard deviation) for a
  Log-Normal prior on the λ parameter in the Poisson distribution. The λ
  parameter can be interpreted as the mean number of barcode counts since we
  assume any barcode count `n⁽ᵇ⁾ ~ Poisson(λ⁽ᵇ⁾)`. **NOTE**: This method assigns
    the same prior to **all** mutant fitness error values to be inferred.
"""
Turing.@model function env_fitness_lognormal(
    R̲̲⁽ⁿ⁾::Matrix{Int64},
    R̲̲⁽ᵐ⁾::Matrix{Int64},
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64};
    envs::Vector{<:Any},
    s_pop_prior::Vector{Float64}=[0.0, 2.0],
    σ_pop_prior::Vector{Float64}=[0.0, 1.0],
    s_mut_prior::Vector{Float64}=[0.0, 2.0],
    σ_mut_prior::Vector{Float64}=[0.0, 1.0],
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

    ## %%%%%%%%%%%%%% Population mean fitness  %%%%%%%%%%%%%% ##

    # Prior on population mean fitness π(s̲ₜ) 
    s̲ₜ ~ Turing.MvNormal(
        repeat([s_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* s_pop_prior[2] .^ 2
    )
    # Prior on LogNormal error π(σ̲ₜ)
    σ̲ₜ ~ Turing.MvLogNormal(
        repeat([σ_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* σ_pop_prior[2] .^ 2
    )

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Prior on mutant fitness π(s̲⁽ᵐ⁾)
    s̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([s_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2) * n_env),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2) * n_env) .* s_mut_prior[2] .^ 2
    )
    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
        repeat([σ_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2) * n_env),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2) * n_env) .* σ_mut_prior[2] .^ 2
    )


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
    Γ̲̲⁽ⁿ⁾ = @view Γ̲̲[:, 1:size(R̲̲⁽ⁿ⁾, 2)]
    Γ̲̲⁽ᵐ⁾ = @view Γ̲̲[:, size(R̲̲⁽ⁿ⁾, 2)+1:size(R̲̲⁽ⁿ⁾, 2)+size(R̲̲⁽ᵐ⁾, 2)]

    # Prob of total number of barcodes read given the Poisosn distribution
    # parameters π(nₜ | λ̲ₜ)
    n̲ₜ ~ Turing.arraydist([Turing.Poisson(sum(Λ̲̲[t, :])) for t = 1:size(R̲̲⁽ⁿ⁾, 1)])

    # Loop through time points
    for t = 1:size(R̲̲⁽ⁿ⁾, 1)
        # Prob of reads given parameters π(R̲ₜ | nₜ, f̲ₜ). Note: We add the
        # check_args=false option to avoid the recurrent problem of
        # > Multinomial: p is not a probability vector.
        # due to rounding errors
        R̲̲[t, :] ~ Turing.Multinomial(n̲ₜ[t], F̲̲[t, :]; check_args=false)
    end # for

    ## %%%%%%%%%%%% Reshape arrays to split replicate variables %%%%%%%%%%%% ##
    s̲⁽ᵐ⁾ = reshape(s̲⁽ᵐ⁾, n_env, :)  # n_env × n_mut
    σ̲⁽ᵐ⁾ = reshape(σ̲⁽ᵐ⁾, n_env, :)  # n_env × n_mut

    ## %%%%%%%%%%%%%% Log-Likelihood functions %%%%%%%%%%%%%% ##

    # Sample posterior for neutral lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ⁿ⁾| sₜ, σₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            repeat(-s̲ₜ, size(Γ̲̲⁽ⁿ⁾, 2)),
            LinearAlgebra.Diagonal(repeat(σ̲ₜ .^ 2, size(Γ̲̲⁽ⁿ⁾, 2)))
        ),
        Γ̲̲⁽ⁿ⁾[:]
    )

    # Sample posterior for nutant lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ᵐ⁾ | s⁽ᵐ⁾, σ⁽ᵐ⁾, s̲ₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            # Build vector for fitness differences
            vcat([s⁽ᵐ⁾[env_idx] .- s̲ₜ for s⁽ᵐ⁾ in eachcol(s̲⁽ᵐ⁾)]...),
            # Build vector for variances
            LinearAlgebra.Diagonal(
                vcat([σ⁽ᵐ⁾[env_idx] for σ⁽ᵐ⁾ in eachcol(σ̲⁽ᵐ⁾)]...) .^ 2
            )
        ),
        Γ̲̲⁽ᵐ⁾[:]
    )
    return F̲̲
end # @model function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# single mutant fitness inference in multiple environemnts 
# π(s1⁽ᵐ⁾, s2⁽ᵐ⁾,.. | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    env_fitness_lognormal(R̲̲⁽ⁿ⁾, r̲⁽ᵐ⁾, R̲̲, n̲ₜ; kwargs)

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
- `s_pop_prior::Vector{Float64}=[0.0, 2.0]`: Vector with the correspnding
    parameters (`s_pop_prior[1]` = mean, `s_pop_prior[2]` = standard deviation)
    for a Normal prior on the population mean fitness values. **NOTE**: This
    method assigns the same prior to **all** population mean fitness to be
    inferred.
- `σ_pop_prior::Vector{Float64}=[0.0, 1.0]`: Vector with the correspnding
    parameters (`σ_pop_prior[1]` = mean, `σ_pop_prior[2]` = standard deviation)
    for a Log-Normal prior on the population mean fitness error utilized in the
    log-likelihood function. **NOTE**: This method assigns the same prior to
    **all** population mean fitness errors to be inferred.
- `s_mut_prior::Vector{Float64}=[0.0, 2.0]`: Vector with the correspnding
    parameters (`s_mut_prior[1]` = mean, `s_mut_prior[2]` = standard deviation)
    for a Normal prior on the mutant fitness values. **NOTE**: This method
    assigns the same prior to **all** mutant fitness values to be inferred.
- `σ_mut_prior::Vector{Float64}=[0.0, 1.0]`: Vector with the correspnding
    parameters (`σ_mut_prior[1]` = mean, `σ_mut_prior[2]` = standard deviation)
    for a Log-Normal prior on the mutant fitness error utilized in the
    log-likelihood function. **NOTE**: This method assigns the same prior to
    **all** mutant fitness error values to be inferred.
- `λ_prior::Vector{Float64}=[3.0, 3.0]`: Vector with the corresponding
  parameters (`λ_prior[1]` = mean, `λ_prior[2]` = standard deviation) for a
  Log-Normal prior on the λ parameter in the Poisson distribution. The λ
  parameter can be interpreted as the mean number of barcode counts since we
  assume any barcode count `n⁽ᵇ⁾ ~ Poisson(λ⁽ᵇ⁾)`. **NOTE**: This method assigns
    the same prior to **all** mutant fitness error values to be inferred.
"""
Turing.@model function env_fitness_lognormal(
    R̲̲⁽ⁿ⁾::Matrix{Int64},
    r̲⁽ᵐ⁾::Vector{Int64},
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64};
    envs::Vector{<:Any},
    s_pop_prior::Vector{Float64}=[0.0, 2.0],
    σ_pop_prior::Vector{Float64}=[0.0, 1.0],
    s_mut_prior::Vector{Float64}=[0.0, 2.0],
    σ_mut_prior::Vector{Float64}=[0.0, 1.0],
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
    s̲ₜ ~ Turing.MvNormal(
        repeat([s_pop_prior[1]], n_time - 1),
        LinearAlgebra.I(n_time - 1) .* s_pop_prior[2] .^ 2
    )
    # Prior on LogNormal error π(σ̲ₜ)
    σ̲ₜ ~ Turing.MvLogNormal(
        repeat([σ_pop_prior[1]], n_time - 1),
        LinearAlgebra.I(n_time - 1) .* σ_pop_prior[2] .^ 2
    )

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Prior on mutant fitness π(s̲⁽ᵐ⁾)
    s̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([s_mut_prior[1]], n_env),
        LinearAlgebra.I(n_env) .* s_mut_prior[2] .^ 2
    )
    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
        repeat([σ_mut_prior[1]], n_env),
        LinearAlgebra.I(n_env) .* σ_mut_prior[2] .^ 2
    )

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
    Γ̲̲⁽ⁿ⁾ = @view Γ̲̲[:, 1:size(R̲̲⁽ⁿ⁾, 2)]
    γ̲⁽ᵐ⁾ = @view Γ̲̲[:, size(R̲̲⁽ⁿ⁾, 2)+1]

    # Prob of total number of barcodes read given the Poisosn distribution
    # parameters π(nₜ | λ̲ₜ)
    n̲ₜ ~ Turing.arraydist([Turing.Poisson(sum(Λ̲̲[t, :])) for t = 1:n_time])

    # Loop through time points
    for t = 1:n_time
        # Prob of reads given parameters π(R̲ₜ | nₜ, f̲ₜ). Note: We add the
        # check_args=false option to avoid the recurrent problem of
        # > Multinomial: p is not a probability vector.
        # due to rounding errors
        R̲̲[t, :] ~ Turing.Multinomial(n̲ₜ[t], F̲̲[t, :]; check_args=false)
    end # for

    ## %%%%%%%%%%%%%% Log-Likelihood functions %%%%%%%%%%%%%% ##

    # Sample posterior for neutral lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ⁿ⁾| sₜ, σₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            repeat(-s̲ₜ, size(Γ̲̲⁽ⁿ⁾, 2)),
            LinearAlgebra.Diagonal(repeat(σ̲ₜ .^ 2, size(Γ̲̲⁽ⁿ⁾, 2)))
        ),
        Γ̲̲⁽ⁿ⁾[:]
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
        γ̲⁽ᵐ⁾[:]
    )
    return F̲̲
end # @model function