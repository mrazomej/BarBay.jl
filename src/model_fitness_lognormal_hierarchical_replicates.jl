# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Hierarchical model for multiple experimental replicates π(θ̲ᴹ, s̲ᴹ, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    fitness_lognormal(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲, n̲ₜ; s_pop_prior, σ_pop_prior, s_mut_prior, σ_mut_prior, λ_prior)

`Turing.jl` model to sample the joint posterior distribution for a competitive
fitness experiment.

# Model
`[write model here]`

# Arguments
- `R̲̲⁽ⁿ⁾::Array{Int64, 3}`: `T × N × R` array where `T` is the number of time
  points in the data set, `N` is the number of neutral lineage barcodes, and `R`
  is the number of experimental replicates. For each slice on the `R`-axis, each
  column represents the barcode count trajectory for a single neutral lineage.
  **NOTE**: The model assumes the rows are sorted in order of increasing time.
- `R̲̲⁽ᵐ⁾::Array{Int64, 3}`: `T × M × R` array where `T` is the number of time
points in the data set, `M` is the number of mutant lineage barcodes, and `R` is
the number of experimental replicates. For each slice on the `R`-axis, each
  column represents the barcode count trajectory for a single mutant lineage.
  **NOTE**: The model assumes the rows are sorted in order of increasing time.
- `R̲̲::Array{Int64, 3}`:: `T × B × R` where `T` is the number of time points in
  the data set, `B` is the number of barcodes, and `R` is the number of
  experimental replicates. For each slince in the `R` axis, each column
  represents the barcode count trajectory for a single lineage. **NOTE**: This
  matrix does not necessarily need to be equivalent to `hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾)`.
  This is because `R̲̲⁽ᵐ⁾` can exclude mutant barcodes to perform the joint
  inference only for a subgroup, but `R̲̲` must still contain all counts.
  Usually, if `R̲̲⁽ᵐ⁾` excludes mutant barcodes, `R̲̲` must be of the form
  `hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲⁽ᴹ⁾)`, where `R̲̲⁽ᴹ⁾` is a vector that aggregates all
  excluded mutant barcodes into a "super barcode."
- `n̲ₜ::Vector{Vector{Int64}}`: `R` Vectors with the total number of barcode
  counts for each time point on each experimental replicate. **NOTE**: Each
  vector vector **must** be equivalent to computing `vec(sum(R̲̲, dims=2))`. The
  reason it is an independent input parameter is to avoid the `sum` computation
  within the `Turing` model.

## Optional Keyword Arguments
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
    adjacent time points × number of replicates in dataset.
- `s_mut_prior::VecOrMat{Float64}=[0.0, 2.0]`: Vector or Matrix with the
    correspnding parameters (Vector: `s_mut_prior[1]` = mean, `s_mut_prior[2]` =
    standard deviation, Matrix: `s_mut_prior[:, 1] = mean`, `s_mut_prior[:, 2] =
    standard deviation`) for a Normal prior on the mutant fitness values. If
    `typeof(s_mut_prior) <: Matrix`, there should be as many rows in the matrix
    as number of mutant lineages × number of replicates in the dataset.
- `σ_mut_prior::VecOrMat{Float64}=[0.0, 1.0]`: Vector or Matrix with the
  correspnding parameters (Vector: `s_mut_prior[1]` = mean, `s_mut_prior[2]` =
  standard deviation, Matrix: `s_mut_prior[:, 1] = mean`, `s_mut_prior[:, 2] =
  standard deviation`) for a Log-Normal prior on the mutant fitness error
  utilized in the log-likelihood function. If `typeof(σ_mut_prior) <: Matrix`,
  there should be as many rows in the matrix as mutant lineages × number of
  replicates in the dataset.
- `λ_prior::VecOrMat{Float64}=[3.0, 3.0]`: Vector or Matrix with the
  correspnding parameters (Vector: `λ_prior[1]` = mean, `λ_prior[2]` = standard
  deviation, Matrix: `λ_prior[:, 1] = mean`, `λ_prior[:, 2] = standard
  deviation`) for a Log-Normal prior on the λ parameter in the Poisson
  distribution. The λ parameter can be interpreted as the mean number of barcode
  counts since we assume any barcode count `n⁽ᵇ⁾ ~ Poisson(λ⁽ᵇ⁾)`. If
  `typeof(λ_prior) <: Matrix`, there should be as many rows in the matrix as
  number of barcodes × number of time points × number of replicates in the
  dataset.
"""
Turing.@model function fitness_lognormal_hierarchical_replicates(
    R̲̲⁽ⁿ⁾::Array{Int64,3},
    R̲̲⁽ᵐ⁾::Array{Int64,3},
    R̲̲::Array{Int64,3},
    n̲ₜ::Matrix{Int64};
    s_pop_prior::VecOrMat{Float64}=[0.0, 2.0],
    σ_pop_prior::VecOrMat{Float64}=[0.0, 1.0],
    s_mut_prior::VecOrMat{Float64}=[0.0, 2.0],
    σ_mut_prior::VecOrMat{Float64}=[0.0, 1.0],
    λ_prior::VecOrMat{Float64}=[3.0, 3.0],
    τ_prior::Vector{Float64}=[0.0, 1.0]
)
    # Define number of experimental replicates
    n_rep = size(R̲̲, 3)
    # Define number of time points
    n_time = size(R̲̲, 1)
    # Define number of neutrals
    n_neutral = size(R̲̲⁽ⁿ⁾, 2)
    # Define number of mutants
    n_mut = size(R̲̲⁽ᵐ⁾, 2)

    ## %%%%%%%%%%%%%% Population mean fitness  %%%%%%%%%%%%%% ##

    # Prior on population mean fitness π(s̲ₜ) 
    if typeof(s_pop_prior) <: Vector
        s̲ₜ ~ Turing.MvNormal(
            repeat([s_pop_prior[1]], (n_time - 1) * n_rep),
            LinearAlgebra.I((n_time - 1) * n_rep) .* s_pop_prior[2] .^ 2
        )
    elseif typeof(s_pop_prior) <: Matrix
        s̲ₜ ~ Turing.MvNormal(
            s_pop_prior[:, 1], LinearAlgebra.Diagonal(s_pop_prior[:, 2] .^ 2)
        )
    end # if

    # Prior on LogNormal error π(σ̲ₜ)
    if typeof(σ_pop_prior) <: Vector
        σ̲ₜ ~ Turing.MvLogNormal(
            repeat([σ_pop_prior[1]], (n_time - 1) * n_rep),
            LinearAlgebra.I((n_time - 1) * n_rep) .* σ_pop_prior[2] .^ 2
        )
    elseif typeof(σ_pop_prior) <: Matrix
        σ̲ₜ ~ Turing.MvNormal(
            σ_pop_prior[:, 1], LinearAlgebra.Diagonal(σ_pop_prior[:, 2] .^ 2)
        )
    end # if

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Hyper prior on mutant fitness π(θ̲⁽ᵐ⁾) 
    if typeof(s_mut_prior) <: Vector
        θ̲⁽ᵐ⁾ ~ Turing.MvNormal(
            repeat([s_mut_prior[1]], n_mut),
            LinearAlgebra.I(n_mut) .* s_mut_prior[2] .^ 2
        )
    elseif typeof(s_mut_prior) <: Matrix
        θ̲⁽ᵐ⁾ ~ Turing.MvNormal(
            s_mut_prior[:, 1], LinearAlgebra.Diagonal(s_mut_prior[:, 2] .^ 2)
        )
    end # if


    # Non-centered samples
    θ̲̃⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([0], n_mut * n_rep), LinearAlgebra.I(n_mut * n_rep)
    )

    # Hyper prior on mutant deviations from hyper prior
    τ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
        repeat([τ_prior[1]], n_mut * n_rep),
        LinearAlgebra.I(n_mut * n_rep) .* τ_prior[2] .^ 2
    )

    # mutant fitness = hyperparameter + deviation
    s̲⁽ᵐ⁾ = repeat(θ̲⁽ᵐ⁾, n_rep) .+ (τ̲⁽ᵐ⁾ .* θ̲̃⁽ᵐ⁾)

    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    if typeof(σ_mut_prior) <: Vector
        σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
            repeat([σ_mut_prior[1]], n_mut * n_rep),
            LinearAlgebra.I(n_mut * n_rep) .* σ_mut_prior[2] .^ 2
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
            repeat([λ_prior[1]], length(R̲̲)),
            LinearAlgebra.I(length(R̲̲)) .* λ_prior[2]^2
        )
    elseif typeof(λ_prior) <: Array{Int64,3}
        # Prior on Poisson distribtion parameters π(λ)
        Λ̲̲ ~ Turing.MvLogNormal(
            λ_prior[:, 1, :][:],
            LinearAlgebra.Diagonal(λ_prior[:, 2, :][:] .^ 2)
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
    Γ̲̲ = F̲̲[2:end, :, :] ./ F̲̲[1:end-1, :, :]

    # Split neutral and mutant frequency ratios. Note: the @view macro means
    # that there is not allocation to memory on this step.
    Γ̲̲⁽ⁿ⁾ = vec(Γ̲̲[:, 1:n_neutral, :])
    Γ̲̲⁽ᵐ⁾ = vec(Γ̲̲[:, n_neutral+1:n_neutral+n_mut, :])

    # Loop through replicates
    for r = 1:n_rep
        # Prob of total number of barcodes read given the Poisosn distribution
        # parameters π(nₜ | λ̲ₜ)
        n̲ₜ[:, r] ~ Turing.arraydist(
            [Turing.Poisson(sum(Λ̲̲[t, :, r])) for t = 1:n_time]
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
                Turing.Multinomial.(
                    n̲ₜ[:, r], eachrow(F̲̲[:, :, r]); check_args=false
                ),
                eachrow(R̲̲[:, :, r])
            ),
        )
    end # for

    ## %%%%%%%%%%%% Reshape arrays to split replicate variables %%%%%%%%%%%% ##

    # Reshape to have a matrix with columns for each replicate
    s̲ₜ = reshape(s̲ₜ, :, n_rep)          # n_time × n_rep
    σ̲ₜ = reshape(σ̲ₜ, :, n_rep)          # n_time × n_rep
    s̲⁽ᵐ⁾ = reshape(s̲⁽ᵐ⁾, :, n_rep)     # n_mut × n_rep
    σ̲⁽ᵐ⁾ = reshape(σ̲⁽ᵐ⁾, :, n_rep)     # n_mut × n_rep

    ## %%%%%%%%%%%%%% Log-Likelihood functions %%%%%%%%%%%%%% ##

    # Sample posterior for neutral lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ⁿ⁾| sₜ, σₜ).
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            # Build array for MvLogNormal mean
            permutedims(cat(repeat([-s̲ₜ], n_neutral)..., dims=3), [1, 3, 2])[:],
            # Build array for MvLogNormal variance
            LinearAlgebra.Diagonal(
                permutedims(
                    cat(repeat([σ̲ₜ .^ 2], n_neutral)..., dims=3), [1, 3, 2]
                )[:]
            )
        ),
        Γ̲̲⁽ⁿ⁾
    )

    # Sample posterior for mutant lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ᵐ⁾ | s⁽ᵐ⁾, σ⁽ᵐ⁾, s̲ₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            # Build vector for fitness differences
            permutedims(
                cat(repeat([s̲⁽ᵐ⁾], (n_time - 1))..., dims=3), [3, 1, 2]
            )[:] .-
            permutedims(cat(repeat([s̲ₜ], n_mut)..., dims=3), [1, 3, 2])[:],
            # Build vector for variances
            LinearAlgebra.Diagonal(
                permutedims(
                    cat(repeat([σ̲⁽ᵐ⁾], (n_time - 1))..., dims=3), [3, 1, 2]
                )[:]
            )
        ),
        Γ̲̲⁽ᵐ⁾
    )

    return s̲⁽ᵐ⁾
end # @model function