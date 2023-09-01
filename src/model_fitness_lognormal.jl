# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Joint inference π(s̲⁽ᵐ⁾, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
fitness_lognormal(R̲̲::Matrix{Int64}, n̲ₜ::Vector{Int64}, n_neutral::Int,
                  n_bc::Int; kwargs...)

Defines a model to estimate fitness effects in a competitive fitness experiment
across growth-dilution cycles.

# Arguments
- `R̲̲::Matrix{Int64}`:: `T × B` matrix--split into a vector of vectors for
  computational efficiency--where `T` is the number of time points in the data
  set and `B` is the number of barcodes. Each column represents the barcode
  count trajectory for a single lineage.
- `n̲ₜ::Vector{Int64}`: Vector with the total number of barcode counts for each
  time point. **NOTE**: This vector **must** be equivalent to computing
  `vec(sum(R̲̲, dims=2))`.
- `n_neutral::Int`: Number of neutral lineages in dataset. 
- `n_bc::Int`: Number of mutant lineages in dataset.

## Optional Keyword Arguments
- `s_pop_prior::VecOrMat{Float64}=[0.0, 2.0]`: Vector or Matrix with the
  corresponding parameters (Vector: `s_pop_prior[1]` = mean, `s_pop_prior[2]` =
  standard deviation, Matrix: `s_pop_prior[:, 1]` = mean, `s_pop_prior[:, 2]` =
  standard deviation) for a Normal prior on the population mean fitness values.
  If `typeof(s_pop_prior) <: Matrix`, there should be as many rows in the matrix
  as pairs of time adjacent time points in dataset.
- `σ_pop_prior::VecOrMat{Float64}=[0.0, 1.0]`: Vector or Matrix with the
  corresponding parameters (Vector: `σ_pop_prior[1]` = mean, `σ_pop_prior[2]` =
  standard deviation, Matrix: `logσ_pop_prior[:, 1]` = mean, `logσ_pop_prior[:,
  2]` = standard deviation) for a Log-Normal prior on the population mean
  fitness error utilized in the log-likelihood function. If
  `typeof(logσ_pop_prior) <: Matrix`, there should be as many rows in the matrix
  as pairs of time adjacent time points × number of replicates in dataset.
- `s_bc_prior::VecOrMat{Float64}=[0.0, 2.0]`: Vector or Matrix with the
  corresponding parameters (Vector: `s_bc_prior[1]` = mean, `s_bc_prior[2]` =
  standard deviation, Matrix: `s_bc_prior[:, 1]` = mean, `s_bc_prior[:, 2]` =
  standard deviation) for a Normal prior on the mutant fitness values. If
  `typeof(s_bc_prior) <: Matrix`, there should be as many rows in the matrix as
  number of mutant lineages × number of replicates in the dataset. 
- `σ_bc_prior::VecOrMat{Float64}=[0.0, 1.0]`: Vector or Matrix with the
  corresponding parameters (Vector: `σ_bc_prior[1]` = mean, `σ_bc_prior[2]` =
  standard deviation, Matrix: `σ_bc_prior[:, 1]` = mean, `σ_bc_prior[:, 2]` =
  standard deviation) for a Log-Normal prior on the mutant fitness error
  utilized in the log-likelihood function. If `typeof(logσ_bc_prior) <:
  Matrix`, there should be as many rows in the matrix as mutant lineages ×
  number of replicates in the dataset.
- `λ_prior::VecOrMat{Float64}=[3.0, 3.0]`: Vector or Matrix with the
  corresponding parameters (Vector: `λ_prior[1]` = mean, `λ_prior[2]` = standard
  deviation, Matrix: `λ_prior[:, 1]` = mean, `λ_prior[:, 2]` = standard
  deviation) for a Log-Normal prior on the λ parameter in the Poisson
  distribution. The λ parameter can be interpreted as the mean number of barcode
  counts since we assume any barcode count `n⁽ᵇ⁾ ~ Poisson(λ⁽ᵇ⁾)`. If
  `typeof(λ_prior) <: Matrix`, there should be as many rows in the matrix as
  number of barcodes × number of time points × number of replicates in the
  dataset.

## Latent Variables 
- Population mean fitness per timepoint.
- Mutant relative fitness.
- λ dispersion parameters per barcode and timepoint.

## Notes
- Models fitness effects as normally distributed. 
- Utilizes a Poisson observation model for barcode counts.
- Setting informative priors is recommended for stable convergence.
"""
Turing.@model function fitness_lognormal(
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64},
    n_neutral::Int,
    n_bc::Int;
    s_pop_prior::VecOrMat{Float64}=[0.0, 2.0],
    σ_pop_prior::VecOrMat{Float64}=[0.0, 1.0],
    s_bc_prior::VecOrMat{Float64}=[0.0, 2.0],
    σ_bc_prior::VecOrMat{Float64}=[0.0, 1.0],
    λ_prior::VecOrMat{Float64}=[3.0, 3.0]
)

    # Extract number of time points
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
            repeat([s_bc_prior[1]], n_bc),
            LinearAlgebra.I(n_bc) .* s_bc_prior[2] .^ 2
        )
    elseif typeof(s_bc_prior) <: Matrix
        s̲⁽ᵐ⁾ ~ Turing.MvNormal(
            s_bc_prior[:, 1], LinearAlgebra.Diagonal(s_bc_prior[:, 2] .^ 2)
        )
    end # if


    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    if typeof(σ_bc_prior) <: Vector
        σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
            repeat([σ_bc_prior[1]], n_bc),
            LinearAlgebra.I(n_bc) .* σ_bc_prior[2] .^ 2
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
    # n̲ₜ ~ Turing.arraydist(Turing.Poisson.(sum.(eachrow(Λ̲̲))))
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
            vcat([s⁽ᵐ⁾ .- s̲ₜ for s⁽ᵐ⁾ in s̲⁽ᵐ⁾]...),
            # Build vector for variances
            LinearAlgebra.Diagonal(
                vcat([repeat([σ], n_time - 1) for σ in σ̲⁽ᵐ⁾]...) .^ 2
            )
        ),
        Γ̲̲⁽ᵐ⁾
    )
    return
end # @model function