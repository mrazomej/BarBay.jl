@doc raw"""
    neutrals_lognormal(R̲̲, R̲̲⁽ⁿ⁾, n̲ₜ; s_pop_prior, σ_pop_prior, λ_prior)

`Turing.jl` model to sample the joint posterior distribution of the population
mean fitness for a competitive fitness experiment using only the neutral
lineages.

# Model
`[write model here]`

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
- `n_bc::Int`: Number of mutant lineages in datset. **NOTE** This argument is
  irrelevant for this function. It is only included to have consistent inputs
  across models.

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
- `λ_prior::VecOrMat{Float64}=[3.0, 3.0]`: Vector or Matrix with the
    correspnding parameters (Vector: `λ_prior[1]` = mean, `λ_prior[2]` =
    standard deviation, Matrix: `λ_prior[:, 1] = mean`, `λ_prior[:, 2] =
    standard deviation`) for a Log-Normal prior on the λ parameter in the
    Poisson distribution. The λ parameter can be interpreted as the mean number
    of barcode counts since we assume any barcode count `n⁽ᵇ⁾ ~ Poisson(λ⁽ᵇ⁾)`.
    If `typeof(λ_prior) <: Matrix`, there should be as many rows in the matrix
    as number of barcodes × number of time points in the dataset.
"""
Turing.@model function neutrals_lognormal(
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64},
    n_neutral::Int,
    n_bc::Int=1;
    s_pop_prior::VecOrMat{Float64}=[0.0, 2.0],
    σ_pop_prior::VecOrMat{Float64}=[0.0, 1.0],
    λ_prior::VecOrMat{Float64}=[3.0, 3.0]
)
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
    return F̲̲
end # @model function