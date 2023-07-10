@doc raw"""
    fitness_lognormal(R̲̲, R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, n̲ₜ; s_pop_prior, σ_pop_prior, s_mut_prior, σ_mut_prior, λ_prior)

`Turing.jl` model to sample the joint posterior distribution for a competitive
fitness experiment.

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
  R̲̲⁽ᴹ⁾)`, where `R̲̲⁽ᴹ⁾` is a vector that aggregates all excluded mutant barcodes
  into a "super barcode."
- `n̲ₜ::Vector{Int64}`: Vector with the total number of barcode counts for each
  time point. **NOTE**: This vector **must** be equivalent to computing
  `vec(sum(R̲̲, dims=2))`. The reason it is an independent input parameter is to
  avoid the `sum` computation within the `Turing` model.

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
Turing.@model function fitness_normal(
    R̲̲⁽ⁿ⁾::Matrix{Int64},
    R̲̲⁽ᵐ⁾::Matrix{Int64},
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64};
    s_pop_prior::Vector{Float64}=[0.0, 2.0],
    σ_pop_prior::Vector{Float64}=[0.0, 1.0],
    s_mut_prior::Vector{Float64}=[0.0, 2.0],
    σ_mut_prior::Vector{Float64}=[0.0, 1.0],
    λ_prior::VecOrMat{Float64}=[3.0, 3.0]
)
    ## %%%%%%%%%%%%%% Population mean fitness  %%%%%%%%%%%%%% ##

    # Prior on population mean fitness π(s̲ₜ) 
    s̲ₜ ~ Turing.MvNormal(
        repeat([s_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* s_pop_prior[2] .^ 2
    )
    # Prior on LogNormal error π(σ̲ₜ)
    logσ̲ₜ ~ Turing.MvNormal(
        repeat([σ_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* σ_pop_prior[2] .^ 2
    )

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Prior on mutant fitness π(s̲⁽ᵐ⁾)
    s̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([s_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* s_mut_prior[2] .^ 2
    )
    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    logσ̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([σ_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* σ_mut_prior[2] .^ 2
    )


    ## %%%%%%%%%%%%%% Barcode frequencies %%%%%%%%%%%%%% ##

    if typeof(λ_prior) <: Vector
        # Prior on Poisson distribtion parameters π(λ)
        logΛ̲̲ ~ Turing.MvNormal(
            repeat([λ_prior[1]], length(R̲̲)),
            LinearAlgebra.I(length(R̲̲)) .* λ_prior[2]^2
        )
    elseif typeof(λ_prior) <: Matrix
        # Prior on Poisson distribtion parameters π(λ)
        logΛ̲̲ ~ Turing.MvNormal(
            λ_prior[:, 1], LinearAlgebra.Diagonal(λ_prior[:, 2] .^ 2)
        )
    end  # if

    # Reshape λ parameters to fit the matrix format. Note: The Λ̲̲ array is
    # originally sampled as a vector for the `Turing.jl` samplers to deal with
    # it. But reshaping it to a matrix simplifies the computation of frequencies
    # and frequency ratios.
    Λ̲̲ = reshape(exp.(logΛ̲̲), size(R̲̲)...)

    # Compute barcode frequencies from Poisson parameters
    F̲̲ = Λ̲̲ ./ sum(Λ̲̲, dims=2)

    # Compute frequency ratios between consecutive time points.
    logΓ̲̲ = log.(F̲̲[2:end, :] ./ F̲̲[1:end-1, :])

    # Split neutral and mutant frequency ratios. Note: the @view macro means
    # that there is not allocation to memory on this step.
    logΓ̲̲⁽ⁿ⁾ = @view logΓ̲̲[:, 1:size(R̲̲⁽ⁿ⁾, 2)]
    logΓ̲̲⁽ᵐ⁾ = @view logΓ̲̲[:, size(R̲̲⁽ⁿ⁾, 2)+1:size(R̲̲⁽ⁿ⁾, 2)+size(R̲̲⁽ᵐ⁾, 2)]

    # Prob of total number of barcodes read given the Poisosn distribution
    # parameters π(nₜ | λ̲ₜ)
    n̲ₜ ~ Turing.arraydist(
        [Turing.Poisson(sum(Λ̲̲[t, :]); check_args=false) for t = 1:size(R̲̲⁽ⁿ⁾, 1)]
    )

    # Loop through time points
    for t = 1:size(R̲̲⁽ⁿ⁾, 1)
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
        Turing.MvNormal(
            repeat(-s̲ₜ, size(logΓ̲̲⁽ⁿ⁾, 2)),
            LinearAlgebra.Diagonal(repeat(exp.(logσ̲ₜ) .^ 2, size(logΓ̲̲⁽ⁿ⁾, 2)))
        ),
        logΓ̲̲⁽ⁿ⁾[:]
    )

    # Sample posterior for nutant lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ᵐ⁾ | s⁽ᵐ⁾, σ⁽ᵐ⁾, s̲ₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvNormal(
            # Build vector for fitness differences
            vcat([s⁽ᵐ⁾ .- s̲ₜ for s⁽ᵐ⁾ in s̲⁽ᵐ⁾]...),
            # Build vector for variances
            LinearAlgebra.Diagonal(
                vcat([repeat([σ], length(s̲ₜ)) for σ in exp.(logσ̲⁽ᵐ⁾)]...) .^ 2
            )
        ),
        logΓ̲̲⁽ᵐ⁾[:]
    )
    return F̲̲
end # @model function