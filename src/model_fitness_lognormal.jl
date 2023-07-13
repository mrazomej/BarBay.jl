# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Joint inference π(s̲⁽ᵐ⁾, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    fitness_lognormal(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲, n̲ₜ; s_pop_prior, σ_pop_prior, s_mut_prior, σ_mut_prior, λ_prior)

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
Turing.@model function fitness_lognormal(
    R̲̲⁽ⁿ⁾::Matrix{Int64},
    R̲̲⁽ᵐ⁾::Matrix{Int64},
    R̲̲::Vector{Vector{Int64}},
    n̲ₜ::Vector{Int64};
    s_pop_prior::Vector{Float64}=[0.0, 2.0],
    σ_pop_prior::Vector{Float64}=[0.0, 1.0],
    s_mut_prior::Vector{Float64}=[0.0, 2.0],
    σ_mut_prior::Vector{Float64}=[0.0, 1.0],
    λ_prior::VecOrMat{Float64}=[3.0, 3.0]
)
    # Define number of time points
    n_time = length(n̲ₜ)
    # Define number of neutrals
    n_neutral = size(R̲̲⁽ⁿ⁾, 2)
    # Define numbero f mutants
    n_mut = size(R̲̲⁽ᵐ⁾, 2)

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
        repeat([s_mut_prior[1]], n_mut),
        LinearAlgebra.I(n_mut) .* s_mut_prior[2] .^ 2
    )
    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
        repeat([σ_mut_prior[1]], n_mut),
        LinearAlgebra.I(n_mut) .* σ_mut_prior[2] .^ 2
    )


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
    # n̲ₜ ~ Turing.arraydist(Turing.Poisson.(sum.(eachrow(Λ̲̲))))
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
            vcat([s⁽ᵐ⁾ .- s̲ₜ for s⁽ᵐ⁾ in s̲⁽ᵐ⁾]...),
            # Build vector for variances
            LinearAlgebra.Diagonal(
                vcat([repeat([σ], n_time - 1) for σ in σ̲⁽ᵐ⁾]...) .^ 2
            )
        ),
        Γ̲̲⁽ᵐ⁾
    )
    return F̲̲
end # @model function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Single-mutant inference π(s⁽ᵐ⁾, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    fitness_lognormal(R̲̲⁽ⁿ⁾, r̲⁽ᵐ⁾, R̲̲, n̲ₜ; s_pop_prior, σ_pop_prior, s_mut_prior, σ_mut_prior, λ_prior)

`Turing.jl` model to sample the joint posterior distribution for a competitive
fitness experiment for the neutral barcodes and *a single mutant barcode*

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
- `R̲̲::Vector{Vector{Int64}}`:: `T × B` matrix--split into a vector of vectors
for computational efficiency--where `T` is the number of time points in the data
  set and `B` is the number of barcodes. Each column represents the barcode
  count trajectory for a single lineage.
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
Turing.@model function fitness_lognormal(
    R̲̲⁽ⁿ⁾::Matrix{Int64},
    r̲⁽ᵐ⁾::Vector{Int64},
    R̲̲::Vector{Vector{Int64}},
    n̲ₜ::Vector{Int64};
    s_pop_prior::Vector{Float64}=[0.0, 2.0],
    σ_pop_prior::Vector{Float64}=[0.0, 1.0],
    s_mut_prior::Vector{Float64}=[0.0, 2.0],
    σ_mut_prior::Vector{Float64}=[0.0, 1.0],
    λ_prior::VecOrMat{Float64}=[3.0, 3.0]
)
    # Define number of time points
    n_time = length(n̲ₜ)
    # Define number of neutrals
    n_neutral = size(R̲̲⁽ⁿ⁾, 2)

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
    s⁽ᵐ⁾ ~ Turing.Normal(s_mut_prior[1], s_mut_prior[2])
    # Prior on LogNormal error π(σ̲⁽ᵐ⁾)
    σ⁽ᵐ⁾ ~ Turing.LogNormal(σ_mut_prior[1], σ_mut_prior[2])

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
    γ̲⁽ᵐ⁾ = Γ̲̲[:, size(R̲̲⁽ⁿ⁾, 2)+1]

    # Prob of total number of barcodes read given the Poisosn distribution
    # parameters π(nₜ | λ̲ₜ)
    # n̲ₜ ~ Turing.arraydist(Turing.Poisson.(sum.(eachrow(Λ̲̲))))
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
            s⁽ᵐ⁾ .- s̲ₜ,
            LinearAlgebra.Diagonal(repeat([σ⁽ᵐ⁾^2], n_time - 1))
        ),
        γ̲⁽ᵐ⁾
    )
    return F̲̲
end # @model function