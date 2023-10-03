# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Hierarchical model for genotypes within single dataset π(θ̲ᴹ, s̲ᴹ, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
`genotype_fitness_normal(R̲̲::Vector{Matrix{Int64}}, n̲ₜ::Vector{Vector{Int64}},
                        n_neutral::Int, n_bc::Int; kwargs...)`

Defines a hierarchical model to estimate fitness effects in a competitive
fitness experiment where multiple barcodes belong to a specific "genotype." This
means that different barcodes are grouped together through a fitness
hyperparameter where each barcode samples from the distribution of this
hyperparameter.

# Model summary

Note: All multivariate normal distributions listed below have diagonal
covariance matrices. This is equivalent to independent normal random variables,
but evaluation and sampling is much more computationally efficient.

- Prior on population mean fitness `π(s̲ₜ)`

`s̲ₜ ~ MvNormal(params=s_pop_prior)`

- Prior on population *log* mean fitness associated error `π(logσ̲ₜ)`

`logσ̲ₜ ~ MvNormal(params=logσ_pop_prior)`

- Prior on non-neutral relative **hyper**-fitness `π(θ̲⁽ᵐ⁾)`

`θ̲⁽ᵐ⁾ ~ MvNormal(params=s_bc_prior)`

- prior on non-centered samples that allow local fitness to vary in the positive
  and negative direction for genotype `i` `π(θ̲̃ᵢ⁽ᵐ⁾)`. Note, this is a standard
  normal with mean zero and standard deviation one. 

`θ̲̃ᵢ⁽ᵐ⁾ ~ MvNormal(μ = 0̲, σ = I̲̲)`

- prior on *log* deviations of local fitness from hyper-fitness for genotype `i`
  π(logτ̲ᵢ⁽ᵐ⁾)

`logτ̲ᵢ⁽ᵐ⁾ ~ MvNormal(params=logτ_prior)`

- *local* relative fitness for non-neutral barcode `m` with genotype `i`
  (deterministic relationship from hyper-priors)

`sᵢ⁽ᵐ⁾ ~ θ̲⁽ᵐ⁾ + θ̲̃ᵢ⁽ᵐ⁾ * exp(logτ̲ᵢ⁽ᵐ⁾)`

- Prior on non-neutral *log* relative fitness associated error for non-neutrla
  barcode `m` with genotype `i`, `π(logσ̲ᵢ⁽ᵐ⁾)`

`logσ̲ᵢ⁽ᵐ⁾ ~ MvNormal(params=logσ_bc_prior)`

- Prior on *log* Poisson distribtion parameters `π(logλ)` (sampled as a `T × B`
  matrix for each of the `B` barcodes over `T` time points)

`logΛ̲̲ ~ MvLogNormal(params=logλ_prior)`

- Probability of total number of barcodes read given the Poisson distribution
  parameters `π(nₜ | exp(logλ̲ₜ))`

`nₜ ~ Poisson(∑ₜ exp(λₜ))`

- Barcode frequencies (deterministic relationship from the Poisson parameters)

`fₜ⁽ʲ⁾ = λₜ⁽ʲ⁾ / ∑ₖ λₜ⁽ᵏ⁾`

- *log* frequency ratios (deterministic relationship from barcode frequencies)

`logγₜ⁽ʲ⁾ = log(fₜ₊₁⁽ʲ⁾ / fₜ⁽ʲ⁾`)

- Probability of number of reads at time t for all barcodes given the total
  number of reads and the barcode frequencies `π(r̲ₜ | nₜ, f̲ₜ)`

`r̲ₜ ~ Multinomial(nₜ, f̲ₜ)`

- Probability of neutral barcodes frequency ratios `π(logγₜ⁽ⁿ⁾| sₜ, σₜ)`

`logγₜ⁽ⁿ⁾ ~ Normal(μ = -sₜ, σ = exp(logσₜ))`

- Probability of non-neutral barcodes frequency ratios for non-neutrla barcode
  `m` with genotype `i` `π(logγₜ⁽ᵐ⁾| sᵢ⁽ᵐ⁾, logσᵢ⁽ᵐ⁾,
  sₜ)`

`logγₜ⁽ᵐ⁾ ~ Normal(μ = sᵢ⁽ᵐ⁾ - sₜ, σ = exp(logσᵢ⁽ᵐ⁾))`

# Arguments
- `R̲̲::Vector{Matrix{Int64}}`:: Length `R` vector wth `T × B` matrices where
  `T` is the number of time points in the data set, `B` is the number of
  barcodes, and `R` is the number of experimental replicates. For each matrix in
  the vector, each column represents the barcode count trajectory for a single
  lineage.
- `n̲ₜ::Vector{Vector{Int64}}`: Vector of vectors with the total number of
  barcode counts for each time point on each replicate. **NOTE**: This vector
  **must** be equivalent to computing `vec.(sum.(R̲̲, dims=2))`.
- `n_neutral::Int`: Number of neutral lineages in dataset.  
- `n_bc::Int`: Number of mutant lineages in dataset.

## Keyword Arguments
- `genotypes::Vector{Vector{<:Any}}`: Vector with the list of genotypes for each
  non-neutral barcode. 

## Optional Keyword Arguments
- `s_pop_prior::VecOrMat{Float64}=[0.0, 2.0]`: Vector or Matrix with the
  corresponding parameters (Vector: `s_pop_prior[1]` = mean, `s_pop_prior[2]` =
  standard deviation, Matrix: `s_pop_prior[:, 1]` = mean, `s_pop_prior[:, 2]` =
  standard deviation) for a Normal prior on the population mean fitness values.
  If `typeof(s_pop_prior) <: Matrix`, there should be as many rows in the matrix
  as pairs of time adjacent time points in dataset.  
- `logσ_pop_prior::VecOrMat{Float64}=[0.0, 1.0]`: Vector or Matrix with the
  corresponding parameters (Vector: `logσ_pop_prior[1]` = mean,
  `logσ_pop_prior[2]` = standard deviation, Matrix: `logσ_pop_prior[:, 1]` =
  mean, `logσ_pop_prior[:, 2]` = standard deviation) for a Normal prior on the
  population mean fitness error utilized in the log-likelihood function. If
  `typeof(logσ_pop_prior) <: Matrix`, there should be as many rows in the matrix
  as pairs of time adjacent time points × number of replicates in dataset.
- `s_bc_prior::VecOrMat{Float64}=[0.0, 2.0]`: Vector or Matrix with the
  corresponding parameters (Vector: `s_bc_prior[1]` = mean, `s_bc_prior[2]` =
  standard deviation, Matrix: `s_bc_prior[:, 1]` = mean, `s_bc_prior[:, 2]` =
  standard deviation) for a Normal prior on the mutant fitness values. If
  `typeof(s_bc_prior) <: Matrix`, there should be as many rows in the matrix as
  number of mutant lineages × number of replicates in the dataset. 
- `logσ_bc_prior::VecOrMat{Float64}=[0.0, 1.0]`: Vector or Matrix with the
  corresponding parameters (Vector: `logσ_bc_prior[1]` = mean,
  `logσ_bc_prior[2]` = standard deviation, Matrix: `logσ_bc_prior[:, 1]` = mean,
  `logσ_bc_prior[:, 2]` = standard deviation) for a Normal prior on the mutant
  fitness error utilized in the log-likelihood function. If
  `typeof(logσ_bc_prior) <: Matrix`, there should be as many rows in the matrix
  as mutant lineages × number of replicates in the dataset.
- `logλ_prior::VecOrMat{Float64}=[3.0, 3.0]`: Vector or Matrix with the
  corresponding parameters (Vector: `logλ_prior[1]` = mean, `logλ_prior[2]` =
  standard deviation, Matrix: `logλ_prior[:, 1]` = mean, `logλ_prior[:, 2]` =
  standard deviation) for a Normal prior on the λ parameter in the Poisson
  distribution. The λ parameter can be interpreted as the mean number of barcode
  counts since we assume any barcode count `n⁽ᵇ⁾ ~ Poisson(λ⁽ᵇ⁾)`. If
  `typeof(logλ_prior) <: Matrix`, there should be as many rows in the matrix as
  number of barcodes × number of time points × number of replicates in the
  dataset.
  
## Latent Variables
- Population mean fitness per timepoint.
- Genotype hyper-fitness effects. 
- Non-centered samples for each of the non-neutral barcodes.
- Deviations from the hyper parameter value for each non-neutral barcode.
- λ dispersion parameters per barcode and timepoint.
  
## Notes
- Models hyper-fitness effects as normally distributed.
- Models fitness effects as normally distributed.
- Utilizes a Poisson observation model for barcode counts.  
- Setting informative priors is recommended for stable convergence.
"""
Turing.@model function genotype_fitness_normal(
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64},
    n_neutral::Int,
    n_bc::Int;
    genotypes::Vector{<:Any},
    s_pop_prior::VecOrMat{Float64}=[0.0, 2.0],
    logσ_pop_prior::VecOrMat{Float64}=[0.0, 1.0],
    s_bc_prior::VecOrMat{Float64}=[0.0, 2.0],
    logσ_bc_prior::VecOrMat{Float64}=[0.0, 1.0],
    logλ_prior::VecOrMat{Float64}=[3.0, 3.0],
    logτ_prior::Vector{Float64}=[-2.0, 1.0]
)
    # Check that the number of assigned genotypes matches number of barcodes
    if n_bc != length(genotypes)
        error("List of genotypes must match number of barcodes")
    end # if

    # Find unique genotypes
    geno_unique = unique(genotypes)
    # Define number of unique genotypes
    n_geno = length(geno_unique)
    # Define genotype indexes
    geno_idx = indexin(genotypes, geno_unique)

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

    # Prior on LogNormal error π(logσ̲ₜ )
    if typeof(logσ_pop_prior) <: Vector
        logσ̲ₜ ~ Turing.MvNormal(
            repeat([logσ_pop_prior[1]], n_time - 1),
            LinearAlgebra.I(n_time - 1) .* logσ_pop_prior[2] .^ 2
        )
    elseif typeof(logσ_pop_prior) <: Matrix
        logσ̲ₜ ~ Turing.MvNormal(
            logσ_pop_prior[:, 1],
            LinearAlgebra.Diagonal(logσ_pop_prior[:, 2] .^ 2)
        )
    end # if

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Hyper prior on genotype fitness π(θ̲⁽ᵐ⁾) 
    if typeof(s_bc_prior) <: Vector
        θ̲⁽ᵐ⁾ ~ Turing.MvNormal(
            repeat([s_bc_prior[1]], n_geno),
            LinearAlgebra.I(n_geno) .* s_bc_prior[2] .^ 2
        )
    elseif typeof(s_bc_prior) <: Matrix
        θ̲⁽ᵐ⁾ ~ Turing.MvNormal(
            s_bc_prior[:, 1], LinearAlgebra.Diagonal(s_bc_prior[:, 2] .^ 2)
        )
    end # if

    # Non-centered samples
    θ̲̃⁽ᵐ⁾ ~ Turing.MvNormal(zeros(n_bc), LinearAlgebra.I(n_bc))

    # Hyper prior on mutant deviations from hyper-fitness
    logτ̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([logτ_prior[1]], n_bc),
        LinearAlgebra.I(n_bc) .* logτ_prior[2] .^ 2
    )

    # mutant fitness = hyperparameter + deviation
    s̲⁽ᵐ⁾ = θ̲⁽ᵐ⁾[geno_idx] .+ (exp.(logτ̲⁽ᵐ⁾) .* θ̲̃⁽ᵐ⁾)

    # Prior on LogNormal error π(logσ̲⁽ᵐ⁾)
    if typeof(logσ_bc_prior) <: Vector
        logσ̲⁽ᵐ⁾ ~ Turing.MvNormal(
            repeat([logσ_bc_prior[1]], n_bc),
            LinearAlgebra.I(n_bc) .* logσ_bc_prior[2] .^ 2
        )
    elseif typeof(logσ_bc_prior) <: Matrix
        logσ̲⁽ᵐ⁾ ~ Turing.MvNormal(
            logσ_bc_prior[:, 1],
            LinearAlgebra.Diagonal(logσ_bc_prior[:, 2] .^ 2)
        )
    end # if

    ## %%%%%%%%%%%%%% Barcode frequencies %%%%%%%%%%%%%% ##

    if typeof(logλ_prior) <: Vector
        # Prior on Poisson distribtion parameters π(λ)
        logΛ̲̲ ~ Turing.MvNormal(
            repeat([logλ_prior[1]], length(R̲̲)),
            LinearAlgebra.I(length(R̲̲)) .* logλ_prior[2]^2
        )
    elseif typeof(logλ_prior) <: Matrix
        # Prior on Poisson distribtion parameters π(λ)
        logΛ̲̲ ~ Turing.MvNormal(
            logλ_prior[:, 1], LinearAlgebra.Diagonal(logλ_prior[:, 2] .^ 2)
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

    # Split neutral and mutant frequency ratios. 
    logΓ̲̲⁽ⁿ⁾ = vec(logΓ̲̲[:, 1:n_neutral])
    logΓ̲̲⁽ᵐ⁾ = vec(logΓ̲̲[:, n_neutral+1:n_neutral+n_bc])

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

    ## %%%%%%%%%%%%%% Log-Likelihood functions %%%%%%%%%%%%%% ##

    # Sample posterior for neutral lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ⁿ⁾| sₜ, σₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvNormal(
            repeat(-s̲ₜ, n_neutral),
            LinearAlgebra.Diagonal(repeat(exp.(logσ̲ₜ) .^ 2, n_neutral))
        ),
        logΓ̲̲⁽ⁿ⁾
    )

    # Sample posterior for nutant lineage frequency ratio. Since it is a sample
    # over a generated quantity, we must use the @addlogprob! macro
    # π(γₜ⁽ᵐ⁾ | s⁽ᵐ⁾, σ⁽ᵐ⁾, s̲ₜ)
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvNormal(
            # Build vector for fitness differences
            reduce(vcat, [s⁽ᵐ⁾ .- s̲ₜ for s⁽ᵐ⁾ in s̲⁽ᵐ⁾]),
            # Build vector for variances
            LinearAlgebra.Diagonal(
                reduce(vcat, [repeat([σ^2], n_time - 1) for σ in exp.(logσ̲⁽ᵐ⁾)])
            )
        ),
        logΓ̲̲⁽ᵐ⁾
    )
    return
end # @ model