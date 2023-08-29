# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Hierarchical model for genotypes within single dataset π(θ̲ᴹ, s̲ᴹ, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

Turing.@model function genotype_fitness_normal(
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64},
    n_neutral::Int,
    n_mut::Int;
    genotypes::Vector{<:Any},
    s_pop_prior::VecOrMat{Float64}=[0.0, 2.0],
    logσ_pop_prior::VecOrMat{Float64}=[0.0, 1.0],
    s_mut_prior::VecOrMat{Float64}=[0.0, 2.0],
    logσ_mut_prior::VecOrMat{Float64}=[0.0, 1.0],
    logλ_prior::VecOrMat{Float64}=[3.0, 3.0],
    logτ_prior::Vector{Float64}=[-2.0, 1.0]
)
    # Check that the number of assigned genotypes matches number of barcodes
    if n_mut != length(genotypes)
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
    if typeof(s_mut_prior) <: Vector
        θ̲⁽ᵐ⁾ ~ Turing.MvNormal(
            repeat([s_mut_prior[1]], n_geno),
            LinearAlgebra.I(n_geno) .* s_mut_prior[2] .^ 2
        )
    elseif typeof(s_mut_prior) <: Matrix
        θ̲⁽ᵐ⁾ ~ Turing.MvNormal(
            s_mut_prior[:, 1], LinearAlgebra.Diagonal(s_mut_prior[:, 2] .^ 2)
        )
    end # if

    # Non-centered samples
    θ̲̃⁽ᵐ⁾ ~ Turing.MvNormal(zeros(n_mut), LinearAlgebra.I(n_mut))

    # Hyper prior on mutant deviations from hyper-fitness
    logτ̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([logτ_prior[1]], n_mut),
        LinearAlgebra.I(n_mut) .* logτ_prior[2] .^ 2
    )

    # mutant fitness = hyperparameter + deviation
    s̲⁽ᵐ⁾ = θ̲⁽ᵐ⁾[geno_idx] .+ (exp.(logτ̲⁽ᵐ⁾) .* θ̲̃⁽ᵐ⁾)

    # Prior on LogNormal error π(logσ̲⁽ᵐ⁾)
    if typeof(logσ_mut_prior) <: Vector
        logσ̲⁽ᵐ⁾ ~ Turing.MvNormal(
            repeat([logσ_mut_prior[1]], n_mut),
            LinearAlgebra.I(n_mut) .* logσ_mut_prior[2] .^ 2
        )
    elseif typeof(logσ_mut_prior) <: Matrix
        logσ̲⁽ᵐ⁾ ~ Turing.MvNormal(
            logσ_mut_prior[:, 1],
            LinearAlgebra.Diagonal(logσ_mut_prior[:, 2] .^ 2)
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
    logΓ̲̲⁽ᵐ⁾ = vec(logΓ̲̲[:, n_neutral+1:n_neutral+n_mut])

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