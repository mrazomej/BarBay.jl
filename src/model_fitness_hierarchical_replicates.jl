# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Hierarchical model for multiple experimental replicates π(θ̲ᴹ, s̲ᴹ, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

Turing.@model function fitness_hierarchical_replicates(
    R̲̲⁽ⁿ⁾::Array{Int64,3},
    R̲̲⁽ᵐ⁾::Array{Int64,3},
    R̲̲::Array{Int64,3},
    n̲ₜ::Matrix{Int64};
    s_pop_prior::Vector{Float64}=[0.0, 2.0],
    σ_pop_prior::Vector{Float64}=[0.0, 1.0],
    s_mut_prior::Vector{Float64}=[0.0, 2.0],
    σ_mut_prior::Vector{Float64}=[0.0, 1.0],
    λ_prior::Union{Vector{Float64},Array{Int64,3}}=[3.0, 3.0],
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
    s̲ₜ ~ Turing.MvNormal(
        repeat([s_pop_prior[1]], (n_time - 1) * n_rep),
        LinearAlgebra.I((n_time - 1) * n_rep) .* s_pop_prior[2] .^ 2
    )

    # Prior on LogNormal error π(σ̲ₜ)
    σ̲ₜ ~ Turing.MvLogNormal(
        repeat([σ_pop_prior[1]], (n_time - 1) * n_rep),
        LinearAlgebra.I((n_time - 1) * n_rep) .* σ_pop_prior[2] .^ 2
    )

    ## %%%%%%%%%%%%%% Mutant fitness  %%%%%%%%%%%%%% ##

    # Hyper prior on mutant fitness π(θ̲⁽ᵐ⁾) 
    θ̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([s_mut_prior[1]], n_mut),
        LinearAlgebra.I(n_mut) .* s_mut_prior[2] .^ 2
    )

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
    σ̲⁽ᵐ⁾ ~ Turing.MvLogNormal(
        repeat([σ_mut_prior[1]], n_mut * n_rep),
        LinearAlgebra.I(n_mut * n_rep) .* σ_mut_prior[2] .^ 2
    )

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
    Γ̲̲⁽ⁿ⁾ = @view Γ̲̲[:, 1:n_neutral, :]
    Γ̲̲⁽ᵐ⁾ = @view Γ̲̲[:, n_neutral+1:n_neutral+n_mut, :]

    # Loop through replicates
    for r = 1:n_rep
        # Prob of total number of barcodes read given the Poisosn distribution
        # parameters π(nₜ | λ̲ₜ)
        n̲ₜ[:, r] ~ Turing.arraydist(
            [Turing.Poisson(sum(Λ̲̲[t, :, r])) for t = 1:n_time]
        )

        # Loop through time points
        for t = 1:n_time
            # Prob of reads given parameters π(R̲ₜ | nₜ, f̲ₜ). Note: We add the
            # check_args=false option to avoid the recurrent problem of
            # > Multinomial: p is not a probability vector.
            # due to rounding errors
            R̲̲[t, :, r] ~ Turing.Multinomial(
                n̲ₜ[t, r], F̲̲[t, :, r]; check_args=false
            )
        end # for
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
        Γ̲̲⁽ⁿ⁾[:]
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
        Γ̲̲⁽ᵐ⁾[:]
    )

    return s̲⁽ᵐ⁾
end # @model function