#

# Activate environment
@load_pkg(".")

# Import package to revise package
import Revise

# Import library package
import BayesFitness

import Turing
import FillArrays
import LinearAlgebra
import Distributions

# Import libraries to manipulate data
import DataFrames as DF
import CSV
import MCMCChains

# Import library to save and load native julia objects
import JLD2

# Import library to list files
import Glob

# Import plotting libraries
using CairoMakie
import ColorSchemes
##

Turing.@model function fitness_lognormal(
    R̲̲::Matrix{Int64},
    R̲̲⁽ⁿ⁾::Matrix{Int64},
    R̲̲⁽ᵐ⁾::Matrix{Int64};
    s_pop_prior::Vector{Float64}=[0.0, 2.0],
    σ_pop_prior::Vector{Float64}=[0.0, 1.0],
    s_mut_prior::Vector{Float64}=[0.0, 2.0],
    σ_mut_prior::Vector{Float64}=[0.0, 1.0],
    λ_prior::Vector{Float64}=[3.0, 3.0]
)
    # Prior on population mean fitness
    s̲ₜ ~ Turing.MvNormal(
        repeat([s_pop_prior[1]], size(R̲̲⁽ⁿ⁾, 1) - 1),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) - 1) .* s_pop_prior[2] .^ 2
    )
    # Prior on LogNormal error σ̲ₜ
    σ̲ₜ ~ Turing.filldist(
        Turing.truncated(Turing.Normal(σ_pop_prior...); lower=0.0),
        size(R̲̲⁽ⁿ⁾, 1) - 1
    )

    # Prior on mutant fitness s̲⁽ᵐ⁾
    s̲⁽ᵐ⁾ ~ Turing.MvNormal(
        repeat([s_mut_prior[1]], size(R̲̲⁽ᵐ⁾, 2)),
        LinearAlgebra.I(size(R̲̲⁽ᵐ⁾, 2)) .* s_mut_prior[2] .^ 2
    )
    # Prior on LogNormal error σ̲⁽ᵐ⁾ 
    σ̲⁽ᵐ⁾ ~ Turing.filldist(
        Turing.truncated(Turing.Normal(σ_mut_prior...); lower=0.0),
        size(R̲̲⁽ᵐ⁾, 2)
    )

    # Prior on λ parameters
    Λ̲̲ ~ Turing.MvLogNormal(
        repeat([λ_prior[1]], size(R̲̲⁽ⁿ⁾, 1) * (size(R̲̲⁽ⁿ⁾, 2) + size(R̲̲⁽ᵐ⁾, 2))),
        LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 1) * (size(R̲̲⁽ⁿ⁾, 2) + size(R̲̲⁽ᵐ⁾, 2))) .*
        λ_prior[2]^2
    )

    Λ̲̲ = reshape(Λ̲̲, size(R̲̲⁽ⁿ⁾, 1), (size(R̲̲⁽ⁿ⁾, 2) + size(R̲̲⁽ᵐ⁾, 2)))


    # Compute frequencies πₜ
    F̲̲ = Λ̲̲ ./ sum(Λ̲̲, dims=2)

    # Check that all frequencies are greater than zero. Although the counts
    # could be zero, we assume that the real frequencies are non-zero always.
    if any(iszero.(F̲̲))
        Turing.@addlogprob! -Inf
        # Exit the model evaluation early
        return
    end

    # Compute frequency ratios γₜ
    Γ̲̲ = F̲̲[2:end, :] ./ F̲̲[1:end-1, :]
    # Split neutral and mutant frequency ratios
    Γ̲̲⁽ⁿ⁾ = @view Γ̲̲[:, 1:size(R̲̲⁽ⁿ⁾, 2)]
    Γ̲̲⁽ᵐ⁾ = @view Γ̲̲[:, size(R̲̲⁽ⁿ⁾, 2)+1:end]

    # Initialize array to save n̲ₜ
    # nₜ = zeros(Int64, size(R̲̲⁽ⁿ⁾, 2))

    # Loop through time
    for t = 1:size(R̲̲⁽ⁿ⁾, 1)
        # Prob nₜ | λ̲ₜ
        # nₜ[t] = rand(Distributions.Poisson(sum(Λ̲̲[t, :])))

        # Prob of reads given parameters R̲ₜ | nₜ, f̲ₜ.
        R̲̲[t, :] ~ Turing.Multinomial(sum(R̲̲[t, :]), F̲̲[t, :])
    end # for

    # Loop through pairs of time points
    for t = 1:size(R̲̲⁽ⁿ⁾, 1)-1
        # Likelihood for neutral lineages frequency ratio. Since Γ̲̲ is not an
        # observed variable, we use @addlogprob! to force Turing to consider it
        # as such.
        Turing.@addlogprob! Turing.logpdf(
            Turing.MvLogNormal(
                FillArrays.Fill(-s̲ₜ[t], size(R̲̲⁽ⁿ⁾, 2)),
                LinearAlgebra.I(size(R̲̲⁽ⁿ⁾, 2)) .* σ̲ₜ[t]^2
            ),
            Γ̲̲⁽ⁿ⁾[t, :]
        )
    end # for

    # Loop through mutant lineages
    for m = 1:size(R̲̲⁽ᵐ⁾, 2)
        # Sample posterior for frequency ratio. Since it is a sample over a
        # generated quantity, we must use the @addlogprob! macro
        Turing.@addlogprob! Turing.logpdf(
            Turing.MvLogNormal(
                s̲⁽ᵐ⁾[m] .- s̲ₜ,
                LinearAlgebra.I(length(s̲ₜ)) .* σ̲⁽ᵐ⁾[m]^2
            ),
            Γ̲̲⁽ᵐ⁾[:, m]
        )
    end # for

    return
end # @model function


##

println("Loading data...\n")
# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

##

time_col = :time
id_col = :barcode
count_col = :count

# Re-extract unique time points
timepoints = sort(unique(data[:, time_col]))

# Group data by unique mutant barcode
data_group = DF.groupby(data[.!data[:, :neutral], :], id_col)

# Check that all barcodes were measured at all points
if any([size(d, 1) for d in data_group] .!= length(timepoints))
    error("Not all barcodes have reported counts in all time points")
end # if

# Extract keys
data_keys = [k[String(id_col)] for k in keys(data_group)]

# Extract total number of barcodes per timepoint
R_tot = DF.combine(DF.groupby(data, time_col), count_col => sum)
# Sort dataframe by time
DF.sort!(R_tot, time_col)
# Extract sorted counts
R̲ = R_tot[:, Symbol(String(count_col) * "_sum")]

# Initialize array to save counts for each mutant at time t
R_mut = Matrix{Int64}(
    undef, length(timepoints), length(data_group)
)

# Loop through each unique barcode
for (i, d) in enumerate(data_group)
    # Sort data by timepoint
    DF.sort!(d, time_col)
    # Extract data
    R_mut[:, i] = d[:, count_col]
end # for

R_mut = R_mut

##

# Group data by unique mutant barcode
data_group = DF.groupby(data[data[:, :neutral], :], id_col)

# Check that all barcodes were measured at all points
if any([size(d, 1) for d in data_group] .!= length(timepoints))
    error("Not all barcodes have reported counts in all time points")
end # if

# Extract keys
data_keys = [k[String(id_col)] for k in keys(data_group)]

# Extract total number of barcodes per timepoint
R_tot = DF.combine(DF.groupby(data, time_col), count_col => sum)
# Sort dataframe by time
DF.sort!(R_tot, time_col)
# Extract sorted counts
R̲ = R_tot[:, Symbol(String(count_col) * "_sum")]

# Initialize array to save counts for each mutant at time t
R_neut = Matrix{Int64}(
    undef, length(timepoints), length(data_group)
)

# Loop through each unique barcode
for (i, d) in enumerate(data_group)
    # Sort data by timepoint
    DF.sort!(d, time_col)
    # Extract data
    R_neut[:, i] = d[:, count_col]
end # for

##

# Concatenate data
R_tot = hcat(R_neut, R_mut)

##

chain = Turing.sample(
    fitness_lognormal(R_tot, R_neut, R_mut),
    Turing.NUTS(0.65),
    Turing.MCMCThreads(),
    1_000,
    4
)

# Write output into memory
JLD2.jldsave("test_chain.jld2", chain=first(chain))