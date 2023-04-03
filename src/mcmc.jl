# Import library to suppress output
import Suppressor

# Import libraries relevant for MCMC
import Turing
import MCMCChains

# Import library to store output
import JLD2

# Import package to handle DataFrames
import DataFrames as DF
import CSV

##
# Export function
export mcmc_mean_fitness

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Running MCMC for population Mean Fitness π(sₜ | Data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""

# Arguments

# Keyword Arguments

## Optional Arguments
- 
"""
function mcmc_mean_fitness(;
    data::DF.AbstractDataFrame,
    n_walkers::Int,
    n_steps::Int,
    outputdir::String,
    outputname::String,
    model::Function,
    model_kwargs::Dict=Dict(),
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rm_T0::Bool=false,
    suppress_output::Bool=false,
    sampler::Turing.Inference.InferenceAlgorithm=Turing.NUTS(0.65),
    verbose::Bool=true
)
    # Check number of walkers 
    if n_walkers > Threads.nthreads()
        error("n_walkers cannot be greater than the number of available threads")
    end # if

    # Extract unique time points
    timepoints = sort(unique(data[:, time_col]))

    # Remove T0 if indicated
    if rm_T0
        if verbose
            println("Deleting T0 as requested...")
        end # if 
        data = data[.!(data[:, time_col] .== first(timepoints)), :]
    end # if

    # Loop through pairs of timepoints
    for t = 1:(length(timepoints)-1)
        if verbose
            println("Preparing time $(timepoints[t]) and $(timepoints[t+1])")
        end # if

        # Define output file name
        fname = "$(outputname)_$(timepoints[t])-$(timepoints[t+1])_meanfitness"

        # Select correspoinding data for the pair of timepoints
        data_pair = data[
            (data[:, time_col].==timepoints[t]).|(data[:, time_col].==timepoints[t+1]),
            :]

        # Group data by neutral ID
        data_group = DF.groupby(data_pair[data_pair[:, neutral_col], :], id_col)

        # Check that time points contain the same barcodes
        if any([size(d, 1) for d in data_group] .!= 2)
            error(
                "There are unpaired barcodes between time $(timepoints[t]) " *
                "and $(timepoints[t+1])"
            )
        end # if

        # Initialize array to save counts
        rₜ = Vector{Int64}(undef, length(data_group) + 1)
        rₜ₊₁ = similar(rₜ)

        # Loop through barcodes
        for (i, group) in enumerate(data_group)
            # Sort data by timepoint
            DF.sort!(group, time_col)
            rₜ[i] = first(group[:, count_col])
            rₜ₊₁[i] = last(group[:, count_col])
        end # for

        # Add cumulative mutant counts
        rₜ[end] = sum(
            data_pair[
                (.!data_pair[:, neutral_col]).&(data_pair[:, time_col].==timepoints[t]),
                count_col]
        )
        rₜ₊₁[end] = sum(
            data_pair[
                (.!data_pair[:, neutral_col]).&(data_pair[:, time_col].==timepoints[t+1]),
                count_col]
        )

        # Define model
        mcmc_model = model(rₜ, rₜ₊₁; model_kwargs...)

        # Initialize object where to save chains
        chain = Vector{MCMCChains.Chains}(undef, 1)

        println("Sampling $(fname)...")
        if suppress_output
            # Suppress warning outputs
            Suppressor.@suppress begin
                # Sample
                chain[1] = Turing.sample(
                    mcmc_model,
                    sampler,
                    Turing.MCMCThreads(),
                    n_steps,
                    n_walkers,
                    progress=false
                )
            end # suppress
        else
            chain[1] = Turing.sample(
                mcmc_model,
                sampler,
                Turing.MCMCThreads(),
                n_steps,
                n_walkers,
                progress=true
            )
        end # if

        if verbose
            println("Saving $(fname) chains...")
        end # if

        # Write output into memory
        JLD2.jldsave(
            "$(outputdir)/$(fname)_mcmcchains.jld2",
            chain=first(chain),
        )

        if verbose
            println("Done with $(fname)")
        end # if
    end # for
end # function