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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Running MCMC for population Mean Fitness π(sₜ | Data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""

# Arguments

# Keyword Arguments

## Optional Arguments
- 
"""
function mcmc_mean_fitness(
    data::DF.DataFrame,
    n_walkers::Int,
    n_steps::Int,
    outputdir::String,
    outputname::String;
    model::Function,
    model_args::Vector{Any},
    model_kwargs::Dict=Dict(),
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
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
    timepoints = sort(unique(data[time_col]))

    # Remove T0 if indicated
    if rm_T0
        if verbose
            println("Deleting T0 as requested...")
        end # if 
        data = data[.!(data[timecol] .== first(timepoints)), :]
    end # if

    # Loop through pairs of timepoints
    for t = 1:(length(timepoints)-1)
        # Define output file name
        fname = "$(outputname)_$(timepoints[t])-$(timepoints[t+1])_meanfitness"

        # Select correspoinding data for the pair of timepoints
        data_pair = data[
            (data.timepoint.==timepoints[t]).|(data.timepoint.==timepoints[t+1]),
            :]

        # Group data by neutral ID
        data_group = DF.groupby(data_pair[data_pair[neutral_col], :], id_col)

        # Initialize array to save counts
        rₜ = Vector{Int64}(undef, length(data_group) + 1)
        rₜ₊₁ = similar(rₜ)

        # Loop through barcodes
        for (i, group) in enumerate(data_group)
            # Sort data by timepoint
            DF.sort!(group, :timepoint)
            rₜ[i] = group.count
        end # for

        # Add cumulative mutant counts
        rₜ[end] = sum(
            data_pair[
                .!(data_pair[neutral_col]).&(data[time_col].==timepoints[t]),
                :count]
        )
        rₜ₊₁[end] = sum(
            data_pair[
                .!(data_pair[neutral_col]).&(data[time_col].==timepoints[t+1]),
                :count]
        )

        # Define model
        mcmc_model = model(rₜ, rₜ₊₁, model_args...; model_kwargs)

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
            "$(out_dir)/$(fname)_mcmcchains.jld2",
            chain=first(chain),
        )

        if verbose
            println("Done with $(fname)")
        end # if
    end # for
end # function