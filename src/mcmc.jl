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
    mcmc_mean_fitness(; kwargs)

Function to sample the posterior distribution of the population mean fitness for
a series of pairs of time points. This function expects the data in a **tidy**
format. This means that every row represents **a single observation**. For
example, if we measure barcode `i` in 4 different time points, each of these
four measurements gets an individual row. Furthermore, measurements of barcode
`j` over time also get their own individual rows.

The `DataFrame` must contain at least the following columns:
- `id_col`: Column identifying the ID of the barcode. This can the barcode
  sequence, for example.
- `time_col`: Column defining the measurement time point.
- `count_col`: Column with the raw barcode count.
- `neutral_col`: Column indicating whether the barcode is from a neutral lineage
  or not.

# Keyword Arguments
- `data::DataFrames.AbstractDataFrame`: **Tidy dataframe** with the data to be
  used to sample from the population mean fitness posterior distribution.
- `n_walkers::Int`: Number of walkers (chains) for the MCMC sample.
- `n_steps::Int`: Number of steps to take.
- `outputdir::String`: Directory where the output `.jld2` files containing the
  MCMC chains should be stored.
- `outputname::String`: Common pattern for all `.jld2` output files. The output
  files of this function will be named as
```
$(outputdir)/$(outputname)_$(t)-$(t+1)_meanfitness_mcmcchains.jld
```
where `t` and `t+1` indicate the time points used during the inference.
- `model::Function`: `Turing.jl` model defining the posterior distribution from
  which to sample (see [BayesFitness.model](model) module). This function must
  take as the first two inputs the following:
    - `r̲ₜ::Vector{Int64}`: Raw counts for **neutral** lineages and the
      cumulative counts for mutant lineages at time `t`. NOTE: The last entry of
      the array must be the sum of all of the counts from mutant lineages.
    - `r̲ₜ₊₁::Vector{Int64}`: Raw counts for **neutral** lineages and the
      cumulative counts for mutant lineages at time `t + 1`. NOTE: The last
      entry of the array must be the sum of all of the counts from mutant
      lineages. 

## Optional Arguments
- `modele_kwargs::Dict=Dict()`: Extra keyword arguments to be passed to the
  `model` function.
- `id_col::Symbol=:barcode`: Name of the column in `data` containing the barcode
    identifyer. The column may contain any type of entry.
- `time_col::Symbol=:time`: Name of the column in `data` defining the time point
  at which measurements were done. The column may contain any type of entry as
  long as `sort` will resulted in time-ordered names.
- `count_col::Symbol=:count`: Name of the column in `data` containing the raw
  barcode count. The column must contain entries of type `Int64`.
- `neutral_col::Symbol=:neutral`: Name of the column in `data` defining whether
  the barcode belongs to a neutral lineage or not. The column must contain
  entries of type `Bool`.
- `rm_T0::Bool=false`: Optional argument to remove the first time point from the
  inference. Commonly, the data from this first time point is of much lower
  quality. Therefore, removing this first time point might result in a better
  inference.
- `suppress_output::Bool=false`: Boolean indicating if the screen output of
  `Turing.jl` must be actively suppressed.
- `sampler::Turing.Inference.InferenceAlgorithm=Turing.NUTS(0.65)`: MCMC sampler
  to be used.
- `verbose::Bool=true`: Boolean indicating if the function should print partial
  progress to the screen or not.
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

    # Re-extract unique time points
    timepoints = sort(unique(data[:, time_col]))

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Running MCMC for mutant fitness π(s⁽ᵐ⁾ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    mcmc_mutant_fitness(; kwargs)

Function to sample the posterior distribution of mutant lineages relative
fitness given a time-series barcode count. 

This function expects the data in a **tidy** format. This means that every row
represents **a single observation**. For example, if we measure barcode `i` in 4
different time points, each of these four measurements gets an individual row.
Furthermore, measurements of barcode `j` over time also get their own individual
rows.

The `DataFrame` must contain at least the following columns:
- `id_col`: Column identifying the ID of the barcode. This can the barcode
  sequence, for example.
- `time_col`: Column defining the measurement time point.
- `count_col`: Column with the raw barcode count.
- `neutral_col`: Column indicating whether the barcode is from a neutral lineage
  or not.

# Keyword Arguments
- `data::DataFrames.AbstractDataFrame`: **Tidy dataframe** with the data to be
  used to sample from the population mean fitness posterior distribution.
- `n_walkers::Int`: Number of walkers (chains) for the MCMC sample.
- `n_steps::Int`: Number of steps to take.
- `outputdir::String`: Directory where the output `.jld2` files containing the
  MCMC chains should be stored.
- `outputname::String`: Common pattern for all `.jld2` output files. The output
  files of this function will be named as
```
$(outputdir)/$(outputname)_$(mutant_id)_mcmcchains.jld
```
where `t` and `t+1` indicate the time points used during the inference.
- `model::Function`: `Turing.jl` model defining the posterior distribution from
  which to sample (see [BayesFitness.model](model) module). This function must
  take as the first two inputs the following:
    - `r̲ₜ::Vector{Int64}`: Raw counts for **neutral** lineages and the
      cumulative counts for mutant lineages at time `t`. NOTE: The last entry of
      the array must be the sum of all of the counts from mutant lineages.
    - `r̲ₜ₊₁::Vector{Int64}`: Raw counts for **neutral** lineages and the
      cumulative counts for mutant lineages at time `t + 1`. NOTE: The last
      entry of the array must be the sum of all of the counts from mutant
      lineages. 

## Optional Arguments
- `modele_kwargs::Dict=Dict()`: Extra keyword arguments to be passed to the
  `model` function.
- `id_col::Symbol=:barcode`: Name of the column in `data` containing the barcode
    identifyer. The column may contain any type of entry.
- `time_col::Symbol=:time`: Name of the column in `data` defining the time point
  at which measurements were done. The column may contain any type of entry as
  long as `sort` will resulted in time-ordered names.
- `count_col::Symbol=:count`: Name of the column in `data` containing the raw
  barcode count. The column must contain entries of type `Int64`.
- `neutral_col::Symbol=:neutral`: Name of the column in `data` defining whether
  the barcode belongs to a neutral lineage or not. The column must contain
  entries of type `Bool`.
- `rm_T0::Bool=false`: Optional argument to remove the first time point from the
  inference. Commonly, the data from this first time point is of much lower
  quality. Therefore, removing this first time point might result in a better
  inference.
- `suppress_output::Bool=false`: Boolean indicating if the screen output of
  `Turing.jl` must be actively suppressed.
- `sampler::Turing.Inference.InferenceAlgorithm=Turing.NUTS(0.65)`: MCMC sampler
  to be used.
- `verbose::Bool=true`: Boolean indicating if the function should print partial
  progress to the screen or not.
"""
function mcmc_mutant_fitness(;
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
    # Extract unique time points
    timepoints = sort(unique(data[:, time_col]))

    # Remove T0 if indicated
    if rm_T0
        if verbose
            println("Deleting T0 as requested...")
        end # if 
        data = data[.!(data[:, time_col] .== first(timepoints)), :]
    end # if

    # Re-extract unique time points
    timepoints = sort(unique(data[:, time_col]))

    if verbose
        println("Grouping data by mutant barcode")
    end # if
    # Group data by unique mutant barcode
    data_group = DF.groupby(data[.!data[:, neutral_col], :], id_col)
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
    r⁽ᵐ⁾_array = Matrix{Int64}(
        undef, length(data_group), length(timepoints)
    )

    # Loop through each unique barcode
    for (i, d) in enumerate(data_group)
        # Sort data by timepoint
        DF.sort!(d, time_col)
        # Extract data
        r⁽ᵐ⁾_array[i, :] = d[:, count_col]
    end # for

    if verbose
        println("Initialize MCMC sampling...")
    end # if
    # Loop through barcodes
    for j = 1:size(r⁽ᵐ⁾_array, 1)
        # Define output_name
        fname = "$(outputname)_$(data_keys[j])_mcmcchains.jld2"
        # Check that file hasn't been processed
        if isfile("$(outputdir)/$(fname)")
            if verbose
                # Print if barcode was already processed
                println("$(data_keys[j]) was already processed")
            end # if
            # Skip cycle for already-processed barcodes
            continue
        end # if

        # Initialize object where to save chains
        chain = Vector{MCMCChains.Chains}(undef, 1)

        # Define model
        mcmc_model = model(r⁽ᵐ⁾_array[j, :], R̲; model_kwargs...)

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
        JLD2.jldsave("$(outputdir)/$(fname)", chain=first(chain))
    end # for

    if verbose
        println("Done!")
    end # if
end # function


@doc raw"""
    mcmc_mutant_fitness_multithread(; kwargs)

Function to sample the posterior distribution of mutant lineages relative
fitness given a time-series barcode count. This function runs the inference of
multiple mutants in a multithread fasion. Because of this, every mutant gets
only one MCMC chain.

This function expects the data in a **tidy** format. This means that every row
represents **a single observation**. For example, if we measure barcode `i` in 4
different time points, each of these four measurements gets an individual row.
Furthermore, measurements of barcode `j` over time also get their own individual
rows.

The `DataFrame` must contain at least the following columns:
- `id_col`: Column identifying the ID of the barcode. This can the barcode
  sequence, for example.
- `time_col`: Column defining the measurement time point.
- `count_col`: Column with the raw barcode count.
- `neutral_col`: Column indicating whether the barcode is from a neutral lineage
  or not.

# Keyword Arguments
- `data::DataFrames.AbstractDataFrame`: **Tidy dataframe** with the data to be
  used to sample from the population mean fitness posterior distribution.
- `n_walkers::Int`: Number of walkers (chains) for the MCMC sample.
- `n_steps::Int`: Number of steps to take.
- `outputdir::String`: Directory where the output `.jld2` files containing the
  MCMC chains should be stored.
- `outputname::String`: Common pattern for all `.jld2` output files. The output
  files of this function will be named as
```
$(outputdir)/$(outputname)_$(mutant_id)_mcmcchains.jld
```
where `t` and `t+1` indicate the time points used during the inference.
- `model::Function`: `Turing.jl` model defining the posterior distribution from
  which to sample (see [BayesFitness.model](model) module). This function must
  take as the first two inputs the following:
    - `r̲ₜ::Vector{Int64}`: Raw counts for **neutral** lineages and the
      cumulative counts for mutant lineages at time `t`. NOTE: The last entry of
      the array must be the sum of all of the counts from mutant lineages.
    - `r̲ₜ₊₁::Vector{Int64}`: Raw counts for **neutral** lineages and the
      cumulative counts for mutant lineages at time `t + 1`. NOTE: The last
      entry of the array must be the sum of all of the counts from mutant
      lineages. 

## Optional Arguments
- `modele_kwargs::Dict=Dict()`: Extra keyword arguments to be passed to the
  `model` function.
- `id_col::Symbol=:barcode`: Name of the column in `data` containing the barcode
    identifyer. The column may contain any type of entry.
- `time_col::Symbol=:time`: Name of the column in `data` defining the time point
  at which measurements were done. The column may contain any type of entry as
  long as `sort` will resulted in time-ordered names.
- `count_col::Symbol=:count`: Name of the column in `data` containing the raw
  barcode count. The column must contain entries of type `Int64`.
- `neutral_col::Symbol=:neutral`: Name of the column in `data` defining whether
  the barcode belongs to a neutral lineage or not. The column must contain
  entries of type `Bool`.
- `rm_T0::Bool=false`: Optional argument to remove the first time point from the
  inference. Commonly, the data from this first time point is of much lower
  quality. Therefore, removing this first time point might result in a better
  inference.
- `suppress_output::Bool=false`: Boolean indicating if the screen output of
  `Turing.jl` must be actively suppressed.
- `sampler::Turing.Inference.InferenceAlgorithm=Turing.NUTS(0.65)`: MCMC sampler
  to be used.
- `verbose::Bool=true`: Boolean indicating if the function should print partial
  progress to the screen or not.
"""
function mcmc_mutant_fitness_multithread(;
    data::DF.AbstractDataFrame,
    n_walkers::Int=1,
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
    # Extract unique time points
    timepoints = sort(unique(data[:, time_col]))

    # Remove T0 if indicated
    if rm_T0
        if verbose
            println("Deleting T0 as requested...")
        end # if 
        data = data[.!(data[:, time_col] .== first(timepoints)), :]
    end # if

    # Re-extract unique time points
    timepoints = sort(unique(data[:, time_col]))

    if verbose
        println("Grouping data by mutant barcode")
    end # if
    # Group data by unique mutant barcode
    data_group = DF.groupby(data[.!data[:, neutral_col], :], id_col)
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
    r⁽ᵐ⁾_array = Matrix{Int64}(
        undef, length(data_group), length(timepoints)
    )

    # Loop through each unique barcode
    for (i, d) in enumerate(data_group)
        # Sort data by timepoint
        DF.sort!(d, time_col)
        # Extract data
        r⁽ᵐ⁾_array[i, :] = d[:, count_col]
    end # for

    if verbose
        println("Initialize MCMC sampling...")
    end # if
    # Loop through barcodes
    Threads.@threads for j = 1:size(r⁽ᵐ⁾_array, 1)
        # Define output_name
        fname = "$(outputname)_$(data_keys[j])_mcmcchains.jld2"
        # Check that file hasn't been processed
        if isfile("$(outputdir)/$(fname)")
            if verbose
                # Print if barcode was already processed
                println("$(data_keys[j]) was already processed")
            end # if
            # Skip cycle for already-processed barcodes
            continue
        end # if

        # Initialize object where to save chains
        chain = Vector{MCMCChains.Chains}(undef, 1)

        # Define model
        mcmc_model = model(r⁽ᵐ⁾_array[j, :], R̲; model_kwargs...)

        if suppress_output
            # Suppress warning outputs
            Suppressor.@suppress begin
                # Sample
                chain[1] = mapreduce(
                    c -> Turing.sample(
                        mcmc_model, sampler, n_steps, progress=false
                    ),
                    Turing.chainscat,
                    1:n_walkers
                )
                # # Sample
                # chain[1] = Turing.sample(
                #     mcmc_model,
                #     sampler,
                #     n_steps,
                #     progress=false
                # )
            end # suppress
        else
            # Sample
            chain[1] = mapreduce(
                c -> Turing.sample(
                    mcmc_model, sampler, n_steps, progress=true
                ),
                Turing.chainscat,
                1:n_walkers
            )
            # chain[1] = Turing.sample(
            #     mcmc_model,
            #     sampler,
            #     n_steps,
            #     progress=true
            # )
        end # if

        if verbose
            println("Saving $(fname) chains...")
        end # if
        # Write output into memory
        JLD2.jldsave("$(outputdir)/$(fname)", chain=first(chain))
    end # for

    if verbose
        println("Done!")
    end # if
end # function