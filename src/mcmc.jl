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
export mcmc_mean_fitness, mcmc_mutant_fitness, mcmc_joint_fitness

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
$(outputdir)/$(outputname)_$(t)-$(t+1).jld
```
where `t` and `t+1` indicate the time points used during the inference.
- `model::Function`: `Turing.jl` model defining the posterior distribution from
  which to sample (see `BayesFitness.model` module). This function must take as
  the first two inputs the following:
    - `r̲ₜ::Vector{Int64}`: Raw counts for **neutral** lineages and the
      cumulative counts for mutant lineages at time `t`. NOTE: The last entry of
      the array must be the sum of all of the counts from mutant lineages.
    - `r̲ₜ₊₁::Vector{Int64}`: Raw counts for **neutral** lineages and the
      cumulative counts for mutant lineages at time `t + 1`. NOTE: The last
      entry of the array must be the sum of all of the counts from mutant
      lineages. 

## Optional Arguments
- `model_kwargs::Dict=Dict()`: Extra keyword arguments to be passed to the
  `model` function.
- `id_col::Symbol=:barcode`: Name of the column in `data` containing the barcode
    identifier. The column may contain any type of entry.
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
- `multithread::Bool=true`: Boolean indicating if the chains should be run in
  parallel.
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
    multithread::Bool=true,
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
        fname = "$(outputname)_$(timepoints[t])-$(timepoints[t+1])"

        # Check that file hasn't been processed
        if isfile("$(outputdir)/$(fname).jld2")
            if verbose
                # Print if barcode was already processed
                println("$(fname) was already processed")
            end # if
            # Skip cycle for already-processed barcodes
            continue
        end # if

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
                if multithread
                    # Sample
                    chain[1] = Turing.sample(
                        mcmc_model,
                        sampler,
                        Turing.MCMCThreads(),
                        n_steps,
                        n_walkers,
                        progress=false
                    )
                else
                    chain[1] = mapreduce(
                        c -> Turing.sample(
                            mcmc_model, sampler, n_steps, progress=false
                        ),
                        Turing.chainscat,
                        1:n_walkers
                    )
                end # if
            end # suppress
        else
            if multithread
                chain[1] = Turing.sample(
                    mcmc_model,
                    sampler,
                    Turing.MCMCThreads(),
                    n_steps,
                    n_walkers,
                    progress=true
                )
            else
                chain[1] = mapreduce(
                    c -> Turing.sample(
                        mcmc_model, sampler, n_steps, progress=true
                    ),
                    Turing.chainscat,
                    1:n_walkers
                )
            end # if
        end # if

        if verbose
            println("Saving $(fname) chains...")
        end # if

        # Write output into memory
        JLD2.jldsave(
            "$(outputdir)/$(fname).jld2",
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
$(outputdir)/$(outputname)_$(mutant_id).jld
```
where `t` and `t+1` indicate the time points used during the inference.
- `model::Function`: `Turing.jl` model defining the posterior distribution from
  which to sample (see `BayesFitness.model` module). This function must take as
  the first two inputs the following:
    - `r̲ₜ::Vector{Int64}`: Raw counts for **neutral** lineages and the
      cumulative counts for mutant lineages at time `t`. NOTE: The last entry of
      the array must be the sum of all of the counts from mutant lineages.
    - `r̲ₜ₊₁::Vector{Int64}`: Raw counts for **neutral** lineages and the
      cumulative counts for mutant lineages at time `t + 1`. NOTE: The last
      entry of the array must be the sum of all of the counts from mutant
      lineages. 

## Optional Arguments
- `model_kwargs::Dict=Dict()`: Extra keyword arguments to be passed to the
  `model` function.
- `id_col::Symbol=:barcode`: Name of the column in `data` containing the barcode
    identifier. The column may contain any type of entry.
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
- `multithread_chain::Bool=false`: Boolean indicating if the chains should be
  run in parallel.
- `multithread_mutant::Bool=false`: Boolean indicating if the chains should be
  run in parallel. NOTE: Only one `multithread_` option can be true at any
  point.
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
    multithread_chain::Bool=false,
    multithread_mutant::Bool=false,
    verbose::Bool=true
)
    # Check multithread options
    if multithread_chain & multithread_mutant
        error("Only one multithread option can be true.")
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

    if verbose
        println("Grouping data by mutant barcode...")
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


    # Run when multithread_mutant is FALSE
    if !multithread_mutant
        # Loop through barcodes
        for j = 1:size(r⁽ᵐ⁾_array, 1)
            # Define output_name
            fname = "$(outputname)_$(data_keys[j])"
            # Check that file hasn't been processed
            if isfile("$(outputdir)/$(fname).jld2")
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
                    if multithread_chain
                        # Sample
                        chain[1] = Turing.sample(
                            mcmc_model,
                            sampler,
                            Turing.MCMCThreads(),
                            n_steps,
                            n_walkers,
                            progress=false
                        )
                    else
                        chain[1] = mapreduce(
                            c -> Turing.sample(
                                mcmc_model, sampler, n_steps, progress=false
                            ),
                            Turing.chainscat,
                            1:n_walkers
                        )
                    end # if
                end # suppress
            else
                if multithread_chain
                    chain[1] = Turing.sample(
                        mcmc_model,
                        sampler,
                        Turing.MCMCThreads(),
                        n_steps,
                        n_walkers,
                        progress=true
                    )
                else
                    chain[1] = mapreduce(
                        c -> Turing.sample(
                            mcmc_model, sampler, n_steps, progress=true
                        ),
                        Turing.chainscat,
                        1:n_walkers
                    )
                end # if
            end # if

            if verbose
                println("Saving $(fname) chains...")
            end # if
            # Write output into memory
            JLD2.jldsave("$(outputdir)/$(fname).jld2", chain=first(chain))
        end # for

    # Run when multithread_mutant is TRUE
    elseif multithread_mutant
        # Loop through barcodes
        Threads.@threads for j = 1:size(r⁽ᵐ⁾_array, 1)
            # Define output_name
            fname = "$(outputname)_$(data_keys[j])"
            # Check that file hasn't been processed
            if isfile("$(outputdir)/$(fname).jld2")
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
                    chain[1] = mapreduce(
                        c -> Turing.sample(
                            mcmc_model, sampler, n_steps, progress=false
                        ),
                        Turing.chainscat,
                        1:n_walkers
                    )
                end # suppress
            else
                chain[1] = mapreduce(
                    c -> Turing.sample(
                        mcmc_model, sampler, n_steps, progress=true
                    ),
                    Turing.chainscat,
                    1:n_walkers
                )
            end # if

            if verbose
                println("Saving $(fname) chains...")
            end # if
            # Write output into memory
            JLD2.jldsave("$(outputdir)/$(fname).jld2", chain=first(chain))
        end # for

    end # if

    if verbose
        println("Done!")
    end # if
end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Running MCMC for full joint fitness inference π(s̲⁽ᵐ⁾, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    mcmc_joint_fitness(; kwargs)

Function to sample the joint posterior distribution for the fitness value of all
mutant and neutral linages given a time-series barcode count.

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
- `outputname::String`: String to be used to name the `.jld2` output file.
- `model::Function`: `Turing.jl` model defining the posterior distribution from
  which to sample (see `BayesFitness.model` module). This function must take as
  the first four inputs the following:
    - `R̲̲⁽ⁿ⁾::Matrix{Int64}`: `T × N` matrix where `T` is the number of time
      points in the data set and `N` is the number of neutral lineage barcodes.
      Each column represents the barcode count trajectory for a single neutral
      lineage.  **NOTE**: The model assumes the rows are sorted in order of
      increasing time.
    - `R̲̲⁽ᵐ⁾::Matrix{Int64}`: `T × M` matrix where `T` is the number of time
      points in the data set and `M` is the number of mutant lineage barcodes.
      Each column represents the barcode count trajectory for a single mutant
      lineage. **NOTE**: The model assumes the rows are sorted in order of
      increasing time.
    - `R̲̲::Matrix{Int64}`:: `T × B` matrix, where `T` is the number of time
      points in the data set and `B` is the number of barcodes. Each column
      represents the barcode count trajectory for a single lineage. **NOTE**:
      This matrix **must** be equivalent to `hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾)`. The reason
      it is an independent input parameter is to avoid the `hcat` computation
      within the `Turing` model.
    - `n̲ₜ::Vector{Int64}`: Vector with the total number of barcode counts for
      each time point. **NOTE**: This vector **must** be equivalent to computing
      `vec(sum(R̲̲, dims=2))`. The reason it is an independent input parameter
      is to avoid the `sum` computation within the `Turing` model.

## Optional Keyword Arguments
- `model_kwargs::Dict=Dict()`: Extra keyword arguments to be passed to the
  `model` function.
- `id_col::Symbol=:barcode`: Name of the column in `data` containing the barcode
    identifier. The column may contain any type of entry.
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
- `sampler::Turing.Inference.InferenceAlgorithm=Turing.NUTS(0.65)`: MCMC sampler
  to be used.
- `multithread::Bool=true`: Boolean indicating if the chains should be run in
    parallel.
- `verbose::Bool=true`: Boolean indicating if the function should print partial
  progress to the screen or not.
"""
function mcmc_joint_fitness(;
    data::DF.AbstractDataFrame,
    n_walkers::Int,
    n_steps::Int,
    outputname::String,
    model::Function,
    model_kwargs::Dict=Dict(),
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rm_T0::Bool=false,
    sampler::Turing.Inference.InferenceAlgorithm=Turing.NUTS(0.65),
    multithread::Bool=true,
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
        println("Preparing input data...")
    end # if

    ## %%%%%%%%%%% Neutral barcodes data %%%%%%%%%%% ##

    # Group data by unique mutant barcode
    data_group = DF.groupby(data[data[:, neutral_col], :], id_col)

    # Check that all barcodes were measured at all points
    if any([size(d, 1) for d in data_group] .!= length(timepoints))
        error("Not all neutral barcodes have reported counts in all time points")
    end # if

    # Extract keys
    data_keys = [k[String(id_col)] for k in keys(data_group)]

    # Initialize array to save counts for each mutant at time t
    R̲̲⁽ⁿ⁾ = Matrix{Int64}(
        undef, length(timepoints), length(data_group)
    )

    # Loop through each unique barcode
    for (i, d) in enumerate(data_group)
        # Sort data by timepoint
        DF.sort!(d, time_col)
        # Extract data
        R̲̲⁽ⁿ⁾[:, i] = d[:, count_col]
    end # for

    ## %%%%%%%%%%% Mutant barcodes data %%%%%%%%%%% ##

    # Group data by unique mutant barcode
    data_group = DF.groupby(data[.!data[:, neutral_col], :], id_col)

    # Check that all barcodes were measured at all points
    if any([size(d, 1) for d in data_group] .!= length(timepoints))
        error("Not all mutant barcodes have reported counts in all time points")
    end # if

    # Extract keys
    data_keys = [k[String(id_col)] for k in keys(data_group)]

    # Initialize array to save counts for each mutant at time t
    R̲̲⁽ᵐ⁾ = Matrix{Int64}(
        undef, length(timepoints), length(data_group)
    )

    # Loop through each unique barcode
    for (i, d) in enumerate(data_group)
        # Sort data by timepoint
        DF.sort!(d, time_col)
        # Extract data
        R̲̲⁽ᵐ⁾[:, i] = d[:, count_col]
    end # for

    ## %%%%%%%%%%% Total barcodes data %%%%%%%%%%% ##

    # Concatenate neutral and mutant data matrices
    R̲̲ = hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾)

    # Compute total counts for each run
    n̲ₜ = vec(sum(R̲̲, dims=2))

    ## %%%%%%%%%%% MCMC sampling %%%%%%%%%%% ##
    # Define output filename
    fname = "$(outputname).jld2"

    # Check if file has been processed before
    if isfile(fname)
        error("$(fname) was already processed")
    end # if

    if verbose
        println("Initialize MCMC sampling with $(Turing.ADBACKEND)...\n")
    end # if

    # Check if sampling should be done in multithread
    if multithread
        if verbose
            println("Sampling posterior in multithread...")
        end # if
        # Sample posterior using Turing.MCMCThreads
        chain = Turing.sample(
            model(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲, n̲ₜ; model_kwargs...),
            sampler,
            Turing.MCMCThreads(),
            n_steps,
            n_walkers,
            progress=true
        )
    else
        if verbose
            println("Sampling posterior in single core...")
        end # if
        # Sample posterior one chain at the time
        chain = mapreduce(
            c -> Turing.sample(
                model(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲, n̲ₜ; model_kwargs...),
                sampler,
                n_steps,
                progress=true
            ),
            Turing.chainscat,
            1:n_walkers
        )
    end # if

    if verbose
        println("Saving $(fname) chains...")
    end # if
    # Write output into memory
    JLD2.jldsave("$(fname)", chain=chain[1])
end # function