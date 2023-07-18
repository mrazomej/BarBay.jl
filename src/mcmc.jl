# Import library to suppress output
import Suppressor

# Import libraries relevant for MCMC
import Turing
import MCMCChains

# Import library to store output
import JLD2

# Import library to locate files
import Glob

# Import package to handle DataFrames
import DataFrames as DF
import CSV

##
# Export function
export mcmc_mean_fitness, mcmc_mutant_fitness, mcmc_joint_fitness

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Running MCMC to sample the population mean fitness posterior distribution
# using neutral lineages only
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    mcmc_popmean_fitness(; kwargs)

Function to sample the joint posterior distribution for the population mean
fitness value using only neutral linages given a time-series barcode count.

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
  the first three inputs the following:
    - `R̲̲⁽ⁿ⁾::Matrix{Int64}`: `T × N` matrix where `T` is the number of time
      points in the data set and `N` is the number of neutral lineage barcodes.
      Each column represents the barcode count trajectory for a single neutral
      lineage.  **NOTE**: The model assumes the rows are sorted in order of
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
- `ensemble::Turing.AbstractMCMC.AbstractMCMCEnsemble=Turing.MCMCSerial()`:
Sampling modality to be used. Options are:
    - `Turing.MCMCSerial()`
    - `Turing.MCMCThreads()`
    - `Turing.MCMCDistributed()`
- `verbose::Bool=true`: Boolean indicating if the function should print partial
  progress to the screen or not.
"""
function mcmc_popmean_fitness(;
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
    ensemble::Turing.AbstractMCMC.AbstractMCMCEnsemble=Turing.MCMCSerial(),
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

    # Extract group keys
    data_keys = first.(values.(keys(data_group)))

    # Check that all barcodes were measured at all points
    if any([size(d, 1) for d in data_group] .!= length(timepoints))
        error("Not all mutant barcodes have reported counts in all time points")
    end # if

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
    R̲̲ = hcat(R̲̲⁽ⁿ⁾, sum(R̲̲⁽ᵐ⁾, dims=2))

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

    if verbose
        println("Sampling posterior...")
    end # if

    # Sample posterior
    chain = Turing.sample(
        model(R̲̲⁽ⁿ⁾, Vector.(eachrow(R̲̲)), n̲ₜ; model_kwargs...),
        sampler,
        ensemble,
        n_steps,
        n_walkers,
        progress=verbose
    )

    if verbose
        println("Saving $(fname) chains...")
    end # if
    # Write output into memory
    JLD2.jldsave("$(fname)", chain=chain)
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
- `ensemble::Turing.AbstractMCMC.AbstractMCMCEnsemble=Turing.MCMCSerial()`:
Sampling modality to be used. Options are:
    - `Turing.MCMCSerial()`
    - `Turing.MCMCThreads()`
    - `Turing.MCMCDistributed()`
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
    ensemble::Turing.AbstractMCMC.AbstractMCMCEnsemble=Turing.MCMCSerial(),
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

    # Extract group keys
    data_keys = first.(values.(keys(data_group)))

    # Check that all barcodes were measured at all points
    if any([size(d, 1) for d in data_group] .!= length(timepoints))
        error("Not all mutant barcodes have reported counts in all time points")
    end # if

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

    if verbose
        println("Sampling posterior...")
    end # if

    # Sample posterior
    chain = Turing.sample(
        model(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, Vector.(eachrow(R̲̲)), n̲ₜ; model_kwargs...),
        sampler,
        ensemble,
        n_steps,
        n_walkers,
        progress=verbose
    )

    if verbose
        println("Saving $(fname) chains...")
    end # if
    # Write output into memory
    JLD2.jldsave("$(fname)", chain=chain, ids=data_keys)
end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Running MCMC for joint distribution of single mutant and all neutral lineages 
# π(s⁽ᵐ⁾, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    mcmc_single_fitness(; kwargs)

Function to sample the joint posterior distribution for the fitness value of a
single mutant barcode and all neutral linages given a time-series barcode count.

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
- `ensemble::Turing.AbstractMCMC.AbstractMCMCEnsemble=Turing.MCMCSerial()`:
Sampling modality to be used. Options are:
    - `Turing.MCMCSerial()`
    - `Turing.MCMCThreads()`
    - `Turing.MCMCDistributed()`
- `verbose::Bool=true`: Boolean indicating if the function should print partial
  progress to the screen or not.
- `multithread::Bool=false`: Boolean indicating whether to use
  `Threads.@threads` when running the `for`-loop over all mutants. NOTE: This
  requires julia to be initialized with multiple threads.
"""
function mcmc_single_fitness(;
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
    ensemble::Turing.AbstractMCMC.AbstractMCMCEnsemble=Turing.MCMCSerial(),
    verbose::Bool=true,
    multithread::Bool=false
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

    # Extract group keys
    data_keys = first.(values.(keys(data_group)))

    # Check that all barcodes were measured at all points
    if any([size(d, 1) for d in data_group] .!= length(timepoints))
        error("Not all mutant barcodes have reported counts in all time points")
    end # if

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

    # Define function to be used with each barcode. This is used to make the
    # multi-thread option simpler. This is because rather than copying the same
    # code twice, we can call this function
    function sample_bc(i, R̲̲⁽ᵐ⁾=R̲̲⁽ᵐ⁾, R̲̲⁽ⁿ⁾=R̲̲⁽ⁿ⁾)
        if verbose
            println("performing inference for $(data_keys[i])...")
        end # if

        ## %%%%%%%%%%% Total barcodes data %%%%%%%%%%% ##
        R̲̲ = hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾[:, i], sum(R̲̲⁽ᵐ⁾[:, 1:size(R̲̲⁽ᵐ⁾, 2).≠i], dims=2))
        # Compute total counts for each run
        n̲ₜ = vec(sum(R̲̲, dims=2))

        ## %%%%%%%%%%% MCMC sampling %%%%%%%%%%% ##
        # Define output filename
        fname = "$(outputname)$(data_keys[i]).jld2"

        # Check if file has been processed before
        if isfile(fname)
            error("$(fname) was already processed")
        end # if

        if verbose
            println("Initialize MCMC sampling with $(Turing.ADBACKEND)...\n")
        end # if

        if verbose
            println("Sampling posterior...")
        end # if

        # Sample posterior
        chain = Turing.sample(
            model(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾[:, i], Vector.(eachrow(R̲̲)), n̲ₜ; model_kwargs...),
            sampler,
            ensemble,
            n_steps,
            n_walkers,
            progress=verbose
        )

        if verbose
            println("Saving $(fname) chains...")
        end # if
        # Write output into memory
        JLD2.jldsave("$(fname)", chain=chain)
    end # function

    # Search for previously-processed files. This is because running in
    # multithread with some files previously process somehow stops the
    # multithreading
    files = Glob.glob("$(outputname)*")
    # Extract previously processed barcodes
    bc_prev = [replace(f, outputname => "", ".jld2" => "") for f in files]
    # Find barcodes not yet processed
    bc_idx = [.!any(bc .== bc_prev) for bc in string.(data_keys)]

    # Check if multithread should be used for mutants
    if multithread
        Threads.@threads for i = collect(1:size(R̲̲⁽ᵐ⁾, 2))[bc_idx]
            try
                sample_bc(i, R̲̲⁽ᵐ⁾, R̲̲⁽ⁿ⁾)
            catch
                @warn "bc $(data_keys[i]) was already processed"
                continue
            end # try/catch
        end # for
    else
        for i = collect(1:size(R̲̲⁽ᵐ⁾, 2))[bc_idx]
            try
                sample_bc(i, R̲̲⁽ᵐ⁾, R̲̲⁽ⁿ⁾)
            catch
                @warn "bc $(data_keys[i]) was already processed"
                continue
            end # try/catch
        end # for
    end # if

end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Running inference for hierarchical model over multiple experimental replicates
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    mcmc_joint_fitness_hierarchical_replicates(; kwargs)

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
- `ensemble::Turing.AbstractMCMC.AbstractMCMCEnsemble=Turing.MCMCSerial()`:
Sampling modality to be used. Options are:
    - `Turing.MCMCSerial()`
    - `Turing.MCMCThreads()`
    - `Turing.MCMCDistributed()`
- `verbose::Bool=true`: Boolean indicating if the function should print partial
  progress to the screen or not.
"""
function mcmc_joint_fitness_hierarchical_replicates(;
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
    rep_col::Symbol=:rep,
    rm_T0::Bool=false,
    sampler::Turing.Inference.InferenceAlgorithm=Turing.NUTS(0.65),
    ensemble::Turing.AbstractMCMC.AbstractMCMCEnsemble=Turing.MCMCSerial(),
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

    # Number of unique replicates in dataset
    n_rep = length(unique(data[:, rep_col]))

    if n_rep == 1
        error("There is only one replicate in the dataset")
    end # if

    ### NOTE: Need to add a verification that all barcodes are reported for 
    # all time points

    ## %%%%%%%%%%% Neutral barcodes data %%%%%%%%%%% ##

    # Extract neutral data
    data_neutral = @view data[data[:, neutral_col], :]
    # Extract unique time points
    timepoints = sort(unique(data_neutral[:, time_col]))
    # Extract unique IDs
    ids = unique(data_neutral[:, id_col])
    # Extract unique reps
    reps = unique(data_neutral[:, rep_col])

    # Initialize array to save counts for each mutant at time t
    R̲̲⁽ⁿ⁾ = Array{Int64,3}(
        undef, length(timepoints), length(ids), length(reps)
    )

    # Loop through each unique id
    for (j, id) in enumerate(ids)
        # Loop through each unique rep
        for (k, rep) in enumerate(reps)
            # Extract data
            d = data_neutral[
                (data_neutral[:, id_col].==id).&(data_neutral[:, rep_col].==rep),
                :]
            # Sort data by timepoint
            DF.sort!(d, time_col)
            # Extract data
            R̲̲⁽ⁿ⁾[:, j, k] = d[:, count_col]
        end # for
    end # for


    ## %%%%%%%%%%% Mutant barcodes data %%%%%%%%%%% ##

    # Extract neutral data
    data_mut = @view data[.!data[:, neutral_col], :]
    # Extract unique time points
    timepoints = sort(unique(data_mut[:, time_col]))
    # Extract unique IDs
    ids_mut = sort(unique(data_mut[:, id_col]))
    # Extract unique reps
    reps = sort(unique(data_mut[:, rep_col]))

    # Initialize array to save counts for each mutant at time t
    R̲̲⁽ᵐ⁾ = Array{Int64,3}(
        undef, length(timepoints), length(ids_mut), length(reps)
    )

    # Loop through each unique id
    for (j, id) in enumerate(ids_mut)
        # Loop through each unique rep
        for (k, rep) in enumerate(reps)
            # Extract data
            d = data_mut[
                (data_mut[:, id_col].==id).&(data_mut[:, rep_col].==rep),
                :]
            # Sort data by timepoint
            DF.sort!(d, time_col)
            # Extract data
            R̲̲⁽ᵐ⁾[:, j, k] = d[:, count_col]
        end # for
    end # for

    ## %%%%%%%%%%% Total barcodes data %%%%%%%%%%% ##

    # Concatenate neutral and mutant data matrices
    R̲̲ = cat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾; dims=2)

    # Compute total counts for each run
    n̲ₜ = reshape(sum(R̲̲, dims=2), length(timepoints), length(reps))

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

    if verbose
        println("Sampling posterior...")
    end # if

    # Sample posterior
    chain = Turing.sample(
        model(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲, n̲ₜ; model_kwargs...),
        sampler,
        ensemble,
        n_steps,
        n_walkers,
        progress=verbose
    )

    if verbose
        println("Saving $(fname) chains...")
    end # if
    # Write output into memory
    JLD2.jldsave("$(fname)", chain=chain, ids=ids_mut)
end # function