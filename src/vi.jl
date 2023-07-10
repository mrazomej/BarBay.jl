# Import Suppressor to silence warnings
import Suppressor

# Import Flux before Turing to get access to optimizers
import Flux

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

# Import needed function from the stats.jl module
using ..stats: build_getq

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Running MCMC for full joint fitness inference π(s̲⁽ᵐ⁾, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    vi_joint_fitness(; kwargs)

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
- `advi::Turing.AdvancedVI.VariationalInference=Tuing.ADVI(1, 10_000)`:
  Variational inference algorithm to infer. Currently, `Turing.jl` only supports
  `ADVI`, where the first input is the number of samples to take (empirically
  one sample works) and the second input is the number of update steps to take.
- `opt::Union{Turing.AdvancedVI.DecayedADAGrad,Flux.Optimise.AbstractOptimiser}
  = Turing.Variational.DecayedADAGrad(1e-2, 1.1, 0.9)`: Algorithm used to
  compute the model gradient and update the parameters. `Turing.ADVI` can take
  `Flux.jl` optimizers. But the recommended algorithm used in `Stan` is the
  default `DecayedADAGrad`.
- `verbose::Bool=true`: Boolean indicating if the function should print partial
  progress to the screen or not.
"""
function vi_joint_fitness(;
    data::DF.AbstractDataFrame,
    outputname::String,
    model::Function,
    model_kwargs::Dict=Dict(),
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rm_T0::Bool=false,
    advi::Turing.AdvancedVI.VariationalInference=Tuing.ADVI(1, 10_000),
    opt::Union{Turing.AdvancedVI.DecayedADAGrad,Flux.Optimise.AbstractOptimiser}=Turing.Variational.DecayedADAGrad(1e-2, 1.1, 0.9),
    fullrank::Bool=false,
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

    ## %%%%%%%%%%% Variational Inference %%%%%%%%%%% ##
    # Define output filename
    fname = "$(outputname).jld2"

    # Check if file has been processed before
    if isfile(fname)
        error("$(fname) was already processed")
    end # if

    if verbose
        println("Initialize Variational Inference Optimization...\n")
    end # if

    # Define model
    m = model(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲, n̲ₜ; model_kwargs...)


    # Check if variational problem is meanfield or full-rank
    if !fullrank
        # Optimize meanfield variational distribution
        q = Turing.vi(m, advi; optimizer=opt)
    else
        # Obtain number of parameters. This is done in a very inefficient way,
        # but Turing.jl does not have a straightforward way to access the number
        # of parameters in a model.
        # 1. Take one sample from the prior distribution suppressing the output
        #    warnings.
        Suppressor.@suppress begin
            global chn = Turing.sample(m, Turing.Prior(), 1)
        end # @suppress
        # 2. obtain number of parameters from chain size. We subtract one to
        #    match the correct dimension. The extra element in the chain is the
        #    evaluation of the log probability.
        n_param = size(chn, 2) - 1

        # Build getq function
        getq = build_getq(n_param, m)

        # Define TOTAL number of parameters to be optimized, including
        # covariance terms.
        n_param_total = (n_param * n_param) + n_param

        # Optimize full-rank variational distribution.
        q = Turing.vi(
            m,
            advi,
            getq,
            randn(n_param_total);
            optimizer=opt
        )
    end # if

    # Write output into memory
    JLD2.jldsave("$(fname)", dist=q, ids=data_keys)
end # function