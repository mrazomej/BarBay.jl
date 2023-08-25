# Import library to suppress output
import Suppressor

# Import libraries relevant for MCMC
import Turing
import MCMCChains
import DynamicHMC

# Import library to store output
import JLD2

# Import library to locate files
import Glob

# Import package to handle DataFrames
import DataFrames as DF
import CSV

using ..utils: data2arrays
##

# Export function

##

@doc raw"""
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
    which to sample (see `BayesFitness.model` module). This function must take
    as the first four inputs the following:
    - `R̲̲::Array{Int64}`:: 2 or 3D array containing the raw barcode counts for
      all tracked genotypes. The dimensions of this array represent:
      - dim=1: time.
      - dim=2: genotype.
      - dim=3 (optional): experimental repeats
    - `n̲ₜ::VecOrMat{Int64}`: Array with the total number of barcode counts for
        each time point (on each experimental repeat, if necessary).
    - `n_neutral::Int`: Number of neutral lineages.
    - `n_mut::Int`: Number of neutral lineages.

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
- `rep_col::Union{Nothing,Symbol}=nothing`: Optional column in tidy dataframe to
  specify the experimental repeat for each observation.
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
function mcmc_sample(;
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
    rep_col::Union{Nothing,Symbol}=nothing,
    rm_T0::Bool=false,
    sampler::Turing.Inference.InferenceAlgorithm=Turing.NUTS(0.65),
    ensemble::Turing.AbstractMCMC.AbstractMCMCEnsemble=Turing.MCMCSerial(),
    verbose::Bool=true
)
    # Define output filename
    fname = "$(outputname).jld2"

    # Check if file has been processed before
    if isfile(fname)
        error("$(fname) was already processed")
    end # if

    # Check if model is hierarchical for experimental replicates
    if occursin("exprep", "$(model)") & (typeof(rep_col) <: Nothing)
        error("Hierarchical models for experimental replicates require argument `:rep_col`")
    end # if

    ## %%%%%%%%%%% Preprocessing data %%%%%%%%%%% ##

    println("Pre-processing data...")
    # Convert from tidy dataframe to model inputs
    data_dict = data2arrays(
        data;
        id_col=id_col,
        time_col=time_col,
        count_col=count_col,
        neutral_col=neutral_col,
        rep_col=rep_col,
        rm_T0=rm_T0,
        verbose=verbose
    )

    ## %%%%%%%%%%% MCMC sampling %%%%%%%%%%% ##

    if verbose
        println("Initialize MCMC sampling with $(Turing.ADBACKEND)...\n")
    end # if

    if verbose
        println("Sampling posterior...")
    end # if

    # Define model
    bayes_model = model(
        data_dict[:bc_count],
        data_dict[:bc_total],
        data_dict[:n_neutral],
        data_dict[:n_mut];
        model_kwargs...
    )

    # Sample posterior
    chain = Turing.sample(
        bayes_model, sampler, ensemble, n_steps, n_walkers, progress=verbose
    )

    if verbose
        println("Saving $(fname) chain...")
    end # if
    # Write output into memory
    JLD2.jldsave("$(fname)", ids=data_dict[:mut_ids], chain=chain)
end # function