# Import logging for progress updates
import Logging

# Import libraries relevant for Bayesian inference
import Turing
import DynamicPPL

# Import package to handle DataFrames
import DataFrames as DF
import CSV

# Import needed function from the stats.jl module
using ..stats: build_getq

# Import needed function from the utils module
using ..utils: data_to_arrays, advi_to_df

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Running MCMC for full joint fitness inference π(s̲⁽ᵐ⁾, s̲ₜ | data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Import libraries relevant for Bayesian inference
import Turing
import DynamicPPL

# Import package to handle DataFrames
import DataFrames as DF
import CSV

# Import needed function from the stats.jl module
using ..stats: build_getq

# Import needed function from the utils module
using ..utils: data_to_arrays, advi_to_df

@doc raw"""
    advi(; kwargs)

Run Automatic Differentiation Variational Inference (ADVI) to sample the joint
posterior distribution for the fitness value of all mutant and neutral lineages
given a time-series barcode count data.

This function expects the data in a **tidy** format. Each row should represent
**a single observation**. For instance, if barcode `i` is measured at 4
different time points, each of these measurements would have its own row. The
same applies to barcode `j` and its measurements.

The `DataFrame` must contain at least the following columns, specified by their
respective keyword arguments:
- `id_col`: Identifies the barcode ID (e.g., barcode sequence).
- `time_col`: Indicates the measurement time point.
- `count_col`: Contains the raw barcode count.
- `neutral_col`: Indicates if the barcode is from a neutral lineage. Additional
  optional columns include `rep_col`, `env_col`, and `genotype_col`.

# Keyword Arguments
- `data::DF.AbstractDataFrame`: Tidy dataframe with data for sampling the
  population mean fitness posterior distribution.
- `outputname::Union{String,Nothing}`: Name for the output `.csv` file (default
  is `nothing`).
- `model::Function`: `Turing.jl` model defining the posterior distribution.
- `model_kwargs::Dict=Dict()`: Extra keyword arguments for the `model` function.
- `id_col::Symbol=:barcode`: Column in `data` containing barcode identifiers.
- `time_col::Symbol=:time`: Column in `data` defining the time point of
  measurements.
- `count_col::Symbol=:count`: Column in `data` containing raw barcode counts.
- `neutral_col::Symbol=:neutral`: Column in `data` defining neutral lineage.
- `rep_col::Union{Nothing,Symbol}=nothing`: Column for experimental replicate.
- `env_col::Union{Nothing,Symbol}=nothing`: Column for environment.
- `genotype_col::Union{Nothing,Symbol}=nothing`: Column for genotype.
- `advi::Turing.AdvancedVI.VariationalInference=Turing.ADVI{Turing.AutoReverseDiff(true)}(1,
    10_000)`, A default instance of `Turing.AdvancedVI.VariationalInference`
    with the following parameters:
    - `Turing.ADVI{AD}`: The variational inference algorithm used with automatic
      differentiation algorithm `AD`. Default is `Turing.AutoReverseDiff(true)`,
      meaning reverse-mode AD is used with compiled tape for random number
      generation.
    - `(samples_per_step::Int64, max_iters::Int64)`: Number of samples used to
      estimate the ELBO in each optimization step, and Maximum number of
      gradient steps. Default is `(1, 10_000)`.
- `opt::Union{Turing.AdvancedVI.TruncatedADAGrad,Turing.AdvancedVI.DecayedADAGrad}=Turing.Variational.TruncatedADAGrad()`:
  Gradient computation and parameter update algorithm. Default is
  `Turing.Variational.TruncatedADAGrad()`.
- `verbose::Bool=true`: Flag for printing progress updates. Default is `true`.

# Returns
- If `outputname` is `nothing`, returns a `DataFrames.DataFrame` containing
  summary statistics of posterior samples for each parameter.
- If `outputname` is specified, saves the results to a CSV file and does not
  return a DataFrame.
"""
function advi(;
    data::DF.AbstractDataFrame,
    outputname::Union{String,Nothing}=nothing,
    model::Function,
    model_kwargs::Dict=Dict(),
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rep_col::Union{Nothing,Symbol}=nothing,
    env_col::Union{Nothing,Symbol}=nothing,
    genotype_col::Union{Nothing,Symbol}=nothing,
    advi::Turing.AdvancedVI.VariationalInference=Turing.ADVI{Turing.AutoReverseDiff(true)}(1, 10_000),
    opt::Union{Turing.AdvancedVI.TruncatedADAGrad,Turing.AdvancedVI.DecayedADAGrad}=Turing.Variational.TruncatedADAGrad(),
    verbose::Bool=true
)
    # Define output filename
    fname = isnothing(outputname) ? nothing : "$(outputname).csv"

    # Check if file has been processed before
    if !isnothing(fname) && isfile(fname)
        error("$(fname) was already processed")
    end # if

    # Check if model is hierarchical for experimental replicates
    if occursin("replicate", "$(model)") & (typeof(rep_col) <: Nothing)
        error("Hierarchical models for experimental replicates require argument `:rep_col`")
    end # if

    # Check if model is multi-environment
    if occursin("multienv", "$(model)") & (typeof(env_col) <: Nothing)
        error("Models with multiple environments require argument `:env_col`")
    end # if

    ## %%%%%%%%%%% Preprocessing data %%%%%%%%%%% ##

    if verbose
        Logging.@info "Pre-processing data..."
    end # if

    # Convert from tidy dataframe to model inputs using DataArrays struct
    data_arrays = data_to_arrays(
        data;
        id_col=id_col,
        time_col=time_col,
        count_col=count_col,
        neutral_col=neutral_col,
        rep_col=rep_col,
        env_col=env_col,
        genotype_col=genotype_col
    )

    ## %%%%%%%%%%% Variational Inference with ADVI %%%%%%%%%%% ##

    if verbose
        Logging.@info "Initialize Variational Inference Optimization..."
    end # if

    # Check if model is multi-environment to manually add the list of environments
    if occursin("multienv", "$(model)")
        # Initialize empty dictionary that accepts any type
        mk = Dict{Symbol,Any}(:envs => data_arrays.envs)
        # Loop through elements of model_kwargs
        for (key, item) in model_kwargs
            # Add element to flexible dictionary
            setindex!(mk, item, key)
        end # for
        # Change mk name to model_kwargs
        model_kwargs = mk
    end # if

    # Check if model is hierarchical on genotypes to manually add genotype list
    if occursin("genotype", "$(model)")
        # Initialize empty dictionary that accepts any type
        mk = Dict{Symbol,Any}(:genotypes => data_arrays.genotypes)
        # Loop through elements of model_kwargs
        for (key, item) in model_kwargs
            # Add element to flexible dictionary
            setindex!(mk, item, key)
        end # for
        # Change mk name to model_kwargs
        model_kwargs = mk
    end # if

    # Define model
    bayes_model = model(
        data_arrays.bc_count,
        data_arrays.bc_total,
        data_arrays.n_neutral,
        data_arrays.n_bc;
        model_kwargs...
    )

    # Extract model VarInfo
    varinfo = DynamicPPL.VarInfo(bayes_model)

    # Extract variable names
    var_keys = keys(varinfo)

    # Extract number of variables per group
    var_len = [length(varinfo[v]) for v in var_keys]

    # Initialize array to save variable names
    var_names = []

    # Loop through variables
    for (i, v) in enumerate(var_keys)
        # Convert variable to string
        v = String(Symbol(v))
        # Add variable names
        push!(var_names, ["$v[$x]" for x in 1:var_len[i]]...)
    end # for

    # Optimize meanfield variational distribution
    q = Turing.vi(bayes_model, advi; optimizer=opt)

    if isnothing(outputname)
        return advi_to_df(
            data,
            q,
            var_names;
            id_col=id_col,
            time_col=time_col,
            count_col=count_col,
            neutral_col=neutral_col,
            rep_col=rep_col,
            env_col=env_col,
            genotype_col=genotype_col
        )
    else
        # Convert and save output as tidy dataframe
        CSV.write(
            fname,
            advi_to_df(
                data,
                q,
                var_names;
                id_col=id_col,
                time_col=time_col,
                count_col=count_col,
                neutral_col=neutral_col,
                rep_col=rep_col,
                env_col=env_col,
                genotype_col=genotype_col
            )
        )
        return nothing
    end # if
end # function