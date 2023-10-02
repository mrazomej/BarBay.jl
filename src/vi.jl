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

@doc raw"""
    advi(; kwargs)

Function to sample the joint posterior distribution for the fitness value of all
mutant and neutral lineages given a time-series barcode count.

This function expects the data in a **tidy** format. This means that every row
represents **a single observation**. For example, if we measure barcode `i` in 4
different time points, each of these four measurements gets an individual row.
Furthermore, barcode `j` measurements over time also get their own individual
rows. 
        
The `DataFrame` must contain at least the following columns:
- `id_col`: Column identifying the ID of the barcode. This can be the barcode
    sequence, for example.
- `time_col`: Column defining the measurement time point.  
- `count_col`: Column with the raw barcode count.
- `neutral_col`: Column indicating whether the barcode is from a neutral
  lineage.

# Keyword Arguments
- `data::DataFrames.AbstractDataFrame`: **Tidy dataframe** with the data to be
used to sample from the population mean fitness posterior distribution.
- `n_walkers::Int`: Number of walkers (chains) for the MCMC sample.
- `n_steps::Int`: Number of steps to take.
- `outputname::String`: String to name the `.csv` output file.
- `model::Function`: `Turing.jl` model defining the posterior distribution from
    which to sample (see `BayesFitness.model` module). This function must take
    as the first four inputs the following:
    - `R̲̲::Array{Int64}`: 2 or 3D array containing the raw barcode counts for
        all tracked genotypes. The dimensions of this array represent:  
        - dim=1: time.
        - dim=2: genotype.
        - dim=3 (optional): experimental repeats
    - `n̲ₜ::VecOrMat{Int64}`: Array with the total barcode counts for each time
        point (on each experimental repeat, if necessary).
    - `n_neutral::Int`: Number of neutral lineages.
    - `n_bc::Int`: Number of neutral lineages.

## Optional Keyword Arguments
- `model_kwargs::Dict=Dict()`: Extra keyword arguments to be passed to the
  `model` function.
    - `id_col::Symbol=:barcode`: Name of the column in `data` containing the
    barcode identifier. The column may include any type of entry.
- `time_col::Symbol=:time`: Name of the column in `data` defining the time point
  at which measurements were done. The column may contain any type of entry as
  long as `sort` will result in time-ordered names.
- `count_col::Symbol=:count`: Name of the column in `data` containing the raw
  barcode count. The column must contain entries of type `Int64`.
- `neutral_col::Symbol=:neutral`: Name of the column in `data` defining whether
  the barcode belongs to a neutral lineage. The column must contain entries of
  type `Bool`.
- `rep_col::Union{Nothing,Symbol}=nothing`: Column indicating the experimental
  replicate each measurement belongs to. Default is `nothing`.
- `env_col::Union{Nothing,Symbol}=nothing`: Column indicating the environment in
  which each measurement was performed. Default is `nothing`.
- `genotype_col::Union{Nothing,Symbol}=nothing`: Column indicating the genotype
  each barcode belongs to when fitting a hierarchical model on genotypes.
  Default is `nothing`.
- `rm_T0::Bool=false`: Optional argument to remove the first time point from the
  inference. The data from this first time point is commonly of much lower
  quality. Therefore, removing this first time point might result in a better
  inference.
- `advi::Turing.AdvancedVI.VariationalInference=Tuing.ADVI(1, 10_000)`:
  Variational inference algorithm to infer. Currently, `Turing.jl` only supports
  `ADVI`, where the first input is the number of samples to take (empirically
  one sample works), and the second is the number of update steps to take.
- `opt::Union{Turing.AdvancedVI.DecayedADAGrad,Flux.Optimise.AbstractOptimiser}
  = Turing.Variational.DecayedADAGrad(1e-2, 1.1, 0.9)`: Algorithm used to
  compute the model gradient and update the parameters. `Turing.ADVI` can take
  `Flux.jl` optimizers. But the recommended algorithm in `Stan` is the default
  `DecayedADAGrad`.  
- `verbose::Bool=true`: Boolean indicating if the function should print partial
  progress to the screen or not.
  
# Returns
- `df::DataFrames.DataFrame`: DataFrame containing summary statistics of
  posterior samples for each parameter. Columns include:
    - `mean, std`: posterior mean and standard deviation for each variable.
    - `varname`: parameter name from the ADVI posterior distribution.
    - `vartype`: Description of the type of parameter. The types are:
        - `pop_mean_fitness`: Population mean fitness value `s̲ₜ`.
        - `pop_error`: (Nuisance parameter) Log of standard deviation in the
          likelihood function for the neutral lineages.
        - `bc_fitness`: Mutant relative fitness `s⁽ᵐ⁾`.
        - `bc_hyperfitness`: For hierarchical models, mutant hyperparameter that
          connects the fitness over multiple experimental replicates or multiple
          genotypes `θ⁽ᵐ⁾`.
        - `bc_noncenter`: (Nuisance parameter) For hierarchical models,
          non-centered samples used to connect the experimental replicates to
          the hyperparameter `θ̃⁽ᵐ⁾`.
        - `bc_deviations`: (Nuisance parameter) For hierarchical models, samples
          that define the log of the deviation from the hyperparameter fitness
          value `logτ⁽ᵐ⁾`.
        - `bc_error`: (Nuisance parameter) Log of standard deviation in the
          likelihood function for the mutant lineages.
        - `freq`: (Nuisance parameter) Log of the Poisson parameter used to
          define the frequency of each lineage.
    - `rep`: Experimental replicate number.
    - `env`: Environment for each parameter.
    - `id`: Mutant or neutral strain ID.
"""
function advi(;
    data::DF.AbstractDataFrame,
    outputname::String,
    model::Function,
    model_kwargs::Dict=Dict(),
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rep_col::Union{Nothing,Symbol}=nothing,
    env_col::Union{Nothing,Symbol}=nothing,
    genotype_col::Union{Nothing,Symbol}=nothing,
    rm_T0::Bool=false,
    advi::Turing.AdvancedVI.VariationalInference=Tuing.ADVI(1, 10_000),
    opt::Union{Turing.AdvancedVI.TruncatedADAGrad,Turing.AdvancedVI.DecayedADAGrad}=Turing.Variational.TruncatedADAGrad(),
    fullrank::Bool=false,
    verbose::Bool=true
)
    # Define output filename
    fname = "$(outputname).csv"

    # Check if file has been processed before
    if isfile(fname)
        error("$(fname) was already processed")
    end # if

    # Check if model is hierarchical for experimental replicates
    if occursin("replicate", "$(model)") & (typeof(rep_col) <: Nothing)
        error("Hierarchical models for experimental replicates require argument `:rep_col`")
    end # if

    # Check if model is hierarchical for experimental replicates
    if occursin("multienv", "$(model)") & (typeof(env_col) <: Nothing)
        error("Models with multiple environments require argument `:env_col`")
    end # if

    ## %%%%%%%%%%% Preprocessing data %%%%%%%%%%% ##

    println("Pre-processing data...")
    # Convert from tidy dataframe to model inputs
    data_dict = data_to_arrays(
        data;
        id_col=id_col,
        time_col=time_col,
        count_col=count_col,
        neutral_col=neutral_col,
        rep_col=rep_col,
        env_col=env_col,
        genotype_col=genotype_col,
        rm_T0=rm_T0,
        verbose=verbose
    )

    ## %%%%%%%%%%% Variational Inference with ADVI %%%%%%%%%%% ##
    if verbose
        println("Initialize Variational Inference Optimization...\n")
    end # if


    # Check if model is multi-environment to manually add the list of
    # environments
    if occursin("multienv", "$(model)")
        # Initialize empty dictionary that accepts any type
        mk = Dict{Symbol,Any}(:envs => data_dict[:envs])
        # Loop through elements of model_kwargs
        for (key, item) in model_kwargs
            # Add element to flexible dictionary
            setindex!(mk, item, key)
        end # for
        # Change mk name to model_kwargs
        model_kwargs = mk
    end

    # Check if model is hierarchical on genotypes to manually add genotype list
    if occursin("genotype", "$(model)")
        # Initialize empty dictionary that accepts any type
        mk = Dict{Symbol,Any}(:genotypes => data_dict[:genotypes])
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
        data_dict[:bc_count],
        data_dict[:bc_total],
        data_dict[:n_neutral],
        data_dict[:n_bc];
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

    # Check if variational problem is meanfield or full-rank
    if !fullrank
        # Optimize meanfield variational distribution
        q = Turing.vi(bayes_model, advi; optimizer=opt)
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
                genotype_col=genotype_col,
                rm_T0=rm_T0
            )
        )
    else
        # Obtain number of parameters.
        n_param = sum(var_len)

        # Build getq function
        getq = build_getq(n_param, bayes_model)

        # Define TOTAL number of parameters to be optimized, including
        # covariance terms.
        n_param_total = (n_param * n_param) + n_param

        # Optimize full-rank variational distribution.
        q = Turing.vi(
            bayes_model,
            advi,
            getq,
            randn(n_param_total);
            optimizer=opt
        )

        return (q, var_names)
    end # if
end # function