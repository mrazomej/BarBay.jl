##
# Import package to handle dataframes
import DataFrames as DF
import CSV

# Import basic math
import StatsBase
import Distributions
import Random

# Import function to list files
import Glob

##

@doc raw"""
    data_to_arrays(data; kwargs)

Function to preprocess the tidy dataframe `data` into the corresponding inputs
for the models in the `model` submodule.

# Arguments
- `data::DataFrames.AbstractDataFrame`: Tidy dataframe with the data to be used
  for sampling the model posterior distribution. 

## Optional Keyword Arguments
- `id_col::Symbol=:barcode`: Name of the column in `data` containing the barcode
    identifier. The column may include any type of entry.
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
- `verbose::Bool=true`: Boolean indicating if printing statements should be
  made.

# Returns
- `data_arrays::Dict`: Dictionary with the following elements:
    - `bc_ids`: List of barcode IDs in the order they are used for the
      inference.
    - `neutral_ids`: List of neutral barcode IDs in the order they are used for
      the inference.
    - `bc_count`: Count time series for each barcode. The options can be:
        - `Matrix{Int64}`: (n_time) × (n_bc) matrix with counts. Rows are time
          points, and columns are barcodes.
        - `Array{Int64, 3}`: The same as the matrix, except the third dimension
          represents multiple experimental replicates.
        - `Vector{Matrix{Int64}}`: List of matrices, one for each experimental
          replicate. This is when replicates have a different number of time
          points.
    - `bc_total`: Total number of barcodes per time point. The options can be:
        - `Vector{Int64}`: Equivalent to summing each matrix row.
        - `Matrix{Int64}`: Equivalent to summing each row of each slice of the
          tensor.
        - `Vector{Vector{Int64}}`: Equivalent to summing each matrix row.
    - `n_rep`: Number of experimental replicates.
    - `n_time`: Number of time points. The options can be:
        - `Int64`: Number of time points on a single replicate or multiple
          replicates.
        - `Vector{Int64}`: Number of time points per replicate when replicates
          have different lengths.
    - `envs`: List of environments. The options can be:
        - `String`: Single placeholder `env1`
        - `Vector{<:Any}`: Environments in the order they were measured.
        - `vector{Vector{<:Any}}`: Environments per replicate when replicates
          have a different number of time points.
    - `n_env`: Number of environmental conditions.
    - `genotypes`: List of genotypes for each of the non-neutral barcodes. The
      options can be:
        - `N/A`: String when no genotype information is given.
        - `Vector{<:Any}`: Vector of the corresponding genotype for each of the
          non-neutral barcodes in the order they are used for the inference.
    - `n_geno`: Number of genotypes. When no genotype information is provided,
      this defaults to zero.
"""
function data_to_arrays(
    data::DF.AbstractDataFrame;
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rep_col::Union{Nothing,Symbol}=nothing,
    env_col::Union{Nothing,Symbol}=nothing,
    genotype_col::Union{Nothing,Symbol}=nothing,
    rm_T0::Bool=false,
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

    ### --------------- When no repeats are given --------------- ### 
    if typeof(rep_col) <: Nothing
        ## %%%%%%%%%%% Neutral barcodes data %%%%%%%%%%% ##

        # Group data by unique mutant barcode
        data_group = DF.groupby(data[data[:, neutral_col], :], id_col)

        # Extract group keys
        neutral_ids = first.(values.(keys(data_group)))

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
        bc_ids = first.(values.(keys(data_group)))

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

        # Define number of replicates
        n_rep = 1

        ### --------------- When replicates are given --------------- ### 
    elseif typeof(rep_col) <: Symbol

        ## === Check if all replicates have the same number of time points === #
        # Group data by replicate
        data_rep_group = DF.groupby(data, rep_col)
        # Define number of replicates
        n_rep = length(data_rep_group)
        # Define number of time points per replicate
        n_rep_time = [length(unique(d[:, time_col])) for d in data_rep_group]

        if length(unique(n_rep_time)) == 1

            ## %%%%%%%%%%% Neutral barcodes data %%%%%%%%%%% ##

            # Extract neutral data
            data_neutral = @view data[data[:, neutral_col], :]
            # Extract unique time points
            timepoints = sort(unique(data_neutral[:, time_col]))
            # Extract unique IDs
            neutral_ids = unique(data_neutral[:, id_col])
            # Extract unique reps
            reps = unique(data_neutral[:, rep_col])

            # Initialize array to save counts for each mutant at time t
            R̲̲⁽ⁿ⁾ = Array{Int64,3}(
                undef, length(timepoints), length(neutral_ids), length(reps)
            )

            # Loop through each unique id
            for (j, id) in enumerate(neutral_ids)
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
            bc_ids = sort(unique(data_mut[:, id_col]))
            # Extract unique reps
            reps = sort(unique(data_mut[:, rep_col]))

            # Initialize array to save counts for each mutant at time t
            R̲̲⁽ᵐ⁾ = Array{Int64,3}(
                undef, length(timepoints), length(bc_ids), length(reps)
            )

            # Loop through each unique id
            for (j, id) in enumerate(bc_ids)
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

        else
            # Initialize Vector to save matrix for each replicate
            R̲̲ = Vector{Matrix{Int64}}(undef, length(n_rep_time))

            # Loop through replicates
            for (rep, d_rep) in enumerate(data_rep_group)
                ## %%%%%%%%%%% Neutral barcodes data %%%%%%%%%%% ##

                # Group data by unique mutant barcode
                data_group = DF.groupby(d_rep[d_rep[:, neutral_col], :], id_col)

                # Extract group keys
                neutral_ids = first.(values.(keys(data_group)))

                # Check that all barcodes were measured at all points
                if any([size(d, 1) for d in data_group] .!= n_rep_time[rep])
                    error("Not all neutral barcodes have reported counts in all time points")
                end # if

                # Initialize array to save counts for each mutant at time t
                R̲̲⁽ⁿ⁾ = Matrix{Int64}(
                    undef, n_rep_time[rep], length(data_group)
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
                data_group = DF.groupby(
                    d_rep[.!d_rep[:, neutral_col], :], id_col
                )

                # Extract group keys
                bc_ids = first.(values.(keys(data_group)))

                # Check that all barcodes were measured at all points
                if any([size(d, 1) for d in data_group] .!= n_rep_time[rep])
                    error("Not all mutant barcodes have reported counts in all time points")
                end # if

                # Initialize array to save counts for each mutant at time t
                R̲̲⁽ᵐ⁾ = Matrix{Int64}(
                    undef, n_rep_time[rep], length(data_group)
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
                R̲̲[rep] = hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾)

            end # for
            # Compute total counts for each run
            n̲ₜ = vec.(sum.(R̲̲, dims=2))
        end #if
    end # if

    # Extract list of environments
    if (typeof(env_col) <: Nothing)
        # Define environments
        envs = ["env1"]
        # Define single environment when no information is given
        n_env = 1
    elseif (typeof(env_col) <: Symbol) & (n_rep == 1)
        # collect environments for single-replicate case
        envs = collect(sort(unique(data[:, [time_col, env_col]]), time_col)[:, env_col])
        # Define number of environments
        n_env = length(unique(envs))
    elseif (typeof(env_col) <: Symbol) & (n_rep > 1)
        # Group data by replicate
        data_rep = DF.groupby(data, rep_col)
        # collect environments for multi-replicate with different number of time
        # points
        envs = [
            collect(sort(unique(d[:, [time_col, env_col]]), time_col)[:, env_col])
            for d in data_rep
        ]
        # Define number of environments
        n_env = length(unique(reduce(vcat, envs)))
        # Check if all replicates have same environments
        if length(unique(envs)) == 1
            envs = first(envs)
        end # if
    end # if

    # Extract number of time points per replicate
    if n_rep == 1
        # Extract number of time points
        n_time = length(unique(data[:, time_col]))
    elseif n_rep > 1
        # Group data by replicate
        data_rep = DF.groupby(data, rep_col)
        # Extract names of replicates
        reps = collect(first.(values.(keys(data_rep))))
        # Extract number of time points per replicate
        n_time = [length(unique(d[:, time_col])) for d in data_rep]
    end #if

    # Check if genotype list is given
    if typeof(genotype_col) <: Nothing
        # Assign N/A to genotypes
        genotypes = "N/A"
        # Assign 0 to genotypes
        n_genotypes = 0
    elseif typeof(genotype_col) <: Symbol
        # Generate dictionary from bc to genotype
        bc_geno_dict = Dict(
            values.(keys(DF.groupby(data, [id_col, genotype_col])))
        )
        # Extract genotypes in the order they will be used in the inference
        genotypes = [bc_geno_dict[m] for m in bc_ids]
        # Compute number of genotypes
        n_genotypes = length(unique(genotypes))
    end # if

    return Dict(
        :bc_count => R̲̲,
        :bc_total => n̲ₜ,
        :n_neutral => size(R̲̲⁽ⁿ⁾, 2),
        :n_bc => size(R̲̲⁽ᵐ⁾, 2),
        :bc_ids => bc_ids,
        :neutral_ids => neutral_ids,
        :envs => envs,
        :n_env => n_env,
        :n_rep => n_rep,
        :n_time => n_time,
        :genotypes => genotypes,
        :n_geno => n_genotypes
    )
end # function

@doc raw"""
advi_to_df(data::DataFrames.AbstractDataFrame, dist::Distribution.Sampleable,
           vars::Vector{<:Any}; kwargs)

Convert the output of automatic differentiation variational inference (ADVI) to
a tidy dataframe.

# Arguments
- `data::DataFrames.AbstractDataFrame`: Tidy dataframe used to perform the ADVI
  inference. See `BayesFitness.vi` module for the dataframe requirements.
- `dist::Distributions.Sampleable`: The ADVI posterior sampleable distribution
  object.
- `vars::Vector{<:Any}`: Vector of variable/parameter names from the ADVI run. 

## Optional Keyword Arguments
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
- `n_samples::Int=10_000`: Number of posterior samples to draw used for
  hierarchical models. Default is 10,000.

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
        - `bc_hyperfitness`: For hierarchical models, mutant hyperparameter
          that connects the fitness over multiple experimental replicates or
          multiple genotypes `θ⁽ᵐ⁾`.
        - `bc_noncenter`: (Nuisance parameter) For hierarchical models,
          non-centered samples used to connect the experimental replicates to
          the hyperparameter `θ̃⁽ᵐ⁾`.
        - `bc_deviations`: (Nuisance parameter) For hierarchical models,
          samples that define the log of the deviation from the hyperparameter
          fitness value `logτ⁽ᵐ⁾`.
        - `bc_error`: (Nuisance parameter) Log of standard deviation in the
          likelihood function for the mutant lineages.
        - `freq`: (Nuisance parameter) Log of the Poisson parameter used to
          define the frequency of each lineage.
    - rep: Experimental replicate number.
    - env: Environment for each parameter.
    - id: Mutant or neutral strain ID.

# Notes
- Converts multivariate posterior into summarized dataframe format.
- Adds metadata like parameter type, replicate, strain ID, etc.
- Can handle models with multiple replicates and environments.
- Can handle models with hierarchical structure on genotypes.
- Useful for post-processing ADVI results for further analysis and plotting.
"""
function advi_to_df(
    data::DF.AbstractDataFrame,
    dist::Distributions.Sampleable,
    vars::Vector{<:Any};
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rep_col::Union{Nothing,Symbol}=nothing,
    env_col::Union{Nothing,Symbol}=nothing,
    genotype_col::Union{Nothing,Symbol}=nothing,
    rm_T0::Bool=false,
    n_samples::Int=10_000
)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Extract information from data
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Convert data to arrays to obtain information
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

    # Extract number of neutral and mutant lineages
    n_neutral = data_arrays[:n_neutral]
    n_bc = data_arrays[:n_bc]

    # Extract bc lineages IDs
    bc_ids = data_arrays[:bc_ids]
    # Extract neutral lineages IDs
    neutral_ids = data_arrays[:neutral_ids]

    # Extract number of replicates
    n_rep = data_arrays[:n_rep]
    # Extract number of time points per replicate
    n_time = data_arrays[:n_time]

    # Extract number of environments
    n_env = data_arrays[:n_env]
    # Extract list of environments
    envs = data_arrays[:envs]
    # For multi-replicate inferences replicate the list of environments when
    # necessary
    if (n_env > 1) & (n_rep > 1) & !(typeof(envs) <: Vector{<:Vector})
        envs = repeat([envs], n_rep)
    end # if

    # Extract genotype information
    genotypes = data_arrays[:genotypes]
    n_geno = data_arrays[:n_geno]

    # Extract unique replicates
    if typeof(rep_col) <: Symbol
        reps = sort(unique(data[:, rep_col]))
    end # if
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Convert distribution parameters into tidy dataframe
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Extract parameters and convert to dataframe
    df_par = DF.DataFrame(hcat(dist.dist.m, dist.dist.σ), ["mean", "std"])

    # Add variable name. Notice that we use deepcopy to avoid the modification
    # of the original variable
    df_par[!, :varname] = deepcopy(vars)

    # Locate variable groups by identifying the first variable of each group
    var_groups = replace.(vars[occursin.("[1]", string.(vars))], "[1]" => "")

    # Define ranges for each variable
    var_range = dist.transform.ranges_out

    # Count how many variables exist per group
    var_count = length.(var_range)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Add vartype information
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Define dictionary from varname to vartype
    varname_to_vartype = Dict(
        "s̲ₜ" => "pop_mean_fitness",
        "logσ̲ₜ" => "pop_std",
        "s̲⁽ᵐ⁾" => "bc_fitness",
        "logσ̲⁽ᵐ⁾" => "bc_std",
        "θ̲⁽ᵐ⁾" => "bc_hyperfitness",
        "θ̲̃⁽ᵐ⁾" => "bc_noncenter",
        "logτ̲⁽ᵐ⁾" => "bc_deviations",
        "logΛ̲̲" => "log_poisson",
    )

    # Add column to be modified with vartype
    df_par[!, :vartype] .= "tmp"
    # Loop through variables
    for (i, var) in enumerate(var_groups)
        df_par[var_range[i], :vartype] .= varname_to_vartype[var]
    end # for

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Add replicate number information
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    if (n_rep == 1) & (typeof(genotype_col) <: Nothing) &
       (length(var_groups) > 5)
        # Report error if no env or rep info is given when needed
        error("The distribution seems to match a hierarchical model but no genotype or replicate information was provided")
    elseif (n_rep == 1) & (length(var_groups) == 5) & (n_env == 1)
        println("Single replicate non-hierarchical model")
        # Add replicate for single-replicate model for consistency
        df_par[!, :rep] .= "R1"
    elseif (n_rep == 1) & (length(var_groups) == 7) &
           (typeof(genotype_col) <: Symbol)
        println("Single replicate genotype hierarchical model")
        # Add replicate for single-replicate model for consistency
        df_par[!, :rep] .= "R1"
    elseif (n_rep > 1)
        # Add replicate column to be modified
        df_par[!, rep_col] = Vector{Any}(undef, size(df_par, 1))
        # Loop through var groups
        for (i, var) in enumerate(var_groups)
            if occursin("̲ₜ", var)
                # Add replicate for population mean fitness variables
                df_par[var_range[i], rep_col] = reduce(
                    vcat,
                    [
                        repeat([reps[j]], n_time[j] - 1) for j = 1:n_rep
                    ]
                )
            elseif var == "θ̲⁽ᵐ⁾"
                # Add information for hyperparameter that does not have
                # replicate information
                df_par[var_range[i], rep_col] .= "N/A"
            elseif var == "logΛ̲̲"
                df_par[var_range[i], rep_col] = reduce(
                    vcat,
                    [
                        repeat(
                            [reps[j]],
                            (n_bc + n_neutral) * (n_time[j])
                        ) for j = 1:n_rep
                    ]
                )
            else
                # Information on the other variables depends on environments
                if n_env == 1
                    # Add information for bc-related parameters when no
                    # environment information is provided
                    df_par[var_range[i], rep_col] = reduce(
                        vcat,
                        [
                            repeat([reps[j]], n_bc) for j = 1:n_rep
                        ]
                    )
                else
                    df_par[var_range[i], rep_col] = reduce(
                        vcat,
                        [
                            repeat([reps[j]], n_bc * n_env) for j = 1:n_rep
                        ]
                    )
                end # if
            end # if
        end # for
    end # if

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Add environment information
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    if (n_env == 1)
        # Add environment information for consistency
        df_par[!, :env] .= "env1"
    elseif (n_env > 1)
        # Add replicate column to be modified
        df_par[!, env_col] = Vector{Any}(undef, size(df_par, 1))
        # Loop through var groups
        for (i, var) in enumerate(var_groups)
            if occursin("̲ₜ", var)
                if n_rep == 1
                    # Add environment information for single replicate
                    df_par[var_range[i], env_col] = envs[2:end]
                elseif n_rep > 1
                    # Add environment information for multiple replicates
                    df_par[var_range[i], env_col] = reduce(
                        vcat, [env[2:end] for env in envs]
                    )
                end # if
            elseif var == "θ̲⁽ᵐ⁾"
                # Add environment information for hyperparameter
                df_par[var_range[i], env_col] = reduce(
                    vcat, repeat([unique(reduce(vcat, envs))], n_bc)
                )
            elseif var == "logΛ̲̲"
                if n_rep == 1
                    # Add environment informatin for each Poisson parameter for
                    # single replicate
                    df_par[var_range[i], env_col] = repeat(
                        envs, (n_bc + n_neutral)
                    )
                elseif n_rep > 1
                    # Add environment information for each Poisson parameter for
                    # multiple replicates
                    df_par[var_range[i], env_col] = reduce(
                        vcat,
                        [repeat(env, (n_bc + n_neutral)) for env in envs]
                    )
                end # if
            else
                if n_rep == 1
                    # Add environment information for each bc-related
                    # variable for single replicate
                    df_par[var_range[i], env_col] = reduce(
                        vcat,
                        repeat(unique(envs), n_bc)
                    )
                elseif n_rep > 1
                    # Add environment information for each bc-related
                    # variable for multiple replicates
                    df_par[var_range[i], env_col] = reduce(
                        vcat,
                        repeat(unique(reduce(vcat, envs)), n_bc * n_rep)
                    )
                end #if
            end # if
        end # for
    end # if

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Add mutant id information
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Add mutant id column. To be modified
    df_par[:, :id] .= "N/A"

    # Loop through var groups
    for (i, var) in enumerate(var_groups)
        if occursin("̲ₜ", var)
            # Population mean fitness variables are not associated with any
            # particular barcode.
            continue
        elseif var == "θ̲⁽ᵐ⁾"
            if (n_env == 1) & (typeof(genotype_col) <: Nothing)
                # Add ID information to hyperparameter for single environment
                # and no genotypes
                df_par[var_range[i], :id] = bc_ids
            elseif (n_env == 1) & (typeof(genotype_col) <: Symbol)
                # Add ID information to hyperparameter fitness for genotypes
                df_par[var_range[i], :id] = unique(genotypes)
            elseif n_env > 1
                # Add ID information to hyperparameter for multiple environments
                df_par[var_range[i], :id] = reduce(
                    vcat,
                    [repeat([bc], n_env) for bc in bc_ids]
                )
            end # if
        elseif var == "logΛ̲̲"
            if n_rep == 1
                # Add ID information to Poisson parameter for single replicate
                df_par[var_range[i], :id] = reduce(
                    vcat,
                    [repeat([bc], n_time) for bc in [neutral_ids; bc_ids]]
                )
            elseif n_rep > 1
                df_par[var_range[i], :id] = reduce(
                    vcat,
                    [
                        repeat([bc], n_time[rep])
                        for rep = 1:n_rep
                        for bc in [neutral_ids; bc_ids]
                    ]
                )
            end # if
        else
            if (n_rep == 1) & (n_env == 1)
                df_par[var_range[i], :id] = bc_ids
            elseif (n_rep == 1) & (n_env > 1)
                df_par[var_range[i], :id] = reduce(
                    vcat,
                    [repeat([bc], n_env) for bc in bc_ids]
                )
            elseif (n_rep > 1) & (n_env == 1)
                df_par[var_range[i], :id] = reduce(
                    vcat,
                    repeat(bc_ids, n_rep)
                )
            elseif (n_rep > 1) & (n_env > 1)
                df_par[var_range[i], :id] = reduce(
                    vcat,
                    [
                        repeat([bc], n_env)
                        for rep = 1:n_rep
                        for bc in bc_ids
                    ]
                )

            end # if
        end # if
    end # for

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Add per-replicate fitness effect variables 
    # (for hierarchical models on experimental replicates)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    if (n_rep > 1) & (length(var_groups) == 7)
        # Sample θ̲ variables
        θ_mat = hcat(
            [
                Random.rand(Distributions.Normal(x...), n_samples)
                for x in eachrow(
                    df_par[
                        df_par.vartype.=="bc_hyperfitness",
                        [:mean, :std]
                    ]
                )
            ]...
        )

        # Sample τ̲ variables
        τ_mat = exp.(
            hcat(
                [
                    Random.rand(Distributions.Normal(x...), n_samples)
                    for x in eachrow(
                        df_par[
                            df_par.vartype.=="bc_deviations",
                            [:mean, :std]
                        ]
                    )
                ]...
            )
        )

        # Sample θ̲̃ variables
        θ_tilde_mat = hcat(
            [
                Random.rand(Distributions.Normal(x...), n_samples)
                for x in eachrow(
                    df_par[
                        df_par.vartype.=="bc_noncenter",
                        [:mean, :std]
                    ]
                )
            ]...
        )


        # Compute individual strains fitness values
        s_mat = hcat(repeat([θ_mat], n_rep)...) .+ (τ_mat .* θ_tilde_mat)

        # Compute mean and standard deviation and turn into dataframe
        df_s = DF.DataFrame(
            hcat(
                [StatsBase.median.(eachcol(s_mat)),
                    StatsBase.std.(eachcol(s_mat))]...
            ),
            ["mean", "std"]
        )

        # To add the rest of the columns, we will use the τ variables that have
        # the same length. Let's locate such variables
        τ_idx = occursin.("τ", string.(vars))

        # Add extra columns
        DF.insertcols!(
            df_s,
            :varname => replace.(df_par[τ_idx, :varname], "logτ" => "s"),
            :vartype .=> "bc_fitness",
            rep_col => df_par[τ_idx, rep_col],
            :env => df_par[τ_idx, :env],
            :id => df_par[τ_idx, :id]
        )

        # Append dataframes
        DF.append!(df_par, df_s)
    end # if

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Add per strain fitness effect variables 
    # (for hierarchical models on genotypes)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    if (n_rep == 1) & (length(var_groups) == 7)
        # Find unique genotypes
        geno_unique = unique(genotypes)
        # Define genotype indexes
        geno_idx = indexin(genotypes, geno_unique)

        # Sample θ̲ variables
        θ_mat = hcat(
            [
                Random.rand(Distributions.Normal(x...), n_samples)
                for x in eachrow(
                    df_par[
                        df_par.vartype.=="bc_hyperfitness",
                        [:mean, :std]
                    ]
                )
            ]...
        )

        # Sample τ̲ variables
        τ_mat = exp.(
            hcat(
                [
                    Random.rand(Distributions.Normal(x...), n_samples)
                    for x in eachrow(
                        df_par[
                            df_par.vartype.=="bc_deviations",
                            [:mean, :std]
                        ]
                    )
                ]...
            )
        )

        # Sample θ̲̃ variables
        θ_tilde_mat = hcat(
            [
                Random.rand(Distributions.Normal(x...), n_samples)
                for x in eachrow(
                    df_par[
                        df_par.vartype.=="bc_noncenter",
                        [:mean, :std]
                    ]
                )
            ]...
        )


        # Compute individual strains fitness values
        s_mat = θ_mat[:, geno_idx] .+ (τ_mat .* θ_tilde_mat)

        # Compute mean and standard deviation and turn into dataframe
        df_s = DF.DataFrame(
            hcat(
                [StatsBase.median.(eachcol(s_mat)),
                    StatsBase.std.(eachcol(s_mat))]...
            ),
            ["mean", "std"]
        )

        # To add the rest of the columns, we will use the τ variables that have
        # the same length. Let's locate such variables
        τ_idx = occursin.("τ", string.(vars))

        # Add extra columns
        DF.insertcols!(
            df_s,
            :varname => replace.(df_par[τ_idx, :varname], "logτ" => "s"),
            :vartype .=> "bc_fitness",
            :rep => df_par[τ_idx, :rep],
            :env => df_par[τ_idx, :env],
            :id => df_par[τ_idx, :id]
        )

        # Append dataframes
        DF.append!(df_par, df_s)
    end # if

    return df_par
end # function