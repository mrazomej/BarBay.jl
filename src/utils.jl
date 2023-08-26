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

# Import package to handle MCMCChains
import MCMCChains

# Import library to load JLD2 files
import JLD2

##

@doc raw"""
    `concat_chains(chains, var_pattern, id_str)`

Function that concatenates multiple `MCMCChains.Chains` objects into a single
one. This function takes a vector of `MCMCChains.Chains` as inputs, extracts the
variables that match the patterns in the array `var_pattern`, and appends all
extracted variables into a single chain adding a pattern of the form
`[$(id_str)i]`, where `i` is the file number. For example, if two chains contain
a variable named `var`, the new chain returned by this function names them as
`var[f1]` and `var[f2]` if `id_str=f`.

NOTE: All chains must have the same number of samples to be concatenated.

# Arguments
- `chains::Vector{<:MCMCChains.Chains}`: Vector with the chains to be
  concatenated into a single chain.
- `var_pattern::Vector{Symbol}`: Patterns that variables must follow to be
  extracted from the chain. For example, if several variables are named
  `var[1]`, `var[2]`, etc, providing a pattern [`var`] extracts all of them,
  while providing `var[1]` extracts only the one that perfectly matches this
  pattern.

## Optional arguments
- `id_str::String=f`: String to be attached to the variable names that
  identifies the different chains being concatenated. For example, if 4 chains
  are being concatenated, each repeated variable will be named `var[$(id_str)i]`
  to distinguish each of them.

# Returns
- `MCMCChains.Chains`: Chain with the requested variables from multiple files
  concatenated into a single object.
"""
function concat_chains(
    chains::Vector{<:MCMCChains.Chains},
    var_pattern::Vector{Symbol};
    id_str::String="f"
)
    # Check that all chains have the same number of samples
    if length(unique([length(range(chn)) for chn in chains])) > 1
        error("All chains must have the same number of samples")
    end # if

    # Initialize array to save names
    varnames = []

    # Initialize array to save chains
    chain_samples = []

    # Loop through files
    for (i, chn) in enumerate(chains)
        # Extract names into single vector
        names_ = reduce(
            vcat, [MCMCChains.namesingroup(chn, x) for x in var_pattern]
        )
        # Convert names to string and append pattern
        push!(
            varnames, String.(names_) .* "[$(id_str)$(i)]"
        )

        # Extract variable chains into an array with the right format used to
        # build an MCMCChains.Chains object
        push!(
            chain_samples,
            cat(Array(chn[names_], append_chains=false)..., dims=3)
        )
    end # for

    # Return MCMCChains.Chains object with all files appended
    return MCMCChains.Chains(
        cat(chain_samples..., dims=2),
        reduce(vcat, varnames)
    )
end # function

@doc raw"""
    `jld2_concat_chains(dir, file_patern, chains, var_pattern, id_str)`

Convenient function that peforms the same concatenation as
`BayesFitness.utils.concat_chains` but giving a directory and a file pattern for
`jld2` files storing the chains. This function reads all files in `dir` that
have the pattern `file pattern`, obtaining a list of `MCMCChains.Chains` as
inputs. It then extracts the variables that match the patterns in the array
`var_pattern`, and appends all extracted variables into a single chain adding a
pattern of the form `[$(id_str)i]`, where `i` is the file number. For example,
if two chains contain a variable named `var`, the new chain returned by this
function names them as `var[f1]` and `var[f2]` if `id_str=f`.

NOTE: All chains must have the same number of samples to be concatenated.

# Arguments
- `dir::String`: Directory where file(s) with MCMC chains are stored.
- `file_pattern::String`: Pattern common among all files to process. NOTE: This is
  use in the `Glob.glob` command to locate all `jld2` files from which to
  extract the chains.
- `var_pattern::Vector{Symbol}`: Patterns that variables must follow to be
  extracted from the chain. For example, if several variables are named
  `var[1]`, `var[2]`, etc, providing a pattern [`var`] extracts all of them,
  while providing `var[1]` extracts only the one that perfectly matches this
  pattern.

## Optional arguments
- `id_str::String=f`: String to be attached to the variable names that
  identifies the different chains being concatenated. For example, if 4 chains
  are being concatenated, each repeated variable will be named `var[$(id_str)i]`
  to distinguish each of them.

# Returns
- `MCMCChains.Chains`: Chain with the requested variables from multiple files
  concatenated into a single object.
- `chainname::String="chain"`: String defining the dictionary key on the `.jld2`
file to extract the MCMC chain.
"""
function jld2_concat_chains(
    dir::String,
    file_pattern::String,
    var_pattern::Vector{Symbol};
    id_str::String="f",
    chainname::String="chain"
)
    # List files
    files = sort(Glob.glob("$(dir)/*$(file_pattern)*.jld2"))

    # Extract variable chains
    chains = [JLD2.load(f)[chainname] for f in files]

    return concat_chains(chains, var_pattern; id_str=id_str)
end # function

@doc raw"""
    group_split(data, n_groups, groupby_col, count_col; sort_function)

Function to split a set of labels into `n_group` subgroups.

# Arguments
- `data::DF.AbstractDataFrame`: Data to be split into groups. This function
    expects a tidy dataframe with at least two columns:
    - `groupby_col`: Column to group entries by. This is commonly the barcode ID
      that distinguishes different strains.
    - `sort_col`: Column with values used to sort the data entries.
- `n_groups::Int`: Number of groups in which to split the data
- `groupby_col::Symbol`: Name of column used to group the unique entries in
  dataset. This is commonly the barcode ID that distinguishes different strains.
- `sort_col::Symbol`: Name of column with quantity used to sort the entries in
  the dataset. This is commonly the number of barcode counts or frequency.

## Optional Keyword Arguments
- `sort_function::Function=x -> StatsBase.mean(log.(x .+ 1))`: Functio to use on
  the `group-apply-combine` routine. The default function computes the mean in
  log-scale, adding a 1 to avoid computing `log(0)`.

# Returns
- `groups::Vector{Vector{typeof(data[groupby_col][1])}}`: Vectors containing the
  different groups in which to split the dataset.
"""
function group_split(
    data::DF.AbstractDataFrame,
    n_groups::Int,
    groupby_col::Symbol,
    sort_col::Symbol;
    sort_function::Function=x -> StatsBase.mean(log.(x .+ 1))
)
    # Split-apply-combine `sort_function` to define grouping criteria
    data_combine = DF.combine(
        DF.groupby(data, groupby_col),
        sort_col => sort_function
    )
    # Rename columns
    DF.rename!(data_combine, [groupby_col, sort_col])

    # Sort data by sort_col 
    DF.sort!(data_combine, sort_col, rev=true)

    # Initialize vector to save groups
    groups = Vector{
        Vector{typeof(data_combine[:, groupby_col][1])}
    }(undef, n_groups)

    # Loop through groups
    for i = 1:n_groups
        # Select elements for group
        groups[i] = data_combine[:, groupby_col][i:n_groups:end]
    end # for

    return groups
end # function

@doc raw"""
    data2arrays(data; kwargs)

Function to preprocess the tidy dataframe `data` into the corresponding inputs
for the models in the `model` submodule.

# Arguments
- `data::DataFrames.AbstractDataFrame`: Tidy dataframe with the data to be used
  for sampling the model posterior distribution. 

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
- `rm_T0::Bool=false`: Optional argument to remove the first time point from the
  inference. Commonly, the data from this first time point is of much lower
  quality. Therefore, removing this first time point might result in a better
  inference.
- `verbose::Bool=true`: Boolean indicating if printing statements should be
  made.

# Returns
if `typeof(rep_col) <: Nothing` Dictionary with the following entries: -
    `bc_count::Matrix`: `T × B` matrix with all barcodes read counts. -
    `bc_total::Vector`: `T`-dimensional vector with the total number of reads
    per time point. - `n_neutral::Int`: Number of neutral lineages. -
    `n_mut::Int`: Number of mutant lineages. - `mut_keys`: List of mutant names
    in the order used to build `R̲̲`.

elseif `typeof(rep_col) <: Symbol` Dictionary with the following entries: -
        `bc_count::Array`: `T × B × R` array with all barcodes read counts. -
        `bc_total::Matrix`: `T × R` matrix with the total number of reads per
        time point per repeat. - `n_neutral::Int`: Number of neutral lineages. -
        `n_mut::Int`: Number of mutant lineages. - `mut_keys`: List of mutant
        names in the order used to build `R̲̲`.
"""
function data2arrays(
    data::DF.AbstractDataFrame;
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rep_col::Union{Nothing,Symbol}=nothing,
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
        mut_ids = first.(values.(keys(data_group)))

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

        ### --------------- When repeats are given --------------- ### 
    elseif typeof(rep_col) <: Symbol
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
        mut_ids = sort(unique(data_mut[:, id_col]))
        # Extract unique reps
        reps = sort(unique(data_mut[:, rep_col]))

        # Initialize array to save counts for each mutant at time t
        R̲̲⁽ᵐ⁾ = Array{Int64,3}(
            undef, length(timepoints), length(mut_ids), length(reps)
        )

        # Loop through each unique id
        for (j, id) in enumerate(mut_ids)
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

    end # if

    return Dict(
        :bc_count => R̲̲,
        :bc_total => n̲ₜ,
        :n_neutral => size(R̲̲⁽ⁿ⁾, 2),
        :n_mut => size(R̲̲⁽ᵐ⁾, 2),
        :mut_ids => mut_ids
    )
end # function

@doc raw"""
advi2df(dist, vars, mut_ids; n_rep=1, envs=[1], n_samples=10_000)

Convert the output of automatic differentiation variational inference (ADVI) to
a tidy dataframe.

# Arguments
- `dist::Distributions.Sampleable`: The ADVI posterior sampleable distribution
  object.
- `vars::Vector{<:Any}`: Vector of variable/parameter names from the ADVI run. 
- `mut_ids::Vector{<:Any}`: Vector of mutant strain IDs.

## Oprtional Keyword Arguments
- `n_rep::Int=1`: Number of experimental replicates. Default is 1. 
- `envs::Vector{<:Any}=[1]`: Vector of environment ids for each timepoint.
  Default is a single environment [1].
- `n_samples::Int=10_000`: Number of posterior samples to draw used for
  hierarchical models. Default is 10,000.

# Returns
- `df::DataFrames.DataFrame`: DataFrame containing summary statistics of
posterior samples for each parameter. Columns include:
    - `mean, std`: posterior mean and standard deviation for each of the
      variables.
    - `varname`: parameter name from the ADVI posterior distribution.
    - `vartype`: Description of the type of parameter. The types are:
        - `pop_mean`: Population mean fitness value `s̲ₜ`.
        - `pop_error`: (Nuisance parameter) Log of standard deviation in the
          likelihood function for the neutral lineages.
        - `mut_fitness`: Mutant relative fitness `s⁽ᵐ⁾`.
        - `mut_hyperfitness`: For hierarchical models, mutant hyper parameter
          that connects the fitness over multiple experimental replicates
          `θ⁽ᵐ⁾`.
        - `mut_noncenter`: (Nuisance parameter) For hierarchical models,
          non-centered samples used to connect the experimental replicates to
          the hyperparameter `θ̃⁽ᵐ⁾`.
        - `mut_deviations`: (Nuisance parameter) For hierarchicaal models,
          samples that define the log of the deviation from the hyper parameter
          fitness value `logτ⁽ᵐ⁾`.
        - `mut_error`: (Nuisance parameter) Log of standard deviation in the
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
- Useful for post-processing ADVI results for further analysis and plotting.
"""
function advi2df(
    dist::Distributions.Sampleable,
    vars::Vector{<:Any},
    mut_ids::Vector{<:Any};
    n_rep::Int=1,
    envs::Vector{<:Any}=[1],
    n_samples::Int=10_000
)
    # Extract parameters and convert to dataframe
    df_par = DF.DataFrame(
        hcat(Distributions.params(dist)...),
        ["mean", "std"]
    )
    # Add variable name. Notice that we use deepcopy to avoid the modification
    # of the original variable
    df_par[!, :varname] = deepcopy(vars)

    # Locate variable groups by identifying the first variable of each group
    var_groups = replace.(vars[occursin.("[1]", string.(vars))], "[1]" => "")

    # Count how many variables exist per group
    var_count = [sum(occursin.(vg, string.(vars))) for vg in var_groups]

    # Define number of time points based on number of population mean fitness
    # variables and the number of repeats
    n_time = sum(occursin.(first(var_groups), string.(vars))) ÷ n_rep + 1

    # Define number of mutants
    n_mut = length(mut_ids)

    # Define number of neutral lineages from the total count of frequency
    # variables
    n_neutral = (sum(occursin.("Λ", vars)) - (n_mut * n_time * n_rep)) ÷
                (n_time * n_rep)
    # Assign neutral IDs. This will be used for the ID of frequency variables
    neutral_ids = ["neutral$(lpad(x, 3, "0"))" for x = 1:n_neutral]

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Add replicate number information
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # 1. One repeat should have 5 var_groups. If larger, it is an error
    if (n_rep == 1) .& (length(var_groups) == 7)
        # Report that this must be a hierarchical model
        error("This seems like a hierarchical model with multiple repeats")

        # 2. One repeat and 5 var_groups: Standard single-dataset model
    elseif (n_rep == 1) .& (length(var_groups) == 5)
        # Define variable types
        vtypes = ["pop_mean", "pop_error", "mut_fitness", "mut_error", "freq"]
        # Repeat variable type var_count times and add it to dataframe
        df_par[!, :vartype] = vcat(
            [repeat([vtypes[i]], var_count[i]) for i in eachindex(var_count)]...
        )

        # Add repeat information. NOTE: This is to have a consistent dataframe
        # for all possible models
        df_par[!, :rep] .= "R1"

        # 3. More than one repeat and 7 var_groups: Hierarchical model for
        #    experimental replicates
    elseif (n_rep > 1) .& (length(var_groups) == 7)
        # Define variable types
        vtypes = ["pop_mean", "pop_error", "mut_hyperfitness", "mut_noncenter",
            "mut_deviations", "mut_error", "freq"]
        # Repeat variable type var_count times and add it to dataframe
        df_par[!, :vartype] = vcat(
            [repeat([vtypes[i]], var_count[i]) for i in eachindex(var_count)]...
        )
        # Initialize array to define replicate information
        rep = Vector{String}(undef, length(vars))
        # Initialize array to define time information
        time = Vector{Int64}(undef, length(vars))
        # Loop through variables
        for (i, vt) in enumerate(vtypes)
            # Locate variables
            var_idx = df_par.vartype .== vt
            # Check if there's no hyper parameter and no freq
            if !occursin("hyper", vt)
                # Add corresponding replicate information
                rep[var_idx] = vcat(
                    [repeat(["R$j"], var_count[i] ÷ n_rep) for j = 1:n_rep]...
                )
            else
                rep[var_idx] .= "N/A"
            end # if
        end # for
        # Add repeat information to dataframe
        df_par[!, :rep] = rep
    end # if

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Add environment information
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # 1. Single environment case
    if length(envs) == 1
        # Add environment information. NOTE: This is to have a consistent
        # dataframe structure
        df_par[!, :env] .= first(envs)
    elseif length(envs) > 1
        # Add environment variable (to be modofied later)
        df_par[!, :env] .= first(envs)
        # Loop through variables
        for (i, var) in enumerate(unique(df_par.vartype))
            # Locate variables
            var_idx = df_par.vartype .== var
            # Check cases
            # 1. population mean fitness-related variables
            if occursin("pop", var)
                df_par[var_idx, :env] = repeat(envs[2:end], n_rep)
                # 2. frequency-related variables
            elseif occursin("freq", var)
                df_par[var_idx, :env] = repeat(
                    envs, (n_mut + n_neutral) * n_rep
                )
                # 3. mutant fitness-related variables
            else
                df_par[var_idx, :env] = repeat(
                    unique(envs), sum(var_idx) ÷ length(unique(envs))
                )
            end # if
        end # for
    end # if

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Add mutant id information
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Add mutant id column. To be modified
    df_par[:, :id] .= "N/A"

    # Loop through variables
    for (i, var) in enumerate(unique(df_par.vartype))
        # Locate variables
        var_idx = df_par.vartype .== var
        # Check cases
        # 1. population mean fitness-related variables
        if occursin("pop", var)
            continue
            # 2. frequency-related variables
        elseif occursin("freq", var)
            df_par[var_idx, :id] = repeat(
                vcat(
                    [repeat([x], n_time)
                     for x in [string.(neutral_ids); string.(mut_ids)]]...
                ),
                n_rep
            )
            # 3. mutant hyperfitness-related variables
        elseif occursin("hyper", var)
            df_par[var_idx, :id] = vcat(
                [repeat([x], length(unique(envs))) for x in mut_ids]...
            )
            # 4. mutant fitness-related variables
        else
            df_par[var_idx, :id] = repeat(
                vcat(
                    [repeat([x], length(unique(envs))) for x in mut_ids]...
                ),
                n_rep
            )
        end # if
    end # for

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Add per-replicate fitness effect variables (for hierarchical models)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    if (n_rep > 1) .& (length(var_groups) == 7)
        # Sample θ̲ variables
        θ_mat = hcat(
            [
                Random.rand(Distributions.Normal(x...), n_samples)
                for x in eachrow(
                    df_par[
                        df_par.vartype.=="mut_hyperfitness",
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
                            df_par.vartype.=="mut_deviations",
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
                        df_par.vartype.=="mut_noncenter",
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
            :vartype .=> "mut_fitness",
            :rep => df_par[τ_idx, :rep],
            :env => df_par[τ_idx, :env],
            :id => df_par[τ_idx, :id]
        )

        # Append dataframes
        DF.append!(df_par, df_s)
    end # if

    return df_par
end # function