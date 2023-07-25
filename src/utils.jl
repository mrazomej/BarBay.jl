##
# Import package to handle dataframes
import DataFrames as DF
import CSV

# Import basic math
import StatsBase

# Import function to list files
import Glob

# Import package to handle MCMCChains
import MCMCChains

# Import library to load JLD2 files
import JLD2

# Import functions from other modules
import BayesFitness.stats: naive_fitness
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
    group_split(data, n_groups; kwargs)

Function to split a set of labels into `n_group` subgroups sorted by a naive
estimate of the fitness value.

# Arguments
- `data::DataFrames.AbstractDataFrame`: **Tidy dataframe** with the data to be
used to infer the fitness values on mutants. The `DataFrame` must contain at
least the following columns:
    - `id_col`: Column identifying the ID of the barcode. This can the barcode
    sequence, for example.
    - `time_col`: Column defining the measurement time point.
    - `count_col`: Column with the raw barcode count.
    - `neutral_col`: Column indicating whether the barcode is from a neutral
    lineage or not.
- `n_groups::Int`: Number of groups in which to split the data.

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
- `pseudo_count::Int=1`: Pseudo count number to add to all counts. This is
  useful to avoid divisions by zero.

# Returns
- `groups::Vector{Vector{typeof(data[groupby_col][1])}}`: Vectors containing the
different groups in which to split the dataset.
"""
function group_split_naive_fitness(
    data::DF.AbstractDataFrame,
    n_groups::Int;
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rm_T0::Bool=false,
    pseudo_count::Int=1
)
    # Compute naive fitness estimate
    data_fitness = naive_fitness(
        data;
        id_col=id_col,
        time_col=time_col,
        count_col=count_col,
        neutral_col=neutral_col,
        rm_T0=rm_T0,
        pseudo_count=pseudo_count
    )

    # Sort data by sort_col 
    DF.sort!(data_fitness, :fitness, rev=true)

    # Initialize vector to save groups
    groups = Vector{
        Vector{typeof(data_fitness[:, id_col][1])}
    }(undef, n_groups)

    # Loop through groups
    for i = 1:n_groups
        # Select elements for group
        groups[i] = data_fitness[:, id_col][i:n_groups:end]
    end # for

    return groups
end # function

@doc raw"""
    df2mats(data; kwargs)

Function that returns the matarices `R̲̲⁽ⁿ⁾`,  `R̲̲⁽ᵐ⁾`, and  `R̲̲` usually
taken by the functions in the `model.jl` module. This function is useful to
prototype new models before properly implementing them. The final user most
likely won't need to ever call this function.

# Arguments
- `data::DataFrames.AbstractDataFrame`: **Tidy dataframe** with the data to be
used to sample from the population mean fitness posterior distribution.

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

# Returns
- `R̲̲⁽ⁿ⁾::Matrix`: T × N matrix with the neutral barcodes read counts.
- `R̲̲⁽ᵐ⁾::Matrix`: T × M matrix with the mutant barcodes read counts.
- `R̲̲::Matrix`: T × B matrix with all barcodes read counts.
- `n̲ₜ::Vector`: T-dimensional vector with the total number of reads per time
  point.
"""
function data2mats(
    data::DF.AbstractDataFrame;
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    count_col::Symbol=:count,
    neutral_col::Symbol=:neutral,
    rm_T0::Bool=false
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

    return R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾, R̲̲, n̲ₜ
end # function