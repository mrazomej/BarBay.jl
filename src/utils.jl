##
# Import package to handle dataframes
import DataFrames as DF
import CSV

# Import function to list files
import Glob

# Import package to handle MCMCChains
import MCMCChains

# Import library to load JLD2 files
import JLD2

##

@doc raw"""
    var_jld2_to_df(dir, pattern, varname, chainname)

Function that takes `.jld2` files in `dir` with `pattern` and extracts a single
variable into a dataframe. This function is useful to extract, for example, the
chains for each inference of the population mean fitness over multiple time
points.

NOTE: All chains from which samples will be extracted must have the same number
of samples.

# Arguments
- `dir::String`: Directory where file(s) with MCMC chains are stored.
- `pattern::String`: Pattern common among all files to process. NOTE: This is
  use in the `Glob.glob` command to locate all `jld2` files from which to
  extract the chains.
- `varname::Symbol`: Name of variable in chain object to extract.

## Optional Arguments
- `chainname::String="chain"`: String defining the dictionary key on the `.jld2`
  file to extract the MCMC chain.

# Returns
- `DataFrames.DataFrame`: DataFrame containing all variable samples⸺multiple
  chains are collapsed into a single column⸺one variable per column.
"""
function var_jld2_to_df(
    dir::String, pattern::String, varname::Symbol, chainname::String="chain",
)
    # List files
    files = sort(Glob.glob("$(dir)/*$(pattern)*.jld2"))

    # Extract variable chains
    chains = [vec(Matrix(JLD2.load(f)[chainname][varname])) for f in files]

    # Check that all chains have the same number of samples
    if length(unique(length.(chains))) > 1
        error("All chains must have the same number of samples")
    end # if

    # concatenate chains into DataFrame
    return DF.DataFrame(
        hcat(chains...),
        ["$(varname)_$(i)" for i in eachindex(files)]
    )
end # function

@doc raw"""
    var_jld2_to_df(files, varname, chainname)

Function that takes `.jld2` files in `dir` with `pattern` and extracts a single
variable into a dataframe. This function is useful to extract, for example, the
chains for each inference of the population mean fitness over multiple time
points.

NOTE: All chains from which samples will be extracted must have the same number
of samples.

# Arguments
- `files::Vectlr{String}`: List of `jld2` files form which to extract the MCMC
  chain.
- `varname::Symbol`: Name of variable in chain object to extract.

## Optional Arguments
- `chainname::String="chain"`: String defining the dictionary key on the `.jld2`
  file to extract the MCMC chain.

# Returns
- `DataFrames.DataFrame`: DataFrame containing all variable samples⸺multiple
  chains are collapsed into a single column⸺one variable per column.
"""
function var_jld2_to_df(
    files::Vector{String}, varname::Symbol, chainname::String="chain",
)
    # Extract variable chains
    chains = [vec(Matrix(JLD2.load(f)[chainname][varname])) for f in files]

    # Check that all chains have the same number of samples
    if length(unique(length.(chains))) > 1
        error("All chains must have the same number of samples")
    end # if

    # concatenate chains into DataFrame
    return DF.DataFrame(
        hcat(chains...),
        ["$(varname)_$(i)" for i in eachindex(files)]
    )
end # function

@doc raw"""
    var_jld2_concat(dir, pattern, varname)

Function that takes `.jld2` files in `dir` with `pattern` and extracts a single
variable to concatentate into a single chain object. This function is useful to
extract variables with the same name on multiple `.jld2` files and compile them
into a single `MCMCChains.Chains` object.

NOTE: All chains from which samples will be extracted must have the same number
of samples.

# Arguments
- `dir::String`: Directory where file(s) with MCMC chains are stored.
- `pattern::String`: Pattern common among all files to process. NOTE: This is
  use in the `Glob.glob` command to locate all `jld2` files from which to
  extract the chains.
- `varname::Symbol`: Name of variable in chain object to extract.

## Optional Arguments
- `chainname::String="chain"`: String defining the dictionary key on the `.jld2`
  file to extract the MCMC chain.

# Returns
- `MCMCChains.Chains`: Chain containing all variable samples.
"""
function var_jld2_concat(
    dir::String, pattern::String, varname::Symbol, chainname::String="chain",
)
    # List files
    files = sort(Glob.glob("$(dir)/*$(pattern)*.jld2"))

    # Extract variable chains
    chains = [Matrix(JLD2.load(f)[chainname][varname]) for f in files]

    # Check that all chains have the same number of samples
    if length(unique(length.(chains))) > 1
        error("All chains must have the same number of samples")
    end # if

    # concatenate into single data chain. NOTE: The chains axes have to be
    # rearranged to make it match the dimensions expected by MCMCChains.Chains
    return MCMCChains.Chains(
        permutedims(cat(chains..., dims=3), [1, 3, 2]),
        [Symbol(String(varname) * "[$x]") for x = 1:length(chains)]
    )
end # function

@doc raw"""
    chain_to_df(chain, varnames)

Function that extracts the variables in `varnames` from the `chain` object and
concatenates them into a `DataFrame`. Multiple chains are piled on top of each
other.

# Arguments
- `chain::MCMCChains.Chains`: Chain with variables to be converted to dataframe

# Optional arguments
- `varnames::Union{Vector{Symbol}, Nothing}=nothing`: Names of the variables in
  the chain to be extracted. These can be either the full variable name or
  patterns to extract multi-dimensional variables. If `nothing` is given, all
  parameters are extracted into a dataframe.

# Returns
`DataFrames.DataFrame`: Data frame where each column represents all the MCMC
samples for a single variable (multiple chains are stack on top of each other)
"""
function chain_to_df(
    chain::MCMCChains.Chains, varnames::Union{Vector{Symbol},Nothing}=nothing
)
    # Extract parameter names
    chain_param = MCMCChains.names(chain, :parameters)

    # Check if varnames were given
    if typeof(varnames) == Nothing
        # Extract and return selected variables in dataframe
        return DF.DataFrame(
            [vec(Matrix(chain[var])) for var in chain_param],
            chain_param
        )
    else
        # Convert chain_param into string
        chain_str = String.(chain_param)

        # Initialize list of variables to extract
        vars = Symbol[]
        # Loop through varnames
        for v in varnames
            # Search and append parameters that match pattern
            push!(
                vars,
                chain_param[occursin.(String(v), chain_str)]...
            )
        end # for

        # Extract and return selected variables in dataframe
        return DF.DataFrame(
            [vec(Matrix(chain[var])) for var in unique(vars)],
            vars
        )
    end # if
end # function