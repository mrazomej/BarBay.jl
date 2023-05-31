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