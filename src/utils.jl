##

# Import basic libraries
import Logging

# Import package to handle dataframes
import DataFrames as DF
import CSV

# Import basic math
import StatsBase
import Distributions
import Random

## =============================================================================


@doc raw"""
    DataArrays

Structure to store processed barcode count data and associated metadata for
fitness inference.

# Fields
- `bc_count`: Raw barcode counts. Can be:
    - `Matrix{Int64}` for single replicate/environment data
    - `Array{Int64,3}` for multiple replicates/environments
    - `Vector{Matrix{Int64}}` for varying timepoints across replicates
- `bc_total`: Total population size per timepoint. Format matches `bc_count`:
    - `Vector{Int64}` for single replicate
    - `Matrix{Int64}` for multiple replicates
    - `Vector{Vector{Int64}}` for varying timepoints
- `n_neutral`: Number of neutral barcodes
- `n_bc`: Total number of barcodes (neutral + mutant)
- `bc_ids`: Vector of mutant barcode identifiers
- `neutral_ids`: Vector of neutral barcode identifiers
- `envs`: Environmental conditions. Can be:
    - `String` for single environment
    - `Vector{<:Any}` for multiple environments
    - `Vector{Vector{<:Any}}` for nested environment conditions
- `n_env`: Number of unique environments
- `n_rep`: Number of experimental replicates
- `n_time`: Number of timepoints. Can be:
    - `Int` when consistent across replicates
    - `Vector{Int}` when varying across replicates
- `genotypes`: Genotype identifiers. Can be:
    - `String` for single genotype
    - `Vector{<:Any}` for multiple genotypes
- `n_geno`: Number of unique genotypes
"""
struct DataArrays
    bc_count::Union{Matrix{Int64},Array{Int64,3},Vector{Matrix{Int64}}}
    bc_total::Union{Vector{Int64},Matrix{Int64},Vector{Vector{Int64}}}
    n_neutral::Int
    n_bc::Int
    bc_ids::Vector
    neutral_ids::Vector
    envs::Union{String,Vector{<:Any},Vector{Vector{<:Any}}}
    n_env::Int
    n_rep::Int
    n_time::Union{Int,Vector{Int}}
    genotypes::Union{String,Vector{<:Any}}
    n_geno::Int
end

## =============================================================================

"""
    _extract_timepoints(
        data::DF.AbstractDataFrame, time_col::Symbol
    )

Internal function that extracts unique time points from a given DataFrame `data`
based on the specified `time_col`.

# Arguments
- `data::DF.AbstractDataFrame`: The input DataFrame from which to extract time
  points.
- `time_col::Symbol`: The column in `data` that contains the time points.

# Returns
- `timepoints`: A sorted array of unique time points extracted from `data`.
"""
function _extract_timepoints(
    data::DF.AbstractDataFrame,
    time_col::Symbol
)
    # Extract unique time points
    timepoints = sort(unique(data[:, time_col]))

    return timepoints
end # function

# ------------------------------------------------------------------------------
# Helper functions for processing barcode data
# ------------------------------------------------------------------------------

"""
Process neutral barcode data from a single replicate experiment.
"""
function _process_neutral_barcodes_single(
    data::DF.AbstractDataFrame,
    neutral_col::Symbol,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    timepoints::Vector
)::Tuple{Matrix{Int64},Vector}
    # Group data by unique neutral barcode
    data_group = DF.groupby(data[data[:, neutral_col], :], id_col)

    # Extract group keys for neutral barcodes
    neutral_ids = first.(values.(keys(data_group)))

    # Check that all barcodes were measured at all points
    if any([size(d, 1) for d in data_group] .!= length(timepoints))
        error("""
            Not all neutral barcodes have reported counts in all time points.
            Please check your data to ensure:
                - No missing timepoints for any barcode
                - Consistent time series length across barcodes
            Current timepoints: $(join(timepoints, ", "))
            """)
    end

    # Initialize array to save counts for each neutral at time t
    R̲̲⁽ⁿ⁾ = Matrix{Int64}(undef, length(timepoints), length(data_group))

    # Loop through each unique barcode
    for (i, d) in enumerate(data_group)
        # Sort data by timepoint
        DF.sort!(d, time_col)
        # Extract data
        R̲̲⁽ⁿ⁾[:, i] = d[:, count_col]
    end

    return R̲̲⁽ⁿ⁾, neutral_ids
end

"""
Process mutant barcode data from a single replicate experiment.
"""
function _process_mutant_barcodes_single(
    data::DF.AbstractDataFrame,
    neutral_col::Symbol,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    timepoints::Vector
)::Tuple{Matrix{Int64},Vector}
    # Group data by unique mutant barcode
    data_group = DF.groupby(data[.!data[:, neutral_col], :], id_col)

    # Extract group keys for mutant barcodes
    bc_ids = first.(values.(keys(data_group)))

    # Check that all barcodes were measured at all points
    if any([size(d, 1) for d in data_group] .!= length(timepoints))
        error("""
            Not all mutant barcodes have reported counts in all time points.
            Please check your data to ensure:
                - No missing timepoints for any barcode
                - Consistent time series length across barcodes
            Current timepoints: $(join(timepoints, ", "))
            """)
    end

    # Initialize array to save counts for each mutant at time t
    R̲̲⁽ᵐ⁾ = Matrix{Int64}(undef, length(timepoints), length(data_group))

    # Loop through each unique barcode
    for (i, d) in enumerate(data_group)
        # Sort data by timepoint
        DF.sort!(d, time_col)
        # Extract data
        R̲̲⁽ᵐ⁾[:, i] = d[:, count_col]
    end

    return R̲̲⁽ᵐ⁾, bc_ids
end

# ------------------------------------------------------------------------------
# Helper functions for processing barcode data from multiple replicates
# ------------------------------------------------------------------------------

"""
Process neutral barcode data from multiple replicate experiments.
"""
function _process_neutral_barcodes_multi(
    data::DF.AbstractDataFrame,
    neutral_col::Symbol,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    rep_col::Symbol,
    timepoints::Vector
)::Tuple{Array{Int64,3},Vector}
    # Extract neutral data
    data_neutral = @view data[data[:, neutral_col], :]
    # Extract unique neutral IDs
    neutral_ids = unique(data_neutral[:, id_col])
    # Extract unique reps
    reps = unique(data_neutral[:, rep_col])

    # Initialize array to save counts for each neutral at time t
    R̲̲⁽ⁿ⁾ = Array{Int64,3}(
        undef, length(timepoints), length(neutral_ids), length(reps)
    )

    # Loop through each unique neutral ID
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
        end
    end

    return R̲̲⁽ⁿ⁾, neutral_ids
end

# ------------------------------------------------------------------------------

"""
Process mutant barcode data from multiple replicate experiments.
"""
function _process_mutant_barcodes_multi(
    data::DF.AbstractDataFrame,
    neutral_col::Symbol,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    rep_col::Symbol,
    timepoints::Vector
)::Tuple{Array{Int64,3},Vector}
    # Extract mutant data
    data_mut = @view data[.!data[:, neutral_col], :]
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
        end
    end

    return R̲̲⁽ᵐ⁾, bc_ids
end

# ------------------------------------------------------------------------------
# Helper functions for processing barcode data from multiple replicates with
# different timepoints
# ------------------------------------------------------------------------------

"""
Process neutral barcode data from multiple replicates with different timepoints.
"""
function _process_neutral_barcodes_multi_varying(
    data_rep::DF.GroupedDataFrame,
    neutral_col::Symbol,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    n_rep_time::Vector{Int}
)::Tuple{Vector{Matrix{Int64}},Vector}
    # Initialize Vector to save matrix for each replicate
    R̲̲⁽ⁿ⁾ = Vector{Matrix{Int64}}(undef, length(n_rep_time))
    neutral_ids = Vector{Any}(undef, 0)

    # Loop through replicates
    for (rep, d_rep) in enumerate(data_rep)
        # Group data by unique neutral barcode
        data_group = DF.groupby(d_rep[d_rep[:, neutral_col], :], id_col)

        # Extract group keys for first replicate only
        if rep == 1
            neutral_ids = first.(values.(keys(data_group)))
        end

        # Check that all barcodes were measured at all points
        if any([size(d, 1) for d in data_group] .!= n_rep_time[rep])
            error("""
                Not all neutral barcodes have reported counts in all time points
                for replicate $rep.
                Please check your data to ensure:
                    - No missing timepoints for any barcode
                    - Consistent time series length across barcodes
                Expected timepoints: $(n_rep_time[rep])
                """)
        end

        # Initialize array to save counts for each mutant at time t
        R̲̲⁽ⁿ⁾[rep] = Matrix{Int64}(
            undef, n_rep_time[rep], length(data_group)
        )

        # Loop through each unique barcode
        for (i, d) in enumerate(data_group)
            # Sort data by timepoint
            DF.sort!(d, time_col)
            # Extract data
            R̲̲⁽ⁿ⁾[rep][:, i] = d[:, count_col]
        end
    end

    return R̲̲⁽ⁿ⁾, neutral_ids
end

# ------------------------------------------------------------------------------

"""
Process mutant barcode data from multiple replicates with different timepoints.
"""
function _process_mutant_barcodes_multi_varying(
    data_rep::DF.GroupedDataFrame,
    neutral_col::Symbol,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    n_rep_time::Vector{Int}
)::Tuple{Vector{Matrix{Int64}},Vector}
    # Initialize Vector to save matrix for each replicate
    R̲̲⁽ᵐ⁾ = Vector{Matrix{Int64}}(undef, length(n_rep_time))
    bc_ids = Vector{Any}(undef, 0)

    # Loop through replicates
    for (rep, d_rep) in enumerate(data_rep)
        # Group data by unique mutant barcode
        data_group = DF.groupby(d_rep[.!d_rep[:, neutral_col], :], id_col)

        # Extract group keys for first replicate only
        if rep == 1
            bc_ids = first.(values.(keys(data_group)))
        end

        # Check that all barcodes were measured at all points
        if any([size(d, 1) for d in data_group] .!= n_rep_time[rep])
            error("""
                Not all mutant barcodes have reported counts in all time points
                for replicate $rep.
                Please check your data to ensure:
                    - No missing timepoints for any barcode
                    - Consistent time series length across barcodes
                Expected timepoints: $(n_rep_time[rep])
                """)
        end

        # Initialize array to save counts for each mutant at time t
        R̲̲⁽ᵐ⁾[rep] = Matrix{Int64}(
            undef, n_rep_time[rep], length(data_group)
        )

        # Loop through each unique barcode
        for (i, d) in enumerate(data_group)
            # Sort data by timepoint
            DF.sort!(d, time_col)
            # Extract data
            R̲̲⁽ᵐ⁾[rep][:, i] = d[:, count_col]
        end
    end

    return R̲̲⁽ᵐ⁾, bc_ids
end

# ============================================================================
# Base case: Single condition, with neutrals
# ============================================================================

@doc raw"""
    _extract_R(
        data::DF.AbstractDataFrame, id_col::Symbol, time_col::Symbol,
        count_col::Symbol, neutral_col::Symbol, rep_col::Nothing, 
        env_col::Nothing, genotype_col::Nothing
    )

Base case extractor for single condition experiment with neutral lineages.
This handles the simplest case where:
- There is a single experimental condition
- Has neutral lineages (neutral_col is provided)
- No replicates (rep_col is Nothing)
- No environment variations (env_col is Nothing)
- No genotype groupings (genotype_col is Nothing)

# Returns
- `DataArrays`: Struct containing processed barcode data including:
    - Single matrix for counts
    - Single vector for totals
    - Basic metadata (no replicates, environments, or genotypes)
"""
function _extract_R(
    data::DF.AbstractDataFrame,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    neutral_col::Symbol,
    rep_col::Nothing,
    env_col::Nothing,
    genotype_col::Nothing
)::DataArrays
    # Extract unique time points
    timepoints = _extract_timepoints(data, time_col)

    # Process neutral and mutant barcodes
    R̲̲⁽ⁿ⁾, neutral_ids = _process_neutral_barcodes_single(
        data, neutral_col, id_col, time_col, count_col, timepoints
    )
    R̲̲⁽ᵐ⁾, bc_ids = _process_mutant_barcodes_single(
        data, neutral_col, id_col, time_col, count_col, timepoints
    )

    # Concatenate matrices and compute totals
    R̲̲ = hcat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾)
    n̲ₜ = vec(sum(R̲̲, dims=2))

    return DataArrays(
        R̲̲,                    # bc_count
        n̲ₜ,                   # bc_total
        size(R̲̲⁽ⁿ⁾, 2),       # n_neutral
        size(R̲̲⁽ᵐ⁾, 2),       # n_bc
        bc_ids,               # bc_ids
        neutral_ids,          # neutral_ids
        "env1",              # envs
        1,                   # n_env
        1,                   # n_rep
        length(timepoints),  # n_time
        "N/A",              # genotypes
        0                    # n_geno
    )
end

# ============================================================================
# Case: Single condition, no neutrals
# ============================================================================

@doc raw"""
    _extract_R(
        data::DF.AbstractDataFrame, id_col::Symbol, time_col::Symbol,
        count_col::Symbol, neutral_col::Nothing, rep_col::Nothing, 
        env_col::Nothing, genotype_col::Nothing
    )

Extractor for single condition experiment without neutral lineages.
This handles the case where:
- There is a single experimental condition
- No neutral lineages (neutral_col is Nothing)
- No replicates (rep_col is Nothing)
- No environment variations (env_col is Nothing)
- No genotype groupings (genotype_col is Nothing)

# Returns
- `DataArrays`: Struct containing processed barcode data including:
    - Single matrix for counts
    - Single vector for totals
    - Basic metadata (no neutrals, replicates, environments, or genotypes)
"""
function _extract_R(
    data::DF.AbstractDataFrame,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    neutral_col::Nothing,
    rep_col::Nothing,
    env_col::Nothing,
    genotype_col::Nothing
)::DataArrays
    # Extract unique time points
    timepoints = _extract_timepoints(data, time_col)

    # Process all barcodes as mutants
    R̲̲⁽ᵐ⁾, bc_ids = _process_mutant_barcodes_single(
        data, nothing, id_col, time_col, count_col, timepoints
    )

    # Calculate totals
    n̲ₜ = vec(sum(R̲̲⁽ᵐ⁾, dims=2))

    return DataArrays(
        R̲̲⁽ᵐ⁾,                # bc_count
        n̲ₜ,                   # bc_total
        0,                    # n_neutral
        size(R̲̲⁽ᵐ⁾, 2),       # n_bc
        bc_ids,               # bc_ids
        Vector{Any}(),        # neutral_ids
        "env1",              # envs
        1,                   # n_env
        1,                   # n_rep
        length(timepoints),  # n_time
        "N/A",              # genotypes
        0                    # n_geno
    )
end

# ============================================================================
# Case: Multiple replicates
# ============================================================================

@doc raw"""
    _extract_R(
        data::DF.AbstractDataFrame, id_col::Symbol, time_col::Symbol,
        count_col::Symbol, neutral_col::Symbol, rep_col::Symbol, 
        env_col::Nothing, genotype_col::Nothing
    )

Extractor for experiments with multiple replicates.
This handles cases where:
- Multiple experimental replicates exist (rep_col is provided)
- Has neutral lineages (neutral_col is provided)
- No environment variations (env_col is Nothing)
- No genotype groupings (genotype_col is Nothing)

Can handle both cases where:
- All replicates have same number of timepoints (returns 3D array)
- Replicates have different timepoints (returns vector of matrices)

# Returns
- `DataArrays`: Struct containing processed barcode data including:
    - Either 3D array or vector of matrices for counts
    - Matrix or vector of vectors for totals
    - Metadata including replicate information
"""
function _extract_R(
    data::DF.AbstractDataFrame,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    neutral_col::Symbol,
    rep_col::Symbol,
    env_col::Nothing,
    genotype_col::Nothing
)::DataArrays
    # Extract unique time points
    timepoints = _extract_timepoints(data, time_col)

    # Group data by replicate
    data_rep_group = DF.groupby(data, rep_col)
    n_rep = length(data_rep_group)
    n_rep_time = [length(unique(d[:, time_col])) for d in data_rep_group]

    if length(unique(n_rep_time)) == 1
        # All replicates have same timepoints - use 3D array
        R̲̲⁽ⁿ⁾, neutral_ids = _process_neutral_barcodes_multi(
            data, neutral_col, id_col, time_col, count_col, rep_col, timepoints
        )
        R̲̲⁽ᵐ⁾, bc_ids = _process_mutant_barcodes_multi(
            data, neutral_col, id_col, time_col, count_col, rep_col, timepoints
        )

        R̲̲ = cat(R̲̲⁽ⁿ⁾, R̲̲⁽ᵐ⁾; dims=2)
        n̲ₜ = reshape(sum(R̲̲, dims=2), length(timepoints), n_rep)
    else
        # Replicates have different timepoints - use vector of matrices
        R̲̲⁽ⁿ⁾, neutral_ids = _process_neutral_barcodes_multi_varying(
            data_rep_group, neutral_col, id_col, time_col, count_col, n_rep_time
        )
        R̲̲⁽ᵐ⁾, bc_ids = _process_mutant_barcodes_multi_varying(
            data_rep_group, neutral_col, id_col, time_col, count_col, n_rep_time
        )

        R̲̲ = [hcat(R̲̲⁽ⁿ⁾[i], R̲̲⁽ᵐ⁾[i]) for i in 1:n_rep]
        n̲ₜ = vec.(sum.(R̲̲, dims=2))
    end

    return DataArrays(
        R̲̲,                    # bc_count
        n̲ₜ,                   # bc_total
        size(R̲̲⁽ⁿ⁾[1], 2),    # n_neutral
        size(R̲̲⁽ᵐ⁾[1], 2),    # n_bc
        bc_ids,               # bc_ids
        neutral_ids,          # neutral_ids
        "env1",              # envs
        1,                   # n_env
        n_rep,               # n_rep
        n_rep_time,          # n_time
        "N/A",              # genotypes
        0                    # n_geno
    )
end

# ============================================================================
# Case: Multiple environments
# ============================================================================

@doc raw"""
    _extract_R(
        data::DF.AbstractDataFrame, id_col::Symbol, time_col::Symbol,
        count_col::Symbol, neutral_col::Symbol, rep_col::Nothing, 
        env_col::Symbol, genotype_col::Nothing
    )

Extractor for experiments with multiple environments.
This handles cases where:
- Multiple environments exist (env_col is provided)
- May or may not have neutral lineages (neutral_col is Union{Symbol,Nothing})
- No replicates (rep_col is Nothing)
- No genotype groupings (genotype_col is Nothing)

# Returns
- `DataArrays`: Struct containing processed barcode data including:
    - Standard count data
    - Environment information
"""
function _extract_R(
    data::DF.AbstractDataFrame,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    neutral_col::Symbol,
    rep_col::Nothing,
    env_col::Symbol,
    genotype_col::Nothing
)::DataArrays
    # First process without environment information
    base_arrays = _extract_R(
        data, id_col, time_col, count_col, neutral_col,
        rep_col, nothing, genotype_col
    )

    # Extract environment information
    envs = collect(
        sort(unique(data[:, [time_col, env_col]]), time_col)[:, env_col]
    )
    n_env = length(unique(envs))

    # Create new DataArrays with environment information
    return DataArrays(
        base_arrays.bc_count,    # bc_count
        base_arrays.bc_total,    # bc_total
        base_arrays.n_neutral,   # n_neutral
        base_arrays.n_bc,        # n_bc
        base_arrays.bc_ids,      # bc_ids
        base_arrays.neutral_ids, # neutral_ids
        envs,                   # envs
        n_env,                  # n_env
        base_arrays.n_rep,      # n_rep
        base_arrays.n_time,     # n_time
        base_arrays.genotypes,  # genotypes
        base_arrays.n_geno      # n_geno
    )
end

# ============================================================================
# Case: Multiple environments and replicates
# ============================================================================

@doc raw"""
    _extract_R(
        data::DF.AbstractDataFrame, id_col::Symbol, time_col::Symbol,
        count_col::Symbol, neutral_col::Symbol, rep_col::Symbol, 
        env_col::Symbol, genotype_col::Nothing
    )

Extractor for experiments with both multiple environments and replicates.
This handles the most complex case where:
- Multiple environments exist (env_col is provided)
- Multiple replicates exist (rep_col is provided)
- No genotype groupings (genotype_col is Nothing)

Can handle cases where replicates have different numbers of timepoints.

# Returns
- `DataArrays`: Struct containing processed barcode data including:
- Count data (3D array or vector of matrices)
- Environment and replicate information
"""
function _extract_R(
    data::DF.AbstractDataFrame,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    neutral_col::Symbol,
    rep_col::Symbol,
    env_col::Symbol,
    genotype_col::Nothing
)::DataArrays
    # First process with just replicate information
    base_arrays = _extract_R(
        data, id_col, time_col, count_col, neutral_col,
        rep_col, nothing, genotype_col
    )

    # Group data by replicate
    data_rep = DF.groupby(data, rep_col)

    # Extract environment information
    envs = [
        collect(sort(unique(d[:, [time_col, env_col]]), time_col)[:, env_col])
        for d in data_rep
    ]
    n_env = length(unique(reduce(vcat, envs)))

    # If all replicates have same environments, simplify list
    if length(unique(envs)) == 1
        envs = first(envs)
    end

    # Create new DataArrays with environment information
    return DataArrays(
        base_arrays.bc_count,    # bc_count
        base_arrays.bc_total,    # bc_total
        base_arrays.n_neutral,   # n_neutral
        base_arrays.n_bc,        # n_bc
        base_arrays.bc_ids,      # bc_ids
        base_arrays.neutral_ids, # neutral_ids
        envs,                   # envs
        n_env,                  # n_env
        base_arrays.n_rep,      # n_rep
        base_arrays.n_time,     # n_time
        base_arrays.genotypes,  # genotypes
        base_arrays.n_geno      # n_geno
    )
end

# ============================================================================
# Case: Single replicate with genotypes
# ============================================================================

@doc raw"""
    _extract_R(
        data::DF.AbstractDataFrame, id_col::Symbol, time_col::Symbol,
        count_col::Symbol, neutral_col::Symbol, rep_col::Nothing, 
        env_col::Nothing, genotype_col::Symbol
    )

Extractor for experiments with genotype groupings.
This handles cases where:
- Single replicate (rep_col is Nothing)
- No environment variations (env_col is Nothing)
- Has genotype groupings (genotype_col is provided)

# Returns
- `DataArrays`: Struct containing processed barcode data including:
    - Standard count data
    - Genotype grouping information
"""
function _extract_R(
    data::DF.AbstractDataFrame,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    neutral_col::Symbol,
    rep_col::Nothing,
    env_col::Nothing,
    genotype_col::Symbol
)::DataArrays
    # First process without genotype information
    data_dict = _extract_R(
        data, id_col, time_col, count_col, neutral_col,
        rep_col, env_col, nothing
    )

    # Generate dictionary from bc to genotype
    bc_geno_dict = Dict(
        values.(keys(DF.groupby(data, [id_col, genotype_col])))
    )
    # Extract genotypes in the order they will be used in the inference
    genotypes = [bc_geno_dict[m] for m in data_dict.bc_ids]
    # Compute number of genotypes
    n_genotypes = length(unique(genotypes))

    return DataArrays(
        data_dict.bc_count,    # bc_count
        data_dict.bc_total,    # bc_total
        data_dict.n_neutral,   # n_neutral
        data_dict.n_bc,        # n_bc
        data_dict.bc_ids,      # bc_ids
        data_dict.neutral_ids, # neutral_ids
        data_dict.envs,       # envs
        data_dict.n_env,      # n_env
        data_dict.n_rep,      # n_rep
        data_dict.n_time,     # n_time
        genotypes,            # genotypes
        n_genotypes          # n_geno
    )
end

# ============================================================================
# Case: Multiple replicates with genotypes
# ============================================================================

@doc raw"""
    _extract_R(
        data::DF.AbstractDataFrame, id_col::Symbol, time_col::Symbol,
        count_col::Symbol, neutral_col::Symbol, rep_col::Symbol, 
        env_col::Nothing, genotype_col::Symbol
    )

Extractor for experiments with both replicates and genotype groupings.
This handles cases where:
- Multiple replicates exist (rep_col is provided)
- No environment variations (env_col is Nothing)
- Has genotype groupings (genotype_col is provided)

# Returns
- `DataArrays`: Struct containing processed barcode data including:
    - Count data (3D array or vector of matrices)
    - Replicate and genotype information
"""
function _extract_R(
    data::DF.AbstractDataFrame,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    neutral_col::Symbol,
    rep_col::Symbol,
    env_col::Nothing,
    genotype_col::Symbol
)::DataArrays
    # First process with replicate information
    data_dict = _extract_R(
        data, id_col, time_col, count_col, neutral_col,
        rep_col, env_col, nothing
    )

    # Generate dictionary from bc to genotype
    bc_geno_dict = Dict(
        values.(keys(DF.groupby(data, [id_col, genotype_col])))
    )
    # Extract genotypes in the order they will be used in the inference
    genotypes = [bc_geno_dict[m] for m in data_dict.bc_ids]
    # Compute number of genotypes
    n_genotypes = length(unique(genotypes))

    return DataArrays(
        data_dict.bc_count,    # bc_count
        data_dict.bc_total,    # bc_total
        data_dict.n_neutral,   # n_neutral
        data_dict.n_bc,        # n_bc
        data_dict.bc_ids,      # bc_ids
        data_dict.neutral_ids, # neutral_ids
        data_dict.envs,       # envs
        data_dict.n_env,      # n_env
        data_dict.n_rep,      # n_rep
        data_dict.n_time,     # n_time
        genotypes,            # genotypes
        n_genotypes          # n_geno
    )
end

# ============================================================================
# Case: Multiple environments with genotypes
# ============================================================================

@doc raw"""
    _extract_R(
        data::DF.AbstractDataFrame, id_col::Symbol, time_col::Symbol,
        count_col::Symbol, neutral_col::Symbol, rep_col::Nothing, 
        env_col::Symbol, genotype_col::Symbol
    )

Extractor for experiments with both environments and genotype groupings.
This handles cases where:
- Multiple environments exist (env_col is provided)
- No replicates (rep_col is Nothing)
- Has genotype groupings (genotype_col is provided)

# Returns
- `DataArrays`: Struct containing processed barcode data including:
    - Standard count data
    - Environment and genotype information
"""
function _extract_R(
    data::DF.AbstractDataFrame,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    neutral_col::Symbol,
    rep_col::Nothing,
    env_col::Symbol,
    genotype_col::Symbol
)::DataArrays
    # First process with environment information
    data_dict = _extract_R(
        data, id_col, time_col, count_col, neutral_col,
        rep_col, env_col, nothing
    )

    # Generate dictionary from bc to genotype
    bc_geno_dict = Dict(
        values.(keys(DF.groupby(data, [id_col, genotype_col])))
    )
    # Extract genotypes in the order they will be used in the inference
    genotypes = [bc_geno_dict[m] for m in data_dict.bc_ids]
    # Compute number of genotypes
    n_genotypes = length(unique(genotypes))

    return DataArrays(
        data_dict.bc_count,    # bc_count
        data_dict.bc_total,    # bc_total
        data_dict.n_neutral,   # n_neutral
        data_dict.n_bc,        # n_bc
        data_dict.bc_ids,      # bc_ids
        data_dict.neutral_ids, # neutral_ids
        data_dict.envs,       # envs
        data_dict.n_env,      # n_env
        data_dict.n_rep,      # n_rep
        data_dict.n_time,     # n_time
        genotypes,            # genotypes
        n_genotypes          # n_geno
    )
end

# ============================================================================
# Case: Multiple environments, replicates, and genotypes
# ============================================================================

@doc raw"""
    _extract_R(
        data::DF.AbstractDataFrame, id_col::Symbol, time_col::Symbol,
        count_col::Symbol, neutral_col::Symbol, rep_col::Symbol, 
        env_col::Symbol, genotype_col::Symbol
    )

Extractor for the most complex case with environments, replicates, and
genotypes. This handles cases where:
- Multiple environments exist (env_col is provided)
- Multiple replicates exist (rep_col is provided)
- Has genotype groupings (genotype_col is provided)

# Returns
- `DataArrays`: Struct containing processed barcode data including:
    - Count data (3D array or vector of matrices)
    - Environment, replicate, and genotype information
"""
function _extract_R(
    data::DF.AbstractDataFrame,
    id_col::Symbol,
    time_col::Symbol,
    count_col::Symbol,
    neutral_col::Symbol,
    rep_col::Symbol,
    env_col::Symbol,
    genotype_col::Symbol
)::DataArrays
    # First process with environment and replicate information
    data_dict = _extract_R(
        data, id_col, time_col, count_col, neutral_col,
        rep_col, env_col, nothing
    )

    # Generate dictionary from bc to genotype
    bc_geno_dict = Dict(
        values.(keys(DF.groupby(data, [id_col, genotype_col])))
    )
    # Extract genotypes in the order they will be used in the inference
    genotypes = [bc_geno_dict[m] for m in data_dict.bc_ids]
    # Compute number of genotypes
    n_genotypes = length(unique(genotypes))

    return DataArrays(
        data_dict.bc_count,    # bc_count
        data_dict.bc_total,    # bc_total
        data_dict.n_neutral,   # n_neutral
        data_dict.n_bc,        # n_bc
        data_dict.bc_ids,      # bc_ids
        data_dict.neutral_ids, # neutral_ids
        data_dict.envs,       # envs
        data_dict.n_env,      # n_env
        data_dict.n_rep,      # n_rep
        data_dict.n_time,     # n_time
        genotypes,            # genotypes
        n_genotypes          # n_geno
    )
end

# ============================================================================
# Main function to preprocess data
# ============================================================================

@doc raw"""
`data_to_arrays(data; kwargs)`

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

# Returns
- `data_arrays::DataArrays`: Struct containing the following elements:
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
    - `n_neutral`: Number of neutral barcodes.
    - `n_bc`: Number of non-neutral barcodes.
    - `bc_ids`: List of barcode IDs in the order they are used for the
      inference.
    - `neutral_ids`: List of neutral barcode IDs in the order they are used for
      the inference.
    - `envs`: List of environments. The options can be:
        - `String`: Single placeholder `env1`
        - `Vector{<:Any}`: Environments in the order they were measured.
        - `vector{Vector{<:Any}}`: Environments per replicate when replicates
          have a different number of time points.
    - `n_env`: Number of environmental conditions.
    - `n_rep`: Number of experimental replicates.
    - `n_time`: Number of time points. The options can be:
        - `Int64`: Number of time points on a single replicate or multiple
          replicates.
        - `Vector{Int64}`: Number of time points per replicate when replicates
          have different lengths.
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
    genotype_col::Union{Nothing,Symbol}=nothing
)
    # Check that all *_col arguments exist in the dataframe if non-nothing
    for x_col in [id_col, time_col, count_col, neutral_col, rep_col, env_col, genotype_col]
        if typeof(x_col) <: Symbol
            if !(string(x_col) in names(data))
                error("Column $x_col does not exist in the dataframe")
            end # if
        end # if
    end # for

    # Check that neutral_col is of type Bool
    if !(typeof(data[!, neutral_col]) <: Vector{Bool})
        error("Column $neutral_col must be of type Bool")
    end # if

    # Extract information from dataframe with internal function. Note: multiple
    # dispatch is used to call the correct function based on the columns
    # provided.
    return _extract_R(
        data,
        id_col,
        time_col,
        count_col,
        neutral_col,
        rep_col,
        env_col,
        genotype_col
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    advi_to_df(
        data::DataFrames.AbstractDataFrame, dist::Distribution.Sampleable,
        vars::Vector{<:Any}; kwargs
    )

Convert the output of automatic differentiation variational inference (ADVI) to
a tidy dataframe.

# Arguments
- `data::DataFrames.AbstractDataFrame`: Tidy dataframe used to perform the ADVI
  inference. See `BarBay.vi` module for the dataframe requirements.
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
        - `pop_std`: (Nuisance parameter) Log of standard deviation in the
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
        - `bc_std`: (Nuisance parameter) Log of standard deviation in the
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
    df_par[!, :id] = Vector{Any}(undef, size(df_par, 1))

    # Loop through var groups
    for (i, var) in enumerate(var_groups)
        if occursin("̲ₜ", var)
            # Population mean fitness variables are not associated with any
            # particular barcode.
            df_par[var_range[i], :id] .= "N/A"
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