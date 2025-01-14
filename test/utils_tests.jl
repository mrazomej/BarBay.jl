using BarBay
using Test
using CSV
using DataFrames
using StatsBase
using Turing
using AdvancedVI
using ReverseDiff

@testset "utils tests" begin
    # ========================================================================
    # Test helper functions
    # ========================================================================

    @testset "Helper Functions" begin
        # Load sample data
        data = CSV.read("data/data001_single.csv", DataFrame)

        @testset "_extract_timepoints" begin
            timepoints = BarBay.utils._extract_timepoints(data, :time)
            @test typeof(timepoints) <: Vector
            @test issorted(timepoints)
        end

        @testset "_process_neutral_barcodes_single" begin
            timepoints = BarBay.utils._extract_timepoints(data, :time)
            R̲̲⁽ⁿ⁾, neutral_ids = BarBay.utils._process_neutral_barcodes_single(
                data, :neutral, :barcode, :time, :count, timepoints
            )
            @test typeof(R̲̲⁽ⁿ⁾) <: Matrix{Int64}
            @test size(R̲̲⁽ⁿ⁾, 1) == length(timepoints)
            @test !isempty(neutral_ids)
        end

        @testset "_process_mutant_barcodes_single" begin
            timepoints = BarBay.utils._extract_timepoints(data, :time)
            R̲̲⁽ᵐ⁾, bc_ids = BarBay.utils._process_mutant_barcodes_single(
                data, :neutral, :barcode, :time, :count, timepoints
            )
            @test typeof(R̲̲⁽ᵐ⁾) <: Matrix{Int64}
            @test size(R̲̲⁽ᵐ⁾, 1) == length(timepoints)
            @test !isempty(bc_ids)
        end

        @testset "_process_neutral_barcodes_multi" begin
            # Load data with multiple replicates
            data = CSV.read("data/data002_hier-rep.csv", DataFrame)
            timepoints = BarBay.utils._extract_timepoints(data, :time)

            R̲̲⁽ⁿ⁾, neutral_ids = BarBay.utils._process_neutral_barcodes_multi(
                data, :neutral, :barcode, :time, :count, :rep, timepoints
            )

            @test typeof(R̲̲⁽ⁿ⁾) <: Array{Int64,3}
            @test size(R̲̲⁽ⁿ⁾, 1) == length(timepoints)
            @test !isempty(neutral_ids)
            @test size(R̲̲⁽ⁿ⁾, 3) == length(unique(data.rep))
        end

        @testset "_process_mutant_barcodes_multi" begin
            # Load data with multiple replicates
            data = CSV.read("data/data002_hier-rep.csv", DataFrame)
            timepoints = BarBay.utils._extract_timepoints(data, :time)

            R̲̲⁽ᵐ⁾, bc_ids = BarBay.utils._process_mutant_barcodes_multi(
                data, :neutral, :barcode, :time, :count, :rep, timepoints
            )

            @test typeof(R̲̲⁽ᵐ⁾) <: Array{Int64,3}
            @test size(R̲̲⁽ᵐ⁾, 1) == length(timepoints)
            @test !isempty(bc_ids)
            @test size(R̲̲⁽ᵐ⁾, 3) == length(unique(data.rep))
        end

        @testset "_process_neutral_barcodes_multi_varying" begin
            # Create data with varying timepoints
            data = CSV.read("data/data002_hier-rep.csv", DataFrame)
            # Remove last timepoint from last replicate to create varying
            # timepoints
            data_uneven = data[
                (data.rep.!=maximum(data.rep)).|(data.time.!=maximum(data.time)),
                :]

            # Group data by replicate
            data_rep = groupby(data_uneven, :rep)
            # Get number of timepoints per replicate
            n_rep_time = [length(unique(d.time)) for d in data_rep]

            R̲̲⁽ⁿ⁾, neutral_ids = BarBay.utils._process_neutral_barcodes_multi_varying(
                data_rep, :neutral, :barcode, :time, :count, n_rep_time
            )

            @test typeof(R̲̲⁽ⁿ⁾) <: Vector{Matrix{Int64}}
            @test length(R̲̲⁽ⁿ⁾) == length(n_rep_time)
            @test !isempty(neutral_ids)
            @test size(R̲̲⁽ⁿ⁾[1], 1) == n_rep_time[1]
        end

        @testset "_process_mutant_barcodes_multi_varying" begin
            # Create data with varying timepoints
            data = CSV.read("data/data002_hier-rep.csv", DataFrame)
            # Remove last timepoint from last replicate to create varying timepoints
            data_uneven = data[
                (data.rep.!=maximum(data.rep)).|(data.time.!=maximum(data.time)),
                :]

            # Group data by replicate
            data_rep = groupby(data_uneven, :rep)
            # Get number of timepoints per replicate
            n_rep_time = [length(unique(d.time)) for d in data_rep]

            R̲̲⁽ᵐ⁾, bc_ids = BarBay.utils._process_mutant_barcodes_multi_varying(
                data_rep, :neutral, :barcode, :time, :count, n_rep_time
            )

            @test typeof(R̲̲⁽ᵐ⁾) <: Vector{Matrix{Int64}}
            @test length(R̲̲⁽ᵐ⁾) == length(n_rep_time)
            @test !isempty(bc_ids)
            @test size(R̲̲⁽ᵐ⁾[1], 1) == n_rep_time[1]
        end

        @testset "Error handling in processing functions" begin
            # Test error when timepoints are missing
            data = CSV.read("data/data002_hier-rep.csv", DataFrame)
            # Remove a timepoint from one barcode
            data_missing = data[2:end, :]

            @test_throws ErrorException BarBay.utils._process_neutral_barcodes_single(
                data_missing, :neutral, :barcode, :time, :count,
                BarBay.utils._extract_timepoints(data, :time)
            )

            @test_throws ErrorException BarBay.utils._process_mutant_barcodes_single(
                data_missing, :neutral, :barcode, :time, :count,
                BarBay.utils._extract_timepoints(data, :time)
            )
        end
    end

    # ========================================================================
    # Test _extract_R variants with data001_single.csv (single condition)
    # ========================================================================

    @testset "Single Condition Data (data001)" begin
        data = CSV.read("data/data001_single.csv", DataFrame)

        @testset "Base case" begin
            result = BarBay.utils._extract_R(
                data, :barcode, :time, :count, :neutral,
                nothing, nothing, nothing
            )
            @test typeof(result) == BarBay.utils.DataArrays
            @test result.n_env == 1
            @test result.n_rep == 1
            @test result.n_neutral > 0
            @test typeof(result.bc_count) <: Matrix{Int64}
        end
    end

    # ========================================================================
    # Test _extract_R variants with data002_hier-rep.csv (hierarchical replicates)
    # ========================================================================

    @testset "Hierarchical Replicates Data (data002)" begin
        data = CSV.read("data/data002_hier-rep.csv", DataFrame)

        @testset "Multiple replicates" begin
            result = BarBay.utils._extract_R(
                data, :barcode, :time, :count, :neutral,
                :rep, nothing, nothing
            )
            @test typeof(result) == BarBay.utils.DataArrays
            @test result.n_rep > 1
            @test typeof(result.bc_count) <: Union{Array{Int64,3},Vector{Matrix{Int64}}}
            @test result.n_env == 1
        end

        # Test with removing last timepoint from a replicate
        @testset "Multiple replicates with uneven timepoints" begin
            data_uneven = data[
                (data.rep.!=maximum(data.rep)).|(data.time.!=maximum(data.time)), :]

            result = BarBay.utils._extract_R(
                data_uneven, :barcode, :time, :count, :neutral,
                :rep, nothing, nothing
            )
            @test typeof(result) == BarBay.utils.DataArrays
            @test typeof(result.n_time) <: Vector{Int}
            @test typeof(result.bc_count) <: Vector{Matrix{Int64}}
        end
    end

    # ========================================================================
    # Test _extract_R variants with data003_multienv.csv (multiple environments)
    # ========================================================================

    @testset "Multiple Environments Data (data003)" begin
        data = CSV.read("data/data003_multienv.csv", DataFrame)

        @testset "Multiple environments" begin
            result = BarBay.utils._extract_R(
                data, :barcode, :time, :count, :neutral,
                nothing, :env, nothing
            )
            @test typeof(result) == BarBay.utils.DataArrays
            @test result.n_env > 1
            @test typeof(result.envs) <: Vector
            @test result.n_rep == 1
        end
    end

    # ========================================================================
    # Test _extract_R variants with data004_multigen.csv (multiple genotypes)
    # ========================================================================

    @testset "Multiple Genotypes Data (data004)" begin
        data = CSV.read("data/data004_multigen.csv", DataFrame)

        @testset "Single condition with genotypes" begin
            result = BarBay.utils._extract_R(
                data, :barcode, :time, :count, :neutral,
                nothing, nothing, :genotype
            )
            @test typeof(result) == BarBay.utils.DataArrays
            @test result.n_geno > 0
            @test typeof(result.genotypes) <: Vector
            @test result.n_env == 1
            @test result.n_rep == 1
        end
    end

    # ========================================================================
    # Test data_to_arrays wrapper function with all datasets
    # ========================================================================
    @testset "data_to_arrays wrapper function" begin
        @testset "Single condition data" begin
            data = CSV.read("data/data001_single.csv", DataFrame)
            result = BarBay.utils.data_to_arrays(data)
            @test typeof(result) <: BarBay.utils.DataArrays
            @test hasproperty(result, :bc_count)
            @test hasproperty(result, :n_neutral)
        end

        @testset "Hierarchical replicates data" begin
            data = CSV.read("data/data002_hier-rep.csv", DataFrame)
            result = BarBay.utils.data_to_arrays(data, rep_col=:rep)
            @test typeof(result) <: BarBay.utils.DataArrays
            @test hasproperty(result, :n_rep)
            @test result.n_rep > 1
        end

        @testset "Multiple environments data" begin
            data = CSV.read("data/data003_multienv.csv", DataFrame)
            result = BarBay.utils.data_to_arrays(data, env_col=:env)
            @test typeof(result) <: BarBay.utils.DataArrays
            @test hasproperty(result, :envs)
            @test result.n_env > 1
        end

        @testset "Multiple genotypes data" begin
            data = CSV.read("data/data004_multigen.csv", DataFrame)
            result = BarBay.utils.data_to_arrays(data, genotype_col=:genotype)
            @test typeof(result) <: BarBay.utils.DataArrays
            @test hasproperty(result, :genotypes)
            @test result.n_geno > 0
        end
    end

    # ========================================================================
    # Test error handling
    # ========================================================================

    @testset "Error Handling" begin
        data = CSV.read("data/data001_single.csv", DataFrame)

        # Test missing required columns
        @test_throws ErrorException BarBay.utils.data_to_arrays(
            data, id_col=:nonexistent_column
        )

        # Test mismatched timepoints
        bad_data = vcat(data, data[1:5, :])  # Duplicate some rows
        @test_throws ErrorException BarBay.utils.data_to_arrays(bad_data)

        # Test invalid neutral_col type
        data.neutral .= "true"
        @test_throws ErrorException BarBay.utils.data_to_arrays(data)
    end

    @testset "data_to_arrays function tests" begin
        # ========================================================================
        # Test basic functionality with minimal dataset
        # ========================================================================
        @testset "Basic functionality (data001)" begin
            data = CSV.read("data/data001_single.csv", DataFrame)

            # Test default parameters
            result = BarBay.utils.data_to_arrays(data)
            @test result isa BarBay.utils.DataArrays

            # Test data types of each field
            @test result.bc_count isa Matrix{Int64}
            @test result.bc_total isa Vector{Int64}
            @test result.n_neutral isa Int
            @test result.n_bc isa Int
            @test result.bc_ids isa Vector
            @test result.neutral_ids isa Vector
            @test result.envs isa String  # "env1" for single environment
            @test result.n_env isa Int
            @test result.n_rep isa Int
            @test result.n_time isa Int
            @test result.genotypes isa String  # "N/A" for no genotypes
            @test result.n_geno isa Int

            # Test data dimensions
            @test size(result.bc_count, 2) == result.n_neutral + result.n_bc
            @test length(result.bc_total) == size(result.bc_count, 1)

        end

        # ========================================================================
        # Test hierarchical replicate functionality
        # ========================================================================
        @testset "Hierarchical replicates (data002)" begin
            data = CSV.read("data/data002_hier-rep.csv", DataFrame)

            # Test with replicates - all timepoints equal
            result = BarBay.utils.data_to_arrays(data, rep_col=:rep)
            @test result isa BarBay.utils.DataArrays
            @test result.n_rep > 1
            @test result.bc_count isa Array{Int64,3}
            @test result.bc_total isa Matrix{Int64}

            # Test with uneven timepoints between replicates
            data_uneven = data[
                (data.rep.!=maximum(data.rep)).|(data.time.!=maximum(data.time)),
                :]
            result_uneven = BarBay.utils.data_to_arrays(data_uneven, rep_col=:rep)
            @test result_uneven.bc_count isa Vector{Matrix{Int64}}
            @test result_uneven.bc_total isa Vector{Vector{Int64}}
            @test result_uneven.n_time isa Vector{Int}
        end

        # ========================================================================
        # Test multiple environment functionality
        # ========================================================================
        @testset "Multiple environments (data003)" begin
            data = CSV.read("data/data003_multienv.csv", DataFrame)

            # Test with environments
            result = BarBay.utils.data_to_arrays(data, env_col=:env)
            @test result isa BarBay.utils.DataArrays
            @test result.n_env > 1
            @test result.envs isa Vector{<:Any}
            @test length(result.envs) == size(result.bc_count, 1)
        end

        # ========================================================================
        # Test genotype functionality
        # ========================================================================
        @testset "Multiple genotypes (data004)" begin
            data = CSV.read("data/data004_multigen.csv", DataFrame)

            # Test with genotypes
            result = BarBay.utils.data_to_arrays(data, genotype_col=:genotype)
            @test result isa BarBay.utils.DataArrays
            @test result.n_geno > 0
            @test result.genotypes isa Vector{<:Any}
            @test length(result.genotypes) == result.n_bc

            # Verify genotype assignments
            genotype_mapping = Dict(zip(result.bc_ids, result.genotypes))
            for id in result.bc_ids
                orig_genotype = first(data[data.barcode.==id, :genotype])
                @test genotype_mapping[id] == orig_genotype
            end
        end

        # ========================================================================
        # Test error handling
        # ========================================================================
        @testset "Error handling" begin
            data = CSV.read("data/data001_single.csv", DataFrame)

            # Test missing column errors
            @test_throws Exception BarBay.utils.data_to_arrays(
                data, id_col=:nonexistent
            )

            # Test incomplete data errors
            incomplete_data = data[2:end, :]  # Remove a timepoint
            @test_throws Exception BarBay.utils.data_to_arrays(incomplete_data)
        end

        # ========================================================================
        # Test Optional Parameters and Output Consistency
        # ========================================================================
        @testset "Optional parameters and consistency" begin
            data = CSV.read("data/data001_single.csv", DataFrame)

            # Test custom column names
            renamed_data = copy(data)
            rename!(
                renamed_data,
                :barcode => :sequence_id,
                :time => :cycle,
                :count => :reads,
                :neutral => :is_neutral
            )

            result = BarBay.utils.data_to_arrays(
                renamed_data,
                id_col=:sequence_id,
                time_col=:cycle,
                count_col=:reads,
                neutral_col=:is_neutral
            )
            @test result isa BarBay.utils.DataArrays
            @test !isempty(result.bc_ids)
            @test result.n_neutral > 0
        end
    end
end