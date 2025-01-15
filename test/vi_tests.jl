## =============================================================================

using BarBay
using Test
using CSV
using DataFrames
using StatsBase
using Turing
using AdvancedVI
using ReverseDiff

## =============================================================================

@testset "vi tests" begin
    # ========================================================================
    # Test ADVI with data001_single.csv (single condition)
    # ========================================================================
    @testset "Single Condition Data (data001)" begin
        # Load test data
        data = CSV.read("data/data001_single.csv", DataFrame)

        # Define test parameters
        test_advi = Turing.ADVI(1, 1, Turing.AutoReverseDiff(true))

        # Test basic functionality
        result = BarBay.vi.advi(
            data=data,
            model=BarBay.model.fitness_normal,
            advi=test_advi
        )

        # Test result properties
        @test result isa DataFrame
        @test :mean in propertynames(result)
        @test :std in propertynames(result)
        @test :vartype in propertynames(result)
        @test :varname in propertynames(result)

        # Test that all required variable types are present
        required_vartypes = [
            "pop_mean_fitness",
            "pop_std",
            "bc_fitness",
            "bc_std",
            "log_poisson"
        ]
        @test all(vt in result.vartype for vt in required_vartypes)

        # Test with ForwardDiffAD
        test_advi_forward = Turing.ADVI(1, 1, Turing.AutoForwardDiff())
        result_forward = BarBay.vi.advi(
            data=data,
            model=BarBay.model.fitness_normal,
            advi=test_advi_forward
        )
        @test result_forward isa DataFrame
    end

    # ========================================================================
    # Test ADVI with data002_hier-rep.csv (hierarchical replicates)
    # ========================================================================
    @testset "Hierarchical Replicates Data (data002)" begin
        # Load test data
        data = CSV.read("data/data002_hier-rep.csv", DataFrame)

        # Define test parameters
        test_advi = Turing.ADVI(1, 1, Turing.AutoReverseDiff(true))

        # Test with replicates
        result = BarBay.vi.advi(
            data=data,
            model=BarBay.model.replicate_fitness_normal,
            rep_col=:rep,
            advi=test_advi
        )

        # Test result properties
        @test result isa DataFrame
        @test :rep in propertynames(result)
        @test :id in propertynames(result)
        @test :vartype in propertynames(result)

        # Test that hierarchical model variables are present
        hierarchical_vartypes = [
            "bc_hyperfitness",
            "bc_noncenter",
            "bc_deviations"
        ]
        @test all(vt in result.vartype for vt in hierarchical_vartypes)

        # Test with ForwardDiffAD
        test_advi_forward = Turing.ADVI(1, 1, Turing.AutoForwardDiff())
        result_forward = BarBay.vi.advi(
            data=data,
            model=BarBay.model.replicate_fitness_normal,
            rep_col=:rep,
            advi=test_advi_forward
        )
        @test result_forward isa DataFrame

        # Test with uneven timepoints
        data_uneven = data[
            (data.rep.!=maximum(data.rep)).|(data.time.!=maximum(data.time)),
            :]
        result_uneven = BarBay.vi.advi(
            data=data_uneven,
            model=BarBay.model.replicate_fitness_normal,
            rep_col=:rep,
            advi=test_advi
        )
        @test result_uneven isa DataFrame
    end

    # ========================================================================
    # Test ADVI with data003_multienv.csv (multiple environments)
    # ========================================================================
    @testset "Multiple Environments Data (data003)" begin
        # Load test data
        data = CSV.read("data/data003_multienv.csv", DataFrame)

        # Define test parameters
        test_advi = Turing.ADVI(1, 1, Turing.AutoReverseDiff(true))

        # Test with environments
        result = BarBay.vi.advi(
            data=data,
            model=BarBay.model.multienv_fitness_normal,
            env_col=:env,
            advi=test_advi
        )

        # Test result properties
        @test result isa DataFrame
        @test :env in propertynames(result)

        # Test with ForwardDiffAD
        test_advi_forward = Turing.ADVI(1, 1, Turing.AutoForwardDiff())
        result_forward = BarBay.vi.advi(
            data=data,
            model=BarBay.model.multienv_fitness_normal,
            env_col=:env,
            advi=test_advi_forward
        )
        @test result_forward isa DataFrame
    end

    # ========================================================================
    # Test ADVI with data004_multigen.csv (multiple genotypes)
    # ========================================================================
    @testset "Multiple Genotypes Data (data004)" begin
        # Load test data
        data = CSV.read("data/data004_multigen.csv", DataFrame)

        # Define test parameters
        test_advi = Turing.ADVI(1, 1, Turing.AutoReverseDiff(true))

        # Test with genotypes
        result = BarBay.vi.advi(
            data=data,
            model=BarBay.model.genotype_fitness_normal,
            genotype_col=:genotype,
            advi=test_advi
        )

        # Test result properties
        @test result isa DataFrame

        # Test that hierarchical model variables are present
        hierarchical_vartypes = [
            "bc_hyperfitness",
            "bc_noncenter",
            "bc_deviations"
        ]
        @test all(vt in result.vartype for vt in hierarchical_vartypes)

        # Test with ForwardDiffAD
        test_advi_forward = Turing.ADVI(1, 1, Turing.AutoForwardDiff())
        result_forward = BarBay.vi.advi(
            data=data,
            model=BarBay.model.genotype_fitness_normal,
            genotype_col=:genotype,
            advi=test_advi_forward
        )
        @test result_forward isa DataFrame
    end

    # ========================================================================
    # Test error handling
    # ========================================================================
    @testset "Error Handling" begin
        data = CSV.read("data/data001_single.csv", DataFrame)
        test_advi = Turing.ADVI(1, 1, Turing.AutoReverseDiff(true))

        # Test error when trying to use hierarchical model without replicate
        # info
        @test_throws ErrorException BarBay.vi.advi(
            data=data,
            model=BarBay.model.replicate_fitness_normal,
            advi=test_advi
        )

        # Test error when trying to use multienv model without environment info
        @test_throws ErrorException BarBay.vi.advi(
            data=data,
            model=BarBay.model.multienv_fitness_normal,
            advi=test_advi
        )
    end

    # ========================================================================
    # Test output file handling
    # ========================================================================
    @testset "Output File Handling" begin
        data = CSV.read("data/data001_single.csv", DataFrame)
        test_advi = Turing.ADVI(1, 1, Turing.AutoReverseDiff(true))

        # Test file output
        output_file = tempname()
        result = BarBay.vi.advi(
            data=data,
            model=BarBay.model.fitness_normal,
            outputname=output_file,
            advi=test_advi
        )

        # Test that result is nothing (since saving to file)
        @test isnothing(result)

        # Test that file exists and is readable
        @test isfile("$(output_file).csv")
        df = CSV.read("$(output_file).csv", DataFrame)
        @test df isa DataFrame

        # Clean up
        rm("$(output_file).csv")
    end
end