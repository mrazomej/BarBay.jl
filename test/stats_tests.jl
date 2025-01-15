using BarBay
using Test
using CSV
using DataFrames
using StatsBase
using Turing
using Distributions
using LinearAlgebra
using AdvancedVI
using ReverseDiff

## -----------------------------------------------------------------------------

@testset "stats tests" begin
    # ========================================================================
    # Test matrix_quantile_range function
    # ========================================================================
    @testset "matrix_quantile_range" begin
        # Create test matrix
        test_matrix = randn(10, 5)

        # Test basic functionality
        quantiles = [0.95]
        result = BarBay.stats.matrix_quantile_range(quantiles, test_matrix)

        # Test output shape
        @test size(result) == (size(test_matrix, 2), length(quantiles), 2)

        # Test quantile values are reasonable
        @test all(result[:, 1, 2] .>= result[:, 1, 1])  # upper bounds > lower bounds

        # Test error for invalid quantiles
        @test_throws ErrorException BarBay.stats.matrix_quantile_range([1.5], test_matrix)
        @test_throws ErrorException BarBay.stats.matrix_quantile_range([-0.5], test_matrix)

        # Test error for invalid dimensions
        @test_throws ErrorException BarBay.stats.matrix_quantile_range(quantiles, test_matrix, dims=3)
    end

    # ========================================================================
    # Test posterior predictive check functions
    # ========================================================================
    @testset "Posterior Predictive Checks" begin
        # Load test data
        data = CSV.read("data/data001_single.csv", DataFrame)

        @testset "freq_bc_ppc" begin
            # Create test DataFrame with required columns
            test_df = DataFrame(
                :s⁽ᵐ⁾ => randn(100),
                :σ⁽ᵐ⁾ => abs.(randn(100)),
                Symbol("f̲⁽ᵐ⁾[1]") => abs.(randn(100)),
                :s̲ₜ₁ => randn(100),
                :s̲ₜ₂ => randn(100)
            )

            # Test basic functionality
            result = BarBay.stats.freq_bc_ppc(test_df, 10)
            @test size(result, 2) == 3  # initial + 2 timepoints

            # Test with non-default parameters
            custom_params = Dict(
                :bc_mean_fitness => :custom_mean,
                :bc_std_fitness => :custom_std,
                :bc_freq => :custom_freq,
                :population_mean_fitness => :custom_pop
            )

            # Add custom columns
            test_df.custom_mean = test_df.s⁽ᵐ⁾
            test_df.custom_std = test_df.σ⁽ᵐ⁾
            test_df.custom_freq = test_df[!, Symbol("f̲⁽ᵐ⁾[1]")]
            test_df.custom_pop₁ = test_df.s̲ₜ₁
            test_df.custom_pop₂ = test_df.s̲ₜ₂

            result_custom = BarBay.stats.freq_bc_ppc(test_df, 10, param=custom_params)
            @test size(result_custom) == size(result)
        end

        @testset "logfreq_ratio_bc_ppc" begin
            # Create test DataFrame
            test_df = DataFrame(
                s⁽ᵐ⁾=randn(100),
                σ⁽ᵐ⁾=abs.(randn(100)),
                s̲ₜ₁=randn(100),
                s̲ₜ₂=randn(100)
            )

            # Test basic functionality
            result = BarBay.stats.logfreq_ratio_bc_ppc(test_df, 10)
            @test size(result, 2) == 2  # 2 timepoints

            # Test non-flattened output
            result_nonflat = BarBay.stats.logfreq_ratio_bc_ppc(test_df, 10, flatten=false)
            @test size(result_nonflat, 3) == 10  # 10 samples
        end

        @testset "logfreq_ratio_popmean_ppc" begin
            # Create test DataFrame
            test_df = DataFrame(
                sₜ₁=randn(100),
                sₜ₂=randn(100),
                σₜ₁=abs.(randn(100)),
                σₜ₂=abs.(randn(100))
            )

            # Test basic functionality
            result = BarBay.stats.logfreq_ratio_popmean_ppc(test_df, 10)
            @test size(result, 2) == 2  # 2 timepoints

            # Test non-default parameters
            custom_params = Dict(
                :population_mean_fitness => :custom_mean,
                :population_std_fitness => :custom_std
            )

            # Add custom columns
            test_df.custom_mean₁ = test_df.sₜ₁
            test_df.custom_mean₂ = test_df.sₜ₂
            test_df.custom_std₁ = test_df.σₜ₁
            test_df.custom_std₂ = test_df.σₜ₂

            result_custom = BarBay.stats.logfreq_ratio_popmean_ppc(
                test_df, 10, param=custom_params
            )
            @test size(result_custom) == size(result)
        end
    end

    # ========================================================================
    # Test naive estimation functions
    # ========================================================================
    @testset "Naive Estimation Functions" begin
        # Load test data
        data = CSV.read("data/data001_single.csv", DataFrame)

        @testset "naive_fitness" begin
            # Test basic functionality
            result = BarBay.stats.naive_fitness(data)
            @test result isa DataFrame
            @test :fitness in propertynames(result)
            @test :barcode in propertynames(result)

            # Test with custom parameters
            result_custom = BarBay.stats.naive_fitness(
                data,
                id_col=:barcode,
                time_col=:time,
                count_col=:count,
                neutral_col=:neutral,
                pseudocount=2
            )
            @test result_custom isa DataFrame
        end

    end

    @testset "Expanded naive_prior tests" begin
        # ========================================================================
        # Test naive_prior with data001_single.csv (single condition)
        # ========================================================================
        @testset "Single Condition Data (data001)" begin
            # Load test data
            data = CSV.read("data/data001_single.csv", DataFrame)

            # Test basic functionality
            result = BarBay.stats.naive_prior(data)

            # Test result properties
            @test result isa Dict
            @test :s_pop_prior in keys(result)
            @test :logσ_pop_prior in keys(result)
            @test :logλ_prior in keys(result)

            # Test that the priors have reasonable values
            @test !any(isnan.(result[:s_pop_prior]))
            @test !any(isnan.(result[:logσ_pop_prior]))
            @test !any(isnan.(result[:logλ_prior]))

            # Test that λ prior dimensions match data dimensions
            expected_λ_length = size(unique(data.time), 1) *
                                size(unique(data.barcode), 1)
            @test length(result[:logλ_prior]) == expected_λ_length
        end

        # ========================================================================
        # Test naive_prior with data002_hier-rep.csv (hierarchical replicates)
        # ========================================================================
        @testset "Hierarchical Replicates Data (data002)" begin
            # Load test data
            data = CSV.read("data/data002_hier-rep.csv", DataFrame)

            # Test with replicates - all timepoints equal
            result = BarBay.stats.naive_prior(data, rep_col=:rep)

            # Test result properties
            @test result isa Dict
            @test all(k in keys(result) for k in [:s_pop_prior, :logσ_pop_prior, :logλ_prior])

            # Test dimensions with replicates
            n_reps = length(unique(data.rep))
            n_times = length(unique(data.time))
            n_barcodes = length(unique(data.barcode))

            # Test that population parameters account for all replicates
            @test length(result[:s_pop_prior]) == (n_times - 1) * n_reps
            @test length(result[:logσ_pop_prior]) == (n_times - 1) * n_reps

            # Test with uneven timepoints between replicates
            data_uneven = data[
                (data.rep.!=maximum(data.rep)).|(data.time.!=maximum(data.time)),
                :]
            result_uneven = BarBay.stats.naive_prior(data_uneven, rep_col=:rep)

            # Test result properties for uneven timepoints
            @test result_uneven isa Dict
            @test all(k in keys(result_uneven) for k in [:s_pop_prior, :logσ_pop_prior, :logλ_prior])

            # Verify dimensions for uneven timepoints
            n_reps_uneven = length(unique(data_uneven.rep))
            times_per_rep = [length(unique(group.time)) for group in groupby(data_uneven, :rep)]
            expected_params = sum(times_per_rep .- 1)
            @test length(result_uneven[:s_pop_prior]) == expected_params
            @test length(result_uneven[:logσ_pop_prior]) == expected_params
        end

        # ========================================================================
        # Test naive_prior with data003_multienv.csv (multiple environments)
        # ========================================================================
        @testset "Multiple Environments Data (data003)" begin
            # Load test data
            data = CSV.read("data/data003_multienv.csv", DataFrame)

            # Test with environments
            result = BarBay.stats.naive_prior(data)

            # Test result properties
            @test result isa Dict
            @test all(k in keys(result) for k in [:s_pop_prior, :logσ_pop_prior, :logλ_prior])

            # Test dimensions with environments
            n_times = length(unique(data.time))
            n_envs = length(unique(data.env))
            n_barcodes = length(unique(data.barcode))

            # Test that population parameters account for environment transitions
            @test length(result[:s_pop_prior]) == n_times - 1
            @test length(result[:logσ_pop_prior]) == n_times - 1

            # Test proper initialization with environment changes
            env_changes = diff(data[sortperm(data.time), :env])
            @test !any(isnan.(result[:s_pop_prior]))
            @test !any(isnan.(result[:logσ_pop_prior]))
        end

        # ========================================================================
        # Test naive_prior with data004_multigen.csv (multiple genotypes)
        # ========================================================================
        @testset "Multiple Genotypes Data (data004)" begin
            # Load test data
            data = CSV.read("data/data004_multigen.csv", DataFrame)

            # Test with genotypes
            result = BarBay.stats.naive_prior(data)

            # Test result properties
            @test result isa Dict
            @test all(k in keys(result) for k in [:s_pop_prior, :logσ_pop_prior, :logλ_prior])

            # Test dimensions with genotypes
            n_times = length(unique(data.time))
            n_genotypes = length(unique(data.genotype))
            n_barcodes = length(unique(data.barcode))

            # Test that population parameters are computed correctly
            @test length(result[:s_pop_prior]) == n_times - 1
            @test length(result[:logσ_pop_prior]) == n_times - 1

            # Test that priors are reasonable for each genotype
            @test !any(isnan.(result[:s_pop_prior]))
            @test !any(isnan.(result[:logσ_pop_prior]))
        end

        # ========================================================================
        # Test error handling and edge cases
        # ========================================================================
        @testset "Error Handling and Edge Cases" begin
            # Load base data
            data = CSV.read("data/data001_single.csv", DataFrame)

            # Test with missing timepoints
            data_missing = data[2:end, :]
            @test_throws ErrorException BarBay.stats.naive_prior(data_missing)
        end
    end
end