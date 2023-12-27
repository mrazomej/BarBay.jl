using BarBay
using Test
using CSV
using StatsBase

@testset "data_to_array tests" begin
    # Load the minimal dataset
    data = CSV.read("./test/data/data001_single.csv", DataFrame)

    # Call the function and store the result
    result = BarBay.utils.data_to_arrays(data)

    # Check that the result is a Dict
    @test typeof(result) <: Dict

    # Check that the Dict has the correct keys
    expected_keys = [
        :bc_count,
        :bc_total,
        :n_neutral,
        :n_bc,
        :bc_ids,
        :neutral_ids,
        :envs,
        :n_env,
        :n_rep,
        :n_time,
        :genotypes,
        :n_geno
    ]
    @test all([k in keys(result) for k in expected_keys])
end

@testset "naive_prior tests" begin
    # Load the minimal dataset
    data = CSV.read("./test/data/data001_single.csv", DataFrame)

    # Call the function and store the result
    result = BarBay.stats.naive_prior(data)

    # Check that the result is a Dict
    @test typeof(result) <: Dict

    # Check that the Dict has the correct keys
    expected_keys = [:s_pop_prior, :logσ_pop_prior, :logλ_prior]
    @test all([k in keys(result) for k in expected_keys])
end

@testset "compute_mean_fitness tests" begin
    # Load the minimal dataset
    data = CSV.read("./test/data/data001_single.csv", DataFrame)

    # Call the function and store the result
    result = BarBay.stats.naive_fitness(data)

    # Check that the result is a DataFrame
    @test typeof(result) <: DataFrame

    # Check that the DataFrame has the correct columns
    @test all([col in Symbol.(names(result)) for col in [:barcode, :fitness]])
end