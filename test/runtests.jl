using BarBay
using Test
using CSV
using DataFrames
using StatsBase
using Turing
using AdvancedVI
using ReverseDiff

println("Testing utility functions...")
# ---------------------------------------------------------------------------- #

@testset "data_to_array tests" begin
    # Load the minimal dataset
    data = CSV.read("data/data001_single.csv", DataFrame)

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

println("\nTesting stats functions...\n")

# ---------------------------------------------------------------------------- #

@testset "naive_prior tests" begin
    # Load the minimal dataset
    data = CSV.read("data/data001_single.csv", DataFrame)

    # Call the function and store the result
    result = BarBay.stats.naive_prior(data)

    # Check that the result is a Dict
    @test typeof(result) <: Dict

    # Check that the Dict has the correct keys
    expected_keys = [:s_pop_prior, :logσ_pop_prior, :logλ_prior]
    @test all([k in keys(result) for k in expected_keys])
end

@testset "naive_fitness tests" begin
    # Load the minimal dataset
    data = CSV.read("data/data001_single.csv", DataFrame)

    # Call the function and store the result
    result = BarBay.stats.naive_fitness(data)

    # Check that the result is a DataFrame
    @test typeof(result) <: DataFrame

    # Check that the DataFrame has the correct columns
    @test all([col in Symbol.(names(result)) for col in [:barcode, :fitness]])
end

# ---------------------------------------------------------------------------- #

println("\nTesting model functions...\n")

@testset "fitness_normal tests" begin
    # Load the minimal dataset
    data = CSV.read("data/data001_single.csv", DataFrame)

    # Call the function and store the result
    data_dict = BarBay.utils.data_to_arrays(data)

    # Define model
    model = BarBay.model.fitness_normal(
        data_dict[:bc_count],
        data_dict[:bc_total],
        data_dict[:n_neutral],
        data_dict[:n_bc];
    )

    # Check that the result is a Model
    @test typeof(model) <: Turing.AbstractMCMC.AbstractModel
end

# ---------------------------------------------------------------------------- #

@testset "replicate_fitness_normal tests" begin
    # Load the minimal dataset
    data = CSV.read("data/data002_hier-rep.csv", DataFrame)

    # Call the function and store the result
    data_dict = BarBay.utils.data_to_arrays(data; rep_col=:rep)

    # Define model
    model = BarBay.model.replicate_fitness_normal(
        data_dict[:bc_count],
        data_dict[:bc_total],
        data_dict[:n_neutral],
        data_dict[:n_bc];
    )

    # Check that the result is a Model
    @test typeof(model) <: Turing.AbstractMCMC.AbstractModel

    # Remove last time point from last replicate
    data = data[
        (data.rep.!=maximum(data.rep)).|(data.time.!=maximum(data.time)),
        :]

    # Call the function and store the result
    data_dict = BarBay.utils.data_to_arrays(data; rep_col=:rep)

    # Define model
    model = BarBay.model.replicate_fitness_normal(
        data_dict[:bc_count],
        data_dict[:bc_total],
        data_dict[:n_neutral],
        data_dict[:n_bc];
    )

    # Check that the result is a Model
    @test typeof(model) <: Turing.AbstractMCMC.AbstractModel
end

# ---------------------------------------------------------------------------- #

@testset "multienv_fitness_normal tests" begin
    # Load the minimal dataset
    data = CSV.read("data/data003_multienv.csv", DataFrame)

    # Call the function and store the result
    data_dict = BarBay.utils.data_to_arrays(data; env_col=:env)

    # Define model
    model = BarBay.model.multienv_fitness_normal(
        data_dict[:bc_count],
        data_dict[:bc_total],
        data_dict[:n_neutral],
        data_dict[:n_bc];
        Dict(:envs => data_dict[:envs])...
    )

    # Check that the result is a Model
    @test typeof(model) <: Turing.AbstractMCMC.AbstractModel
end

# ---------------------------------------------------------------------------- #

@testset "genotype_fitness_normal tests" begin
    # Load the minimal dataset
    data = CSV.read("data/data004_multigen.csv", DataFrame)

    # Call the function and store the result
    data_dict = BarBay.utils.data_to_arrays(data; genotype_col=:genotype)

    # Define model
    model = BarBay.model.genotype_fitness_normal(
        data_dict[:bc_count],
        data_dict[:bc_total],
        data_dict[:n_neutral],
        data_dict[:n_bc];
        Dict(:genotypes => data_dict[:genotypes])...
    )

    # Check that the result is a Model
    @test typeof(model) <: Turing.AbstractMCMC.AbstractModel
end

# ---------------------------------------------------------------------------- #

println("\nTesting variational inference functions...\n")

@testset "ADVI fitness_normal tests" begin
    # Load the minimal dataset
    data = CSV.read("data/data001_single.csv", DataFrame)

    # Define model
    model = BarBay.model.fitness_normal

    println("ReverseDiffAD")
    # Call the function and store the result
    df = BarBay.vi.advi(
        data=data,
        model=model,
        advi=Turing.ADVI{AdvancedVI.ReverseDiffAD{false}}(1, 1)
    )

    # Check that the result is a Turing.VI
    @test typeof(df) <: DataFrames.DataFrame

    println("ForwardDiffAD")
    # Call the function and store the result
    df = BarBay.vi.advi(
        data=data,
        model=model,
        advi=Turing.ADVI{AdvancedVI.ForwardDiffAD{0}}(1, 1)
    )

    # Check that the result is a Turing.VI
    @test typeof(df) <: DataFrames.DataFrame
end

# ---------------------------------------------------------------------------- #

@testset "ADVI multienv_fitness_normal tests" begin
    # Load the minimal dataset
    data = CSV.read("data/data003_multienv.csv", DataFrame)

    # Define model
    model = BarBay.model.multienv_fitness_normal

    println("ReverseDiffAD")
    # # Call the function and store the result
    df = BarBay.vi.advi(
        data=data,
        model=model,
        env_col=:env,
        advi=Turing.ADVI{AdvancedVI.ReverseDiffAD{false}}(1, 1)
    )

    # Check that the result is a Turing.VI
    @test typeof(df) <: DataFrames.DataFrame

    println("ForwardDiffAD")
    # Call the function and store the result
    df = BarBay.vi.advi(
        data=data,
        model=model,
        env_col=:env,
        advi=Turing.ADVI{AdvancedVI.ForwardDiffAD{0}}(1, 1)
    )

    # Check that the result is a Turing.VI
    @test typeof(df) <: DataFrames.DataFrame
end

# ---------------------------------------------------------------------------- #

@testset "ADVI replicate_fitness_normal tests" begin
    # Load the minimal dataset
    data = CSV.read("data/data002_hier-rep.csv", DataFrame)

    # Define model
    model = BarBay.model.replicate_fitness_normal

    println("ReverseDiffAD")
    # Call the function and store the result
    df = BarBay.vi.advi(
        data=data,
        model=model,
        rep_col=:rep,
        advi=Turing.ADVI{AdvancedVI.ReverseDiffAD{false}}(1, 1)
    )

    # Check that the result is a Turing.VI
    @test typeof(df) <: DataFrames.DataFrame

    println("ForwardDiffAD")
    # Call the function and store the result
    df = BarBay.vi.advi(
        data=data,
        model=model,
        rep_col=:rep,
        advi=Turing.ADVI{AdvancedVI.ForwardDiffAD{0}}(1, 1)
    )

    # Check that the result is a Turing.VI
    @test typeof(df) <: DataFrames.DataFrame

    # Remove last time point from last replicate
    data = data[
        (data.rep.!=maximum(data.rep)).|(data.time.!=maximum(data.time)),
        :]

    println("ReverseDiffAD")
    # Call the function and store the result
    df = BarBay.vi.advi(
        data=data,
        model=model,
        rep_col=:rep,
        advi=Turing.ADVI{AdvancedVI.ReverseDiffAD{false}}(1, 1)
    )

    # Check that the result is a Turing.VI
    @test typeof(df) <: DataFrames.DataFrame

    println("ForwardDiffAD")
    # Call the function and store the result
    df = BarBay.vi.advi(
        data=data,
        model=model,
        rep_col=:rep,
        advi=Turing.ADVI{AdvancedVI.ForwardDiffAD{0}}(1, 1)
    )

    # Check that the result is a Turing.VI
    @test typeof(df) <: DataFrames.DataFrame
end

# ---------------------------------------------------------------------------- #

@testset "ADVI genotype_fitness_normal tests" begin
    # Load the minimal dataset
    data = CSV.read("data/data004_multigen.csv", DataFrame)

    # Define model
    model = BarBay.model.genotype_fitness_normal

    println("ReverseDiffAD")
    # # Call the function and store the result
    df = BarBay.vi.advi(
        data=data,
        model=model,
        genotype_col=:genotype,
        advi=Turing.ADVI{AdvancedVI.ReverseDiffAD{false}}(1, 1)
    )

    # Check that the result is a Turing.VI
    @test typeof(df) <: DataFrames.DataFrame

    println("ForwardDiffAD")
    # Call the function and store the result
    df = BarBay.vi.advi(
        data=data,
        model=model,
        genotype_col=:genotype,
        advi=Turing.ADVI{AdvancedVI.ForwardDiffAD{0}}(1, 1)
    )

    # Check that the result is a Turing.VI
    @test typeof(df) <: DataFrames.DataFrame
end
