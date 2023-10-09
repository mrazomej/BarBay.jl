##

# Activate environment
@load_pkg(".")

# Import package to revise package
import Revise

# Import library package
import BarBay

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import statistics-related packages
import Random
import Distributions
import StatsBase

# Import library to save and load native julia objects
import JLD2

# Import plotting libraries
using CairoMakie
import ColorSchemes

# Import library to perform Bayesian inference
import Turing
import MCMCChains
import DynamicHMC

# Import AutoDiff backend
using ReverseDiff

# Import Memoization
using Memoization

Random.seed!(42)

##

# Set AutoDiff backend
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

##

# Define number of steps
n_steps = 1_000
# Define number of walkers
n_walkers = 4

# Define whether plots should be generated or not
gen_plots = true

##

println("Loading data...\n")
# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

##

if gen_plots
    # Plot trajectories

    println("Plotting frequency trajectories...\n")
    # Initialize figure
    fig = Figure(resolution=(450, 350))

    # Add axis
    ax = Axis(
        fig[1, 1],
        xlabel="time point",
        ylabel="barcode frequency",
        yscale=log10,
        title="frequency trajectories"
    )

    # Plot Mutant barcode trajectories
    BarBay.viz.bc_time_series!(
        ax,
        data[.!data.neutral, :];
        quant_col=:freq,
        zero_lim=1E-7,
        zero_label="extinct",
        alpha=0.25,
        linewidth=2
    )

    # Plot Neutral barcode trajectories
    BarBay.viz.bc_time_series!(
        ax,
        data[data.neutral, :];
        quant_col=:freq,
        zero_lim=1E-7,
        color=ColorSchemes.Blues_9[end],
        alpha=0.9,
        linewidth=2
    )

    # save("../docs/src/figs/fig01.svg", fig)

    fig

end # if
##

if gen_plots
    println("Plotting log-frequency ratio trajectories...\n")
    # Initialize figure
    fig = Figure(resolution=(450, 350))

    # Add axis
    ax = Axis(
        fig[1, 1],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
        title="log-frequency ratio"
    )

    # Plot log-frequency ratio of mutants
    BarBay.viz.logfreq_ratio_time_series!(
        ax,
        data[.!data.neutral, :];
        freq_col=:freq,
        alpha=0.25,
        linewidth=2
    )

    # Plot log-frequency ratio of neutrals
    BarBay.viz.logfreq_ratio_time_series!(
        ax,
        data[data.neutral, :];
        freq_col=:freq,
        color=ColorSchemes.Blues_9[end],
        alpha=1.0,
        linewidth=2
    )

    # save("../docs/src/figs/fig02.svg", fig)

    fig
end # if

##

# Define function parameters

param = Dict(
    :data => data,
    :n_walkers => n_walkers,
    :n_steps => n_steps,
    :outputname => "./output/chain_popmean_fitness_$(n_steps)steps_$(lpad(n_walkers, 2, "0"))walkers",
    :model => BarBay.model.neutrals_lognormal,
    :sampler => Turing.DynamicNUTS(),
    :ensemble => Turing.MCMCThreads(),
)

##

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Run inference
println("Running Inference...")
BarBay.mcmc.mcmc_popmean_fitness(; param...)

##

# Load chain into memory
chn = JLD2.load("$(param[:outputname]).jld2")["chain"]

if gen_plots
    println("Plotting trances and densities...\n")

    # Select variables for population mean fitness and associated variance
    var_name = vcat(MCMCChains.namesingroup.(Ref(chn), [:s̲ₜ, :σ̲ₜ])...)

    # Initialize figure
    fig = Figure(resolution=(600, 800))

    # Generate mcmc_trace_density! plot
    BarBay.viz.mcmc_trace_density!(fig, chn[var_name]; alpha=0.5)

    # save("../docs/src/figs/fig03.svg", fig)

    fig

end # if

##

if gen_plots
    println("Generating posterior predictive checks...\n")
    # Define number of posterior predictive check samples
    n_ppc = 500

    # Define dictionary with corresponding parameters for variables needed for the
    # posterior predictive checks
    param = Dict(
        :population_mean_fitness => :s̲ₜ,
        :population_std_fitness => :σ̲ₜ,
    )

    # Compute posterior predictive checks
    ppc_mat = BarBay.stats.logfreq_ratio_mean_ppc(
        chn, n_ppc; param=param
    )

end # if

##

if gen_plots
    println("Plotting posterior predictive checks...")

    # Initialize figure
    fig = Figure(resolution=(450, 350))

    # Add axis
    ax = Axis(
        fig[1, 1],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
        title="log-frequency ratio PPC"
    )

    # Define quantiles to compute
    qs = [0.05, 0.68, 0.95]

    # Define colors
    colors = get(ColorSchemes.Blues_9, LinRange(0.25, 1, length(qs)))

    # Plot posterior predictive checks
    BarBay.viz.ppc_time_series!(
        ax, qs, ppc_mat; colors=colors
    )

    # Plot log-frequency ratio of neutrals
    BarBay.viz.logfreq_ratio_time_series!(
        ax,
        data[data.neutral, :];
        freq_col=:freq,
        color=:black,
        alpha=1.0,
        linewidth=2,
        markersize=8
    )

    # save("../docs/src/figs/fig04.svg", fig)

    fig
end # if

##

if gen_plots
    println("Fitting parametric distributions to relevant parameters")

    # Select variables for population mean fitness and associated variance
    var_name = MCMCChains.namesingroup.(Ref(chn), [:s̲ₜ, :σ̲ₜ])

    # Fit normal distributions to population mean fitness
    pop_mean = Distributions.fit.(
        Ref(Distributions.Normal), [vec(chn[x]) for x in var_name[1]]
    )

    # Fit lognormal distributions to associated error
    pop_std = Distributions.fit.(
        Ref(Distributions.LogNormal), [vec(chn[x]) for x in var_name[2]]
    )

end # if

##

if gen_plots
    println("Plotting ECDF plots for population mean fitness")

    # Initialize figure
    fig = Figure(resolution=(600, 600))

    # Add axis objects for each timepoint
    axes = [
        Axis(
            fig[i, j],
            xlabel="population mean fitness (s̄ₜ)",
            ylabel="ecdf",
        ) for i = 1:2 for j = 1:2
    ]

    # Loop through time points
    for (i, var) in enumerate(var_name[1])
        # Plot ECDF
        BarBay.viz.mcmc_fitdist_cdf!(
            axes[i],
            vec(chn[var])[:],
            pop_mean[i]
        )

        axes[i].title = "timepoint $(i)"
    end # for

    fig
end # if


##

if gen_plots
    println("Plotting ECDF plots for population mean fitness associated variance")

    # Initialize figure
    fig = Figure(resolution=(600, 600))

    # Add axis objects for each timepoint
    axes = [
        Axis(
            fig[i, j],
            xlabel="log-likelihood error (σₜ)",
            ylabel="ecdf",
        ) for i = 1:2 for j = 1:2
    ]

    # Loop through time points
    for (i, var) in enumerate(var_name[2])
        # Plot ECDF
        BarBay.viz.mcmc_fitdist_cdf!(
            axes[i],
            vec(chn[var])[:],
            pop_std[i]
        )

        axes[i].title = "timepoint $(i)"
    end # for

    fig
end # if

##