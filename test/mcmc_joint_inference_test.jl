# Activate environment
@load_pkg(".")

# Import package to revise package
import Revise

# Import library package
import BayesFitness

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import statistics libraries
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

# Define sampling hyperparameters
n_steps = 1000
n_walkers = 4

# Define whether or not to generate diagnostic plots
gen_plots = true

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...\n")
# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load population mean fitness inference to select priors on log-likelhood
# errors
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Load chain into memory
chn = JLD2.load(
    "./output/chain_popmean_fitness_1000steps_04walkers.jld2"
)["chain"]

println("Fitting parametric distributions to relevant parameters...\n")

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

# Extract distribution parameters into matrix
pop_mean_param = hcat(
    first.(Distributions.params.(pop_mean)),
    last.(Distributions.params.(pop_mean))
)
pop_std_param = hcat(
    first.(Distributions.params.(pop_std)),
    last.(Distributions.params.(pop_std))
)

# Manual inspection of results to select priors
s_pop_prior = [1.0, 0.5]
σ_pop_prior = [-1, 0.3]
σ_mut_prior = [-1, 0.3]

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load barcode frequency inference to define priors
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

chn = JLD2.load(
    "./output/chain_freq_100steps_04walkers.jld2"
)["chain"]

# Select variables for population mean fitness and associated variance
var_name = MCMCChains.namesingroup(chn, :Λ̲̲)

# Fit normal distributions to population mean fitness
freq_dist = Distributions.fit.(
    Ref(Distributions.LogNormal), [vec(chn[x]) for x in var_name]
)

# Extract parameters into matrix
λ_prior = hcat(
    first.(Distributions.params.(freq_dist)),
    last.(Distributions.params.(freq_dist))
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define parameters for Inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define function parameters

param = Dict(
    :data => data,
    :n_walkers => n_walkers,
    :n_steps => n_steps,
    :outputname => "./output/chain_joint_priors_$(n_steps)steps_$(lpad(n_walkers, 2, "0"))walkers",
    :model => BayesFitness.model.fitness_lognormal,
    :model_kwargs => Dict(
        :s_pop_prior => s_pop_prior,
        :σ_pop_prior => σ_pop_prior,
        :σ_mut_prior => σ_mut_prior,
        :λ_prior => λ_prior,
    ),
    :sampler => Turing.DynamicNUTS(),
    :ensemble => Turing.MCMCThreads(),
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Run MCMC sampling
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Run inference
println("Running Inference...")
@time BayesFitness.mcmc.mcmc_joint_fitness(; param...)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load resulting chain
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

if gen_plots
    # Load chain and mutant order into memory
    ids, chn = values(JLD2.load("$(param[:outputname]).jld2"))

    # Find columns with mutant fitness values and error
    fit_vars = MCMCChains.namesingroup(chn, :s̲⁽ᵐ⁾)
    σ_vars = MCMCChains.namesingroup(chn, :σ̲⁽ᵐ⁾)

    # Convert to tidy dataframe
    df = DF.DataFrame(chn)

    # Rename columns to mutant names
    DF.rename!(df, Dict(zip(string.(fit_vars), "sbc" .* string.(ids))))
    DF.rename!(df, Dict(zip(string.(σ_vars), "σbc" .* string.(ids))))

end # if

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot frequency trajectories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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
    BayesFitness.viz.bc_time_series!(
        ax,
        data[.!data.neutral, :];
        quant_col=:freq,
        zero_lim=1E-7,
        zero_label="extinct",
        alpha=0.25,
        linewidth=2
    )

    # Plot Neutral barcode trajectories
    BayesFitness.viz.bc_time_series!(
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot log-frequency ratio trajectories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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
    BayesFitness.viz.logfreq_ratio_time_series!(
        ax,
        data[.!data.neutral, :];
        freq_col=:freq,
        alpha=0.25,
        linewidth=2
    )

    # Plot log-frequency ratio of neutrals
    BayesFitness.viz.logfreq_ratio_time_series!(
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot population mean fitness trace and density
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

if gen_plots
    println("Plotting traces and densities...\n")

    # Initialize figure
    fig = Figure(resolution=(600, 800))

    # Locate mean fitness variables
    var_names = vcat(MCMCChains.namesingroup.(Ref(chn), [:s̲ₜ, :σ̲ₜ])...)

    # Generate mcmc_trace_density! plot
    BayesFitness.viz.mcmc_trace_density!(fig, chn[var_names]; alpha=0.5)

    # save("../docs/src/figs/fig03.svg", fig)

    fig

end # if

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute posterior predictive checks for neutral lineages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

if gen_plots
    println("Generating posterior predictive checks for neutrals...\n")
    # Define number of posterior predictive check samples
    n_ppc = 500

    # Define dictionary with corresponding parameters for variables needed for the
    # posterior predictive checks
    param = Dict(
        :population_mean_fitness => :s̲ₜ,
        :population_std_fitness => :σ̲ₜ,
    )

    # Compute posterior predictive checks
    ppc_mat = BayesFitness.stats.logfreq_ratio_mean_ppc(
        chn, n_ppc; param=param
    )

end # if

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for neutral lineages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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
    BayesFitness.viz.ppc_time_series!(
        ax, qs, ppc_mat; colors=colors
    )

    # Plot log-frequency ratio of neutrals
    BayesFitness.viz.logfreq_ratio_time_series!(
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for a few selected barcodes
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.025, 0.3375, 0.675]

# Define number of rows and columns
n_row, n_col = [4, 4]

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 1, length(qs)))

if gen_plots

    # Initialize figure
    fig = Figure(resolution=(300 * n_col, 300 * n_row))

    # Add axis
    axes = [Axis(fig[i, j]) for i = 1:n_row for j = 1:n_col]

    # List example barcodes to plot
    bc_plot = ids[1:(n_row*n_col)]

    # Initialize plot counter
    counter = 1
    # Loop through rows
    for row in 1:n_row
        # Loop through columns
        for col in 1:n_col
            # Extract data
            data_bc = DF.sort(data[data.barcode.==bc_plot[counter], :], :time)

            # Define dictionary with corresponding parameters for variables needed
            # for the posterior predictive checks
            param = Dict(
                :mutant_mean_fitness => Symbol("sbc$(bc_plot[counter])"),
                :mutant_std_fitness => Symbol("σbc$(bc_plot[counter])"),
                :population_mean_fitness => :s̲ₜ,
            )
            # Compute posterior predictive checks
            ppc_mat = BayesFitness.stats.logfreq_ratio_mutant_ppc(
                df, n_ppc; param=param
            )
            # Plot posterior predictive checks
            BayesFitness.viz.ppc_time_series!(
                axes[counter], qs, ppc_mat; colors=colors
            )

            # Plot log-frequency ratio of neutrals
            BayesFitness.viz.logfreq_ratio_time_series!(
                axes[counter],
                data_bc,
                freq_col=:freq,
                color=:black,
                alpha=1.0,
                linewidth=3,
                markersize=15
            )

            # Add title
            axes[counter].title = "bc $(bc_plot[counter])"
            axes[counter].titlesize = 12

            # Hide axis decorations
            hidedecorations!(axes[counter], grid=false)

            # Update counter
            counter += 1
        end  # for
    end # for

    # Add x-axis label
    Label(fig[end, :, Bottom()], "time points", fontsize=20)
    # Add y-axis label
    Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=20)

    fig
end # if