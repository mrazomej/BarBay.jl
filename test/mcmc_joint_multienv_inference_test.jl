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
import MCMCChains

# Import basic math
import StatsBase
import Random

# Import library to save and load native julia objects
import JLD2

# Import library to perform Bayesian inference
import Turing
# Import AutoDiff backend
using ReverseDiff

# Import library to list files
import Glob

# Import plotting libraries
using CairoMakie
import ColorSchemes

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

# Define boolean to plot trajectories
plot_trajectories = true

##
println("Loading data...\n")
# Import data
data = CSV.read(
    "$(git_root())/test/data/data_example_multi-env.csv", DF.DataFrame
)

##


if plot_trajectories

    ##

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Plot trajectories
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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

    # List unique environments
    envs = unique(data.env)

    # Define environment-color dictionary
    colors = Dict(envs .=> ColorSchemes.Accent_3[1:length(envs)])

    # Define time-environment relation
    time_env = Matrix(unique(data[:, [:time, :env]]))

    # Loop through each time point
    for t = 1:size(time_env, 1)
        # Color plot background
        vspan!(
            ax,
            time_env[t, 1] - 1,
            time_env[t, 1],
            color=(colors[time_env[t, 2]], 0.25)
        )
    end # for

    # Plot mutant barcode frequency trajectories
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

    # Set axis limits
    xlims!(ax, -0.5, 5)

    fig

    ##

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Plot log-freq ratio trajectories
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    println("Plotting log-frequency ratio trajectories...\n")
    # Initialize figure
    fig = Figure(resolution=(450, 350))

    # Add axis
    ax = Axis(
        fig[1, 1],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
        title="log-frequency ratio trajectories"
    )

    # List unique environments
    envs = unique(data.env)

    # Define environment-color dictionary
    colors = Dict(envs .=> ColorSchemes.Accent_3)

    # Define time-environment relation
    time_env = Matrix(unique(data[:, [:time, :env]]))

    # Loop through each time point
    for t = 2:size(time_env, 1)
        # Color plot background
        vspan!(
            ax,
            time_env[t, 1] - 0.5,
            time_env[t, 1] + 0.5,
            color=(colors[time_env[t, 2]], 0.25)
        )
    end # for

    # Plot mutant barcode frequency trajectories
    BarBay.viz.logfreq_ratio_time_series!(
        ax,
        data[.!data.neutral, :];
        freq_col=:freq,
        alpha=0.25,
        linewidth=2
    )

    # Plot Neutral barcode trajectories
    BarBay.viz.logfreq_ratio_time_series!(
        ax,
        data[data.neutral, :];
        freq_col=:freq,
        color=ColorSchemes.Blues_9[end],
        alpha=0.9,
        linewidth=2
    )

    # Set axis limits
    xlims!(ax, 1, 5)
    ylims!(ax, -4, 4)

    fig

end # if
##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Setting sampling parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of barcodes
n_barcode = length(unique(data[data.neutral, :barcode])) + 1

# Define number of time points
n_time = length(unique(data.time))

# Initialize matrix to save λ_prior
λ_prior = ones((n_barcode + 1) * n_time, 2)
# Set λ_prior for barcodes
λ_prior[1:end-n_time, 1] .*= 3
λ_prior[1:end-n_time, 2] .*= 3
# Set λ_prior for grouped barcodes
λ_prior[end-n_time+1:end, 1] *= 6
λ_prior[end-n_time+1:end, 2] *= 3
# Define function parameters
param = Dict(
    :data => data,
    :n_walkers => n_walkers,
    :n_steps => n_steps,
    :outputname => "./output/single_mutant_multi-env_inference/" *
                   "chain_multi-env_$(lpad(n_steps, 2, "0"))steps_" *
                   "$(lpad(n_walkers, 2, "0"))walkers_bc",
    :model => BarBay.model.multienv_fitness_lognormal,
    :model_kwargs => Dict(
        :envs => ["G", "H", "N", "G", "H", "N"],
        :λ_prior => λ_prior,
    ),
    :sampler => Turing.NUTS(500, 0.65),
    :ensemble => Turing.MCMCSerial(),
    :multithread => true
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Sample posterior distribution
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Create output directory
if !isdir("./output/single_mutant_multi-env_inference/")
    mkdir("./output/single_mutant_multi-env_inference/")
end # if


# Run inference
println("Running Inference...")
@time BarBay.mcmc.mcmc_single_fitness(; param...)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Read single-mutant inferences
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# List files from group inference
files = sort(
    Glob.glob(
        "./output/single_mutant_multienv_inference/" *
        "chain_single_multienv_$(lpad(n_steps, 2, "0"))steps_" *
        "$(lpad(n_walkers, 2, "0"))walkers_bc*"
    )
)

# Extract barcode information from each file
bc_list = [
    parse(Int64, replace(split(f, "_")[end], "bc" => "", ".jld2" => ""))
    for f in files
]

# Initialize dictionary to store chains
chn_dict = Dict()
# Initialize dictionary to store tidy dataframes
df_dict = Dict()

# Loop through each file
for (bc, f) in zip(bc_list, files)
    # Add chain to dictionary
    setindex!(chn_dict, JLD2.load(f)["chain"], bc)
    # Add tidy dataframe to dictionary
    setindex!(df_dict, DF.DataFrame(chn_dict[bc]), bc)
end # for

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot population mean fitness ECDF
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of rows and columns
n_row, n_col = [2, 3]

# Initialize figure
fig = Figure(resolution=(300 * n_col, 300 * n_row))

# Add axis objects for each timepoint
axes = [
    Axis(
        fig[i, j],
        xlabel="population mean fitness (s̄ₜ)",
        ylabel="ecdf",
    ) for i = 1:n_row for j = 1:n_col
]

# Extract population mean fitness variable names
pop_mean_vars = MCMCChains.namesingroup(chn_dict[first(bc_list)], :s̲ₜ)

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 0.75, length(files)))

# Loop through population mean fitness variables
for (i, s) in enumerate(pop_mean_vars)
    # Loop through mutant barcodes
    for (j, bc) in enumerate(bc_list)
        # Plot full inference value
        ecdfplot!(axes[i], df_dict[bc][:, s], color=colors[j])
    end # for
    # Set x-axis limit
    # xlims!(axes[i], 0, 1.5)
end # for

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot log-frequency ratio PPC
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of rows and columns
n_row, n_col = [4, 4]
# Initialize figure
fig = Figure(resolution=(300 * n_col, 300 * n_row))

# Add GridLayout
gl = fig[1, 1] = GridLayout()

# Add axis objects
axes = [Axis(fig[i, j]) for i = 1:n_row for j = 1:n_col]

# Define dictionary with corresponding parameters for variables needed for the
# posterior predictive checks
param = Dict(
    :population_mean_fitness => :s̲ₜ,
    :population_std_fitness => :σ̲ₜ,
)
# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.68, 0.95, 0.995]

# Loop through elements
for (i, chn) in enumerate(collect(keys(chn_dict))[1:length(axes)])
    # Compute posterior predictive checks
    ppc_mat = BarBay.stats.logfreq_ratio_mean_ppc(
        chn_dict[chn], n_ppc; param=param
    )

    # Define colors
    colors = get(ColorSchemes.Blues_9, LinRange(0.5, 0.75, length(qs)))

    # Plot posterior predictive checks
    BarBay.viz.ppc_time_series!(
        axes[i], qs, ppc_mat; colors=colors
    )

    # Add plot for median (we use the 5 percentile to have a "thicker" line
    # showing the median)
    BarBay.viz.ppc_time_series!(
        axes[i], [0.05], ppc_mat; colors=ColorSchemes.Blues_9[end:end]
    )

    # Plot log-frequency ratio of neutrals
    BarBay.viz.logfreq_ratio_time_series!(
        axes[i],
        data[data.neutral, :];
        freq_col=:freq,
        color=:black,
        alpha=1.0,
        linewidth=2
    )

    # Add title
    axes[i].title = "bc $(chn)"

    # Hide axis decorations
    hidedecorations!(axes[i], grid=false)
end # for

# Add x-axis label
Label(fig[end, :, Bottom()], "time points", fontsize=20)
# Add y-axis label
Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=20)
# Add Plot title
Label(fig[0, 2:3], text="PPC neutral lineages", fontsize=20)
# Set row and col gaps
colgap!(gl, 10)
rowgap!(gl, 10)

fig
##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for barcodes
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of rows and columns
n_row, n_col = [4, 4]
# Initialize figure
fig = Figure(resolution=(300 * n_col, 300 * n_row))

# Add GridLayout
gl = fig[1, 1] = GridLayout()

# Add axis objects
axes = [Axis(fig[i, j]) for i = 1:n_row for j = 1:n_col]

# Define dictionary with corresponding parameters for variables needed for the
# posterior predictive checks
param = Dict(
    :population_mean_fitness => :s̲ₜ,
    :mutant_mean_fitness => :s̲⁽ᵐ⁾,
    :mutant_std_fitness => :σ̲⁽ᵐ⁾,
)

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.68, 0.95]

# Define environments
envs = ["G", "H", "N", "G", "H", "N"]

# Define colors
col_dict = Dict(
    "G" => ColorSchemes.Greens_9,
    "H" => ColorSchemes.Oranges_9,
    "N" => ColorSchemes.Purples_9,
)

# Loop through elements
for (i, chn) in enumerate(collect(keys(chn_dict))[1:length(axes)])
    # Compute posterior predictive checks
    ppc_mat = BarBay.stats.logfreq_ratio_multienv_ppc(
        chn_dict[chn], n_ppc, envs; param=param
    )

    # Add first environment inference
    BarBay.viz.ppc_time_series!(
        axes[i],
        qs,
        ppc_mat[:, [1, 1]];
        time=[0.75, 1],
        colors=get(col_dict[envs[1]], LinRange(0.5, 0.75, length(qs)))
    )

    # Loop through environments
    for j = 2:size(ppc_mat, 2)
        # Define colors
        colors = get(col_dict[envs[j]], LinRange(0.5, 0.75, length(qs)))

        # Plot posterior predictive checks
        BarBay.viz.ppc_time_series!(
            axes[i], qs, ppc_mat[:, j-1:j]; time=[j - 1, j], colors=colors
        )

        # Add plot for median (we use the 5 percentile to have a "thicker" line
        # showing the median)
        BarBay.viz.ppc_time_series!(
            axes[i],
            [0.05],
            ppc_mat[:, j-1:j];
            time=[j - 1, j],
            colors=col_dict[envs[j]][end:end]
        )
    end # for


    # Plot log-frequency ratio of neutrals
    BarBay.viz.logfreq_ratio_time_series!(
        axes[i],
        data[data.barcode.==chn, :];
        freq_col=:freq,
        color=:black,
        alpha=1.0,
        linewidth=2,
        markersize=10
    )

    # Add title
    axes[i].title = "bc $(chn)"

    # Hide axis decorations
    hidedecorations!(axes[i], grid=false)
end # for

# Add x-axis label
Label(fig[end, :, Bottom()], "time points", fontsize=20)
# Add y-axis label
Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=20)
# Add Plot title
Label(fig[0, 2:3], text="PPC mutant lineages", fontsize=20)
# Set row and col gaps
colgap!(gl, 10)
rowgap!(gl, 10)

fig