#  Activate environment
@load_pkg(".")

# Import package to revise package
import Revise

# Import library package
import BayesFitness

# Import basic math
import StatsBase

# Import libraries to manipulate data
import DataFrames as DF
import CSV
import MCMCChains

# Import library to save and load native julia objects
import JLD2

# Import library to list files
import Glob

# Import plotting libraries
using CairoMakie
import ColorSchemes

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Read data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Read full inference MCMC chain
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Import full joint inference
chain_joint, mut_joint = values(
    JLD2.load("./output/data_example_01_1000steps_04walkers.jld2")
    # JLD2.load("./output/data_example_02_1000steps_04walkers_rmT0.jld2")
)

# Find columns with mutant fitness values and error
fit_vars = MCMCChains.namesingroup(chain_joint, :s̲⁽ᵐ⁾)
σ_vars = MCMCChains.namesingroup(chain_joint, :σ̲⁽ᵐ⁾)

# Convert to tidy dataframe
df_joint = DF.DataFrame(chain_joint)

# Rename columns to mutant names
DF.rename!(df_joint, Dict(zip(string.(fit_vars), "sbc" .* string.(mut_joint))))
DF.rename!(df_joint, Dict(zip(string.(σ_vars), "σbc" .* string.(mut_joint))))

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Read single-mutant inferences
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# List files from group inference
files = sort(
    Glob.glob("./output/data_example_01_*_1000steps_04walkers_bc*.jld2")
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

# Add full inference to dictionaries
setindex!(chn_dict, chain_joint, "joint")
setindex!(df_dict, df_joint, "joint")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot population mean fitness ECDF
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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

# Extract population mean fitness variable names
pop_mean_vars = MCMCChains.namesingroup(chain_joint, :s̲ₜ)

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 0.75, length(files)))

# Loop through population mean fitness variables
for (i, s) in enumerate(pop_mean_vars)
    # Plot full inference value
    ecdfplot!(axes[i], df_joint[:, s], color=:black)
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
# Plot σₜ ECDF
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(600, 600))

# Add axis objects for each timepoint
axes = [
    Axis(
        fig[i, j],
        xlabel="log-normal likelihood error (σₜ)",
        ylabel="ecdf",
    ) for i = 1:2 for j = 1:2
]

# Extract population mean fitness variable names
pop_std_vars = MCMCChains.namesingroup(chain_joint, :σ̲ₜ)

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 0.75, length(files)))

# Loop through population mean fitness variables
for (i, s) in enumerate(pop_std_vars)
    # Plot full inference value
    ecdfplot!(axes[i], df_joint[:, s], label="full", color=:black)
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

# [StatsBase.median(chn[MCMCChains.namesingroup(chn, "Λ̲̲")[end]]) for chn in chain_groups]

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
axes = [Axis(fig[i, j], aspect=AxisAspect(1)) for i = 1:n_row for j = 1:n_col]

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

# Define colors
col = [ColorSchemes.Purples_9; repeat([ColorSchemes.Blues_9], length(axes) - 1)]

# Loop through elements
for (i, chn) in enumerate([["joint"]; collect(keys(chn_dict))[2:length(axes)]])
    # Compute posterior predictive checks
    ppc_mat = BayesFitness.stats.logfreq_ratio_mean_ppc(
        chn_dict[chn], n_ppc; param=param
    )

    # Define colors
    colors = get(col[i], LinRange(0.5, 0.75, length(qs)))

    # Plot posterior predictive checks
    BayesFitness.viz.ppc_time_series!(
        axes[i], qs, ppc_mat; colors=colors
    )

    # Add plot for median (we use the 5 percentile to have a "thicker" line
    # showing the median)
    BayesFitness.viz.ppc_time_series!(
        axes[i], [0.05], ppc_mat; colors=col[i][end:end]
    )

    # Plot log-frequency ratio of neutrals
    BayesFitness.viz.logfreq_ratio_time_series!(
        axes[i],
        data[data.neutral, :];
        freq_col=:freq,
        color=:black,
        alpha=1.0,
        linewidth=2
    )

    # Set y-axis limit
    ylims!(axes[i], -2.5, 0.5)

    # Add title
    if chn == "full"
        axes[i].title = "$(chn)"
    else
        axes[i].title = "bc $(chn)"
    end # if

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
# Compute summary statistics for each barcode
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize dataframe to save summary statistics for single barcode inference
df_single_summary = DF.DataFrame()

# Initialize dataframe to save summary statistics for joint inference
df_joint_summary = DF.DataFrame()

# Find barcode variables
var_names = names(df_joint)[occursin.("sbc", names(df_joint))]

# Define percentiles to include
per = [2.5, 97.5, 16, 84]

# Loop through files
for bc in bc_list
    # Extract fitness chain
    fitness_single = @view df_dict[bc][:, :s⁽ᵐ⁾]
    fitness_joint = @view df_joint[:, Symbol("sbc$(bc)")]

    ## FILTER highest percentiles ##
    fitness_single = fitness_single[
        (fitness_single.≥StatsBase.percentile(fitness_single, 5)).&(fitness_single.≤StatsBase.percentile(fitness_single, 95))
    ]
    fitness_joint = fitness_joint[
        (fitness_joint.≥StatsBase.percentile(fitness_joint, 5)).&(fitness_joint.≤StatsBase.percentile(fitness_joint, 95))
    ]

    # Compute summary statistics
    fitness_single_summary = Dict(
        :mean => StatsBase.mean(fitness_single),
        :median => StatsBase.median(fitness_single),
        :std => StatsBase.std(fitness_single),
        :var => StatsBase.var(fitness_single),
        :skewness => StatsBase.skewness(fitness_single),
        :kurtosis => StatsBase.kurtosis(fitness_single),
    )
    fitness_joint_summary = Dict(
        :mean => StatsBase.mean(fitness_joint),
        :median => StatsBase.median(fitness_joint),
        :std => StatsBase.std(fitness_joint),
        :var => StatsBase.var(fitness_joint),
        :skewness => StatsBase.skewness(fitness_joint),
        :kurtosis => StatsBase.kurtosis(fitness_joint),
    )

    # Loop through percentiles
    for p in per
        setindex!(
            fitness_single_summary,
            StatsBase.percentile(fitness_single, p),
            Symbol("$p")
        )
        setindex!(
            fitness_joint_summary,
            StatsBase.percentile(fitness_joint, p),
            Symbol("$p")
        )
    end # for

    # Convert to dataframe
    df_single_fitness = DF.DataFrame(fitness_single_summary)
    df_joint_fitness = DF.DataFrame(fitness_joint_summary)
    # Add barcode
    df_single_fitness[!, :barcode] .= bc
    df_joint_fitness[!, :barcode] .= bc
    # Append to dataframe
    DF.append!(df_single_summary, df_single_fitness)
    DF.append!(df_joint_summary, df_joint_fitness)
end # for

# Sort by barcode
DF.sort!(df_single_summary, :barcode)
DF.sort!(df_joint_summary, :barcode)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare median fitness 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(400, 400))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="joint inference fitness s⁽ᵐ⁾",
    ylabel="single inference fitness s⁽ᵐ⁾",
)

# Plot identity line
lines!(ax, [-1, 1.5], [-1, 1.5], linestyle=:dash, color="black")

# Plot x-axis error bars
errorbars!(
    ax,
    df_joint_summary.median,
    df_single_summary.median,
    abs.(df_joint_summary.median .- df_joint_summary[:, Symbol("16.0")]),
    abs.(df_joint_summary.median .- df_joint_summary[:, Symbol("84.0")]),
    direction=:x,
    linewidth=1.5,
    color=(:gray, 0.5)
)
# Plot y-axis error bars
errorbars!(
    ax,
    df_joint_summary.median,
    df_single_summary.median,
    abs.(df_single_summary.median .- df_single_summary[:, Symbol("16.0")]),
    abs.(df_single_summary.median .- df_single_summary[:, Symbol("84.0")]),
    direction=:y,
    linewidth=1.5,
    color=(:gray, 0.5)
)

# Plot fitness values
scatter!(ax, df_joint_summary.median, df_single_summary.median, markersize=5)


fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for barcodes
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675]

# Define number of rows and columns
n_row, n_col = [3, 3]

# Initialize figure
fig = Figure(resolution=(300 * n_col, 300 * n_row))

# List example barcodes to plot
bc_plot = [151; bc_list[1:(n_row*n_col)-1]]

# Initialize plot counter
counter = 1
# Loop through rows
for row in 1:n_row
    # Loop through columns
    for col in 1:n_col
        # Add GridLayout
        gl = fig[row, col] = GridLayout()
        # Add axis
        axes = [Axis(gl[i, 1:6]) for i = 1:2]

        # Extract data
        data_bc = DF.sort(data[data.barcode.==bc_plot[counter], :], :time)

        ## === Joint inference === ##
        # Define colors
        colors = get(ColorSchemes.Purples_9, LinRange(0.5, 0.75, length(qs)))

        # Define dictionary with corresponding parameters for variables needed
        # for the posterior predictive checks
        param = Dict(
            :mutant_mean_fitness => Symbol("sbc$(bc_plot[counter])"),
            :mutant_std_fitness => Symbol("σbc$(bc_plot[counter])"),
            :population_mean_fitness => :s̲ₜ,
        )
        # Compute posterior predictive checks
        ppc_mat = BayesFitness.stats.logfreq_ratio_mutant_ppc(
            df_dict["joint"], n_ppc; param=param
        )
        # Plot posterior predictive checks
        BayesFitness.viz.ppc_time_series!(
            axes[1], qs, ppc_mat; colors=colors
        )

        # Add plot for median (we use the 5 percentile to have a "thicker" line
        # showing the median)
        BayesFitness.viz.ppc_time_series!(
            axes[1], [0.05], ppc_mat; colors=ColorSchemes.Purples[end:end]
        )

        # Add scatter of data
        scatterlines!(axes[1], diff(log.(data_bc.freq)), color=:black)

        # Add title
        axes[1].title = "joint inference"
        axes[1].titlesize = 12

        ## === Single barcode inference === ##
        # Define colors
        colors = get(ColorSchemes.Blues_9, LinRange(0.5, 0.75, length(qs)))

        # Define dictionary with corresponding parameters for variables needed
        # for the posterior predictive checks
        param = Dict(
            :mutant_mean_fitness => :s⁽ᵐ⁾,
            :mutant_std_fitness => :σ⁽ᵐ⁾,
            :population_mean_fitness => :s̲ₜ,
        )
        # Compute posterior predictive checks
        ppc_mat = BayesFitness.stats.logfreq_ratio_mutant_ppc(
            df_dict[bc_plot[counter]], n_ppc; param=param
        )
        # Plot posterior predictive checks
        BayesFitness.viz.ppc_time_series!(
            axes[2], qs, ppc_mat; colors=colors
        )

        # Add plot for median (we use the 5 percentile to have a "thicker" line
        # showing the median)
        BayesFitness.viz.ppc_time_series!(
            axes[2], [0.05], ppc_mat; colors=ColorSchemes.Blues[end:end]
        )

        # Add scatter of data
        scatterlines!(axes[2], diff(log.(data_bc.freq)), color=:black)

        # Add title
        axes[2].title = "single barcode"
        axes[2].titlesize = 12
        ## == Plot format == ##

        # Hide axis decorations
        hidedecorations!.(axes, grid=false)
        # Set row and col gaps
        rowgap!(gl, 1)

        # Add barcode as title
        Label(
            gl[0, 3:4],
            text="barcode $(bc_plot[counter])",
            fontsize=12,
            justification=:center,
            lineheight=0.9
        )

        # Update counter
        counter += 1
    end  # for
end # for

# Add x-axis label
Label(fig[end, :, Bottom()], "time points", fontsize=20)
# Add y-axis label
Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=20)

fig

##

# Find barcode with maximum count
bc = data[first(argmax(data.count, dims=1)), :barcode]
# bc = 10050

# Extract data for barcode example
data_bc = data[data.barcode.==bc, :]

# Locate group with barcode
group_idx = [any(g .== bc) for g in bc_groups]

# Define number of posterior predictive check samples
n_ppc = 1_000

# Define dictionary with corresponding parameters for variables needed for the
# posterior predictive checks
param = Dict(
    :mutant_mean_fitness => Symbol("sbc$(bc)"),
    :mutant_std_fitness => Symbol("σbc$(bc)"),
    :population_mean_fitness => :s̲ₜ,
)

# Initialize figure
fig = Figure(resolution=(350 * 2, 350))

# Add axis
ax = [
    Axis(
        fig[1, i],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
    ) for i = 1:2
]

# Define quantiles to compute
qs = [0.95, 0.675]

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.5, 0.75, length(qs)))
# Define the plot titles
titles = ["full", "grouped"]

# Loop through dataframes
for (i, d) in enumerate([df_joint, first(df_groups[group_idx])])

    # Compute posterior predictive checks
    ppc_mat = BayesFitness.stats.logfreq_ratio_mutant_ppc(
        d, n_ppc; param=param
    )

    # Plot posterior predictive checks
    BayesFitness.viz.ppc_time_series!(
        ax[i], qs, ppc_mat; colors=colors
    )

    # Add plot for median (we use the 5 percentile to have a "thicker" line showing
    # the median)
    BayesFitness.viz.ppc_time_series!(
        ax[i], [0.05], ppc_mat; colors=ColorSchemes.Blues_9[end:end]
    )

    # Add scatter of data
    scatterlines!(ax[i], diff(log.(data_bc.freq)), color=:black)

    # Add subplot title
    ax[i].title = "PPC | $(titles[i])"

end # for

fig

##