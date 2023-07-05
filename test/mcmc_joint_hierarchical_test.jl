##

# Activate environment
@load_pkg(".")

# Import package to revise package
import Revise

# Import library package
import BayesFitness

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

##
println("Loading data...\n")
# Import data
data = CSV.read(
    "$(git_root())/test/data/data_hierarchical_example_01.csv", DF.DataFrame
)

##

# Plot trajectories

println("Plotting frequency trajectories...\n")
# Initialize figure
fig = Figure(resolution=(450 * 2, 350))

# Add axis
ax1 = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="barcode frequency",
    yscale=log10,
    title="frequency trajectories | R1"
)

ax2 = Axis(
    fig[1, 2],
    xlabel="time point",
    ylabel="barcode frequency",
    yscale=log10,
    title="frequency trajectories | R2"
)

# compile axes into single object
axes = [ax1, ax2]
# Loop through repeats
for (i, rep) in enumerate(unique(data.rep))
    # Plot Mutant barcode trajectories
    BayesFitness.viz.bc_time_series!(
        axes[i],
        data[(.!data.neutral).&(data.rep.==rep), :];
        quant_col=:freq,
        zero_lim=1E-7,
        zero_label="extinct",
        alpha=0.25,
        linewidth=2
    )

    # Plot Neutral barcode trajectories for R1
    BayesFitness.viz.bc_time_series!(
        axes[i],
        data[(data.neutral).&(data.rep.==rep), :];
        quant_col=:freq,
        zero_lim=1E-7,
        color=ColorSchemes.Blues_9[end],
        alpha=0.9,
        linewidth=2
    )
end # for

fig

##

param = Dict(
    :data => data,
    :n_walkers => n_walkers,
    :n_steps => n_steps,
    :outputname => "./output/data_example_01_hierarchical_$(n_steps)steps_$(lpad(n_walkers, 2, "0"))walkers",
    :model => BayesFitness.model.fitness_hierarchical_replicates,
    :model_kwargs => Dict(
        :λ_prior => [3.0, 3.0]
    ),
    :sampler => Turing.NUTS(0.65),
    :ensemble => Turing.MCMCThreads(),
)

##

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Run inference
println("Running Inference...")
@time BayesFitness.mcmc.mcmc_fitness_hierarchical_replicates(; param...)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load MCMC chain into memory
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

# Define number of repeats in experiment
n_rep = length(unique(data.rep))
# Load chain
chain = JLD2.load(
    "./output/data_example_01_hierarchical_$(n_steps)steps_" *
    "$(lpad(n_walkers, 2, "0"))walkers.jld2"
)["chain"]
# Load mutant IDs
ids = JLD2.load(
    "./output/data_example_01_hierarchical_$(n_steps)steps_" *
    "$(lpad(n_walkers, 2, "0"))walkers.jld2"
)["ids"]

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Format tidy dataframe with proper variable names
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

# 1. Locate variables
# Locate hyperparameter variables
θ_var = MCMCChains.namesingroup(chain, :θ̲⁽ᵐ⁾)
# Find columns with fitness parameter deviation 
τ_vars = MCMCChains.namesingroup(chain, :τ̲⁽ᵐ⁾)
θ_tilde_vars = MCMCChains.namesingroup(chain, :θ̲̃⁽ᵐ⁾)
# Find columns with mutant fitness error
σ_vars = MCMCChains.namesingroup(chain, :σ̲⁽ᵐ⁾)
# Extract population mean fitness variable names
pop_mean_vars = MCMCChains.namesingroup(chain, :s̲ₜ)
# Extract population mean fitness error variables
pop_std_vars = MCMCChains.namesingroup(chain, :σ̲ₜ)

# 2. Define names based on barcode name and replicate number
# Define barcode names. This will be in the format `bc#_Rx`
bc_names = vcat(["bc" .* string.(ids) .* "_R$(i)" for i = 1:n_rep]...)
# Define mean fitness variable names this only includes the `R[x]` part to later
# on attach either s̲ₜ or σ̲ₜ
pop_names = vcat([
    "R$i[" .* string.(1:(length(pop_mean_vars)÷n_rep)) .* "]"
    for i = 1:n_rep
]...)

# 3. Convert chain to tidy dataframe
# Convert chain to tidy dataframe
df_chain = DF.DataFrame(chain)

# 4. Compute individual replicate fitness value. This value is not directly
# track by Turing.jl when sampling because it is a derived quantity, not an
# input to the model. We could use the `Turing.generated_quantities` function to
# compute this, but it is simpler to do it directly from the chains.

# Compute individual strains fitness values
s_mat = hcat(repeat([Matrix(df_chain[:, θ_var])], n_rep)...) .+
        (Matrix(df_chain[:, τ_vars]) .* Matrix(df_chain[:, θ_tilde_vars]))

# 5. Insert individual replicate fitness values to dataframe
# Add fitness values to dataframe
DF.insertcols!(df_chain, (Symbol.("s⁽ᵐ⁾_" .* bc_names) .=> eachcol(s_mat))...)

# 6. Rename corresponding variables
# Rename error columns
DF.rename!(df_chain,)
# Rename population mean fitness variables
DF.rename!(
    df_chain,
    [
        θ_var .=> Symbol.("θ⁽ᵐ⁾_" .* string.(ids))
        σ_vars .=> Symbol.("σ⁽ᵐ⁾_" .* bc_names)
        pop_mean_vars .=> Symbol.("s̲ₜ_" .* pop_names)
        pop_std_vars .=> Symbol.("σ̲ₜ_" .* pop_names)
    ]
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot population mean fitness ECDF
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of rows and columns
n_row, n_col = [2, 2]

# Initialize figure
fig = Figure(resolution=(600, 600))

# Add axis objects for each timepoint
axes = [
    Axis(
        fig[i, j],
        xlabel="population mean fitness (s̄ₜ)",
        ylabel="ecdf",
    ) for i = 1:n_row for j = 1:n_col
]

# Loop through time points
for t = 1:(length(pop_names)÷n_rep)
    # Loop through experimental repeats
    for rep = 1:n_rep
        # Extract population mean fitness variable names
        pop_vars = names(df_chain)[occursin.("s̲ₜ_R$rep", names(df_chain))]
        # Plot ECDF
        ecdfplot!(axes[t], df_chain[:, pop_vars[t]], label="R$rep")
    end # for
    # Add legend
    axislegend(axes[t], labelsize=12, framevisible=false, position=:rb)
    # Add title
    axes[t].title = "time $t"
end # for

fig

##

# Initialize figure
fig = Figure(resolution=(300 * 2, 300))

# Add axis
axes = [
    Axis(
        fig[1, j],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
    ) for j = 1:n_rep
]

# Define number of posterior predictive check samples
n_ppc = 500

# Define quantiles to compute
qs = [0.68, 0.95, 0.997]

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 0.75, length(qs)))

# Loop through repeats
for rep = 1:n_rep
    # Define dictionary with corresponding parameters for variables needed for
    # the posterior predictive checks
    param = Dict(
        :population_mean_fitness => Symbol("s̲ₜ_R$rep"),
        :population_std_fitness => Symbol("σ̲ₜ_R$rep"),
    )

    # Compute posterior predictive checks
    ppc_mat = BayesFitness.stats.logfreq_ratio_mean_ppc(
        df_chain, n_ppc; param=param
    )


    # Plot posterior predictive checks
    BayesFitness.viz.ppc_time_series!(
        axes[rep], qs, ppc_mat; colors=colors
    )

    # Add plot for median (we use the 5 percentile to have a "thicker" line
    # showing the median)
    BayesFitness.viz.ppc_time_series!(
        axes[rep], [0.05], ppc_mat; colors=ColorSchemes.Blues_9[end:end]
    )

    # Plot log-frequency ratio of neutrals
    BayesFitness.viz.logfreq_ratio_time_series!(
        axes[rep],
        data[(data.neutral).&(data.rep.=="R$rep"), :];
        freq_col=:freq,
        color=:black,
        alpha=1.0,
        linewidth=2
    )

    # Set axis title
    axes[rep].title = "log-frequency ratio PPC | R$rep"
end # for

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute summary statistics for each barcode
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize dataframe to save individual replicate summary statistics
df_summary = DF.DataFrame()

# Find barcode variables
var_names = names(df_chain)[occursin.("s⁽ᵐ⁾", names(df_chain))]

# Define percentiles to include
per = [2.5, 97.5, 16, 84]

# Loop through barcodes
for bc in var_names
    # Extract hyperparameter variable name
    θ_name = replace(split(bc, "_")[2], "bc" => "")
    # Extract fitness chain
    fitness_s = @view df_chain[:, bc]
    # Extract hyperparameter chain
    fitness_θ = @view df_chain[:, "θ⁽ᵐ⁾_$(θ_name)"]

    ## FILTER highest percentiles ##
    fitness_s = fitness_s[
        (fitness_s.≥StatsBase.percentile(fitness_s, 5)).&(fitness_s.≤StatsBase.percentile(fitness_s, 95))
    ]
    fitness_θ = fitness_θ[
        (fitness_θ.≥StatsBase.percentile(fitness_θ, 5)).&(fitness_θ.≤StatsBase.percentile(fitness_θ, 95))
    ]

    # Compute summary statistics
    fitness_summary = Dict(
        :s_mean => StatsBase.mean(fitness_s),
        :s_median => StatsBase.median(fitness_s),
        :s_std => StatsBase.std(fitness_s),
        :s_var => StatsBase.var(fitness_s),
        :s_skewness => StatsBase.skewness(fitness_s),
        :s_kurtosis => StatsBase.kurtosis(fitness_s),
        :θ_mean => StatsBase.mean(fitness_θ),
        :θ_median => StatsBase.median(fitness_θ),
        :θ_std => StatsBase.std(fitness_θ),
        :θ_var => StatsBase.var(fitness_θ),
        :θ_skewness => StatsBase.skewness(fitness_θ),
        :θ_kurtosis => StatsBase.kurtosis(fitness_θ),)

    # Loop through percentiles
    for p in per
        setindex!(
            fitness_summary,
            StatsBase.percentile(fitness_s, p),
            Symbol("s_$p")
        )
        setindex!(
            fitness_summary,
            StatsBase.percentile(fitness_θ, p),
            Symbol("θ_$p")
        )
    end # for

    # Convert to dataframe
    df_fitness = DF.DataFrame(fitness_summary)
    # Add barcode
    df_fitness[!, :id] .= bc
    # Append to dataframe
    DF.append!(df_summary, df_fitness)
end # for

# Add barcode and replicated number as extra columns
DF.insertcols!(
    df_summary,
    :barcode => [
        parse(Int64, replace(split(x, "_")[2], "bc" => ""))
        for x in df_summary.id
    ],
    :rep => [split(x, "_")[3] for x in df_summary.id],
)

# Sort by barcode
DF.sort!(df_summary, :barcode)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare median fitness for individual replicates
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(400, 400))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="replicate 1 fitness",
    ylabel="replicate 2 fitness",
)

# Plot identity line
lines!(ax, [-1, 1.5], [-1, 1.5], linestyle=:dash, color="black")

# Group data by repeat
df_group = DF.groupby(df_summary, :rep)

# Plot x-axis error bars
errorbars!(
    ax,
    df_group[1].s_median,
    df_group[2].s_median,
    abs.(df_group[1].s_median .- df_group[1][:, Symbol("s_16.0")]),
    abs.(df_group[1].s_median .- df_group[1][:, Symbol("s_84.0")]),
    direction=:x,
    linewidth=1.5,
    color=(:gray, 0.5)
)
# Plot y-axis error bars
errorbars!(
    ax,
    df_group[1].s_median,
    df_group[2].s_median,
    abs.(df_group[2].s_median .- df_group[2][:, Symbol("s_16.0")]),
    abs.(df_group[2].s_median .- df_group[2][:, Symbol("s_84.0")]),
    direction=:y,
    linewidth=1.5,
    color=(:gray, 0.5)
)

# Plot fitness values
scatter!(ax, df_group[1].s_median, df_group[2].s_median, markersize=5)

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare median fitness with hyperparameter
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(350 * 2, 350))

# Add axis
axes = [
    Axis(
        fig[1, i],
        xlabel="hyper parameter fitness",
        ylabel="individual replicate fitness",
        aspect=AxisAspect(1),
    ) for i = 1:2
]

# Group data by repeat
df_group = DF.groupby(df_summary, :rep)

# Loop through groups
for (i, df) in enumerate(df_group)
    # Plot identity line
    lines!(axes[i], [-1, 1.5], [-1, 1.5], linestyle=:dash, color="black")

    # Plot x-axis error bars
    errorbars!(
        axes[i],
        df.θ_median,
        df.s_median,
        abs.(df.θ_median .- df[:, Symbol("θ_16.0")]),
        abs.(df.θ_median .- df[:, Symbol("θ_84.0")]),
        direction=:x,
        linewidth=1.5,
        color=(:gray, 0.5)
    )
    # Plot y-axis error bars
    errorbars!(
        axes[i],
        df.θ_median,
        df.s_median,
        abs.(df.θ_median .- df[:, Symbol("s_16.0")]),
        abs.(df.θ_median .- df[:, Symbol("s_84.0")]),
        direction=:y,
        linewidth=1.5,
        color=(:gray, 0.5)
    )

    # Plot fitness values
    scatter!(axes[i], df.θ_median, df.s_median, markersize=5)

    # Add plot title
    axes[i].title = "replicate R$(i)"

end # for
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

# List example barcodes to plot
bc_plot = ids[1:(n_row*n_col)]

# Define colors
colors = [
    get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs) + 1)),
    get(ColorSchemes.Purples_9, LinRange(0.5, 1, length(qs) + 1))
]

# Initialize figure
fig = Figure(resolution=(300 * n_col, 300 * n_row))

# Initialize plot counter
counter = 1
# Loop through rows
for row in 1:n_row
    # Loop through columns
    for col in 1:n_col
        # Add GridLayout
        gl = fig[row, col] = GridLayout()
        # Add axis
        axes = [Axis(gl[i, 1:6]) for i = 1:n_rep]

        # Loop through replicates
        for rep = 1:n_rep

            # Extract data
            data_bc = DF.sort(
                data[(data.barcode.==bc_plot[counter]).&(data.rep.=="R$rep"),
                    :],
                :time
            )


            # Define dictionary with corresponding parameters for variables needed
            # for the posterior predictive checks
            param = Dict(
                :mutant_mean_fitness => Symbol(
                    "s⁽ᵐ⁾_bc$(bc_plot[counter])_R$rep"
                ),
                :mutant_std_fitness => Symbol(
                    "σ⁽ᵐ⁾_bc$(bc_plot[counter])_R$rep"
                ),
                :population_mean_fitness => Symbol("s̲ₜ_R$rep"),
            )
            # Compute posterior predictive checks
            ppc_mat = BayesFitness.stats.logfreq_ratio_mutant_ppc(
                df_chain, n_ppc; param=param
            )
            # Plot posterior predictive checks
            BayesFitness.viz.ppc_time_series!(
                axes[rep], qs, ppc_mat; colors=colors[rep]
            )

            # Add plot for median (we use the 5 percentile to have a "thicker"
            # line showing the median)
            BayesFitness.viz.ppc_time_series!(
                axes[rep], [0.05], ppc_mat; colors=[colors[rep][end]]
            )

            # Add scatter of data
            scatterlines!(axes[rep], diff(log.(data_bc.freq)), color=:black)

            # Add title
            axes[rep].title = "replicate R$rep"
            axes[rep].titlesize = 12
        end # for

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