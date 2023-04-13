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

# Import library to save and load native julia objects
import JLD2

# Import library to list files
import Glob

# Import plotting libraries
using CairoMakie
import ColorSchemes
##

# Import data
data = CSV.read("$(git_root())/test/data/data_example_01.csv", DF.DataFrame)

##

# Define function parameters

param = Dict(
    :data => data,
    :n_walkers => 3,
    :n_steps => 1_000,
    :outputdir => "./output/",
    :outputname => "data_01_meanfitness",
    :model => BayesFitness.model.mean_fitness_neutrals_lognormal,
    :model_kwargs => Dict(
        :α => BayesFitness.stats.dirichlet_prior_neutral(
            data[data.time.==0, :neutral],
        )
    )
)
##

# Run inference
BayesFitness.mcmc.mcmc_mean_fitness(; param...)

##

# Concatenate population mean fitness chains into single chain
chains = BayesFitness.utils.var_jld2_concat(
    param[:outputdir], param[:outputname], :sₜ
)

# Initialize figure
fig = Figure(resolution=(600, 600))

# Generate mcmc_trace_density! plot
BayesFitness.viz.mcmc_trace_density!(fig, chains; alpha=0.5)

fig

##

# Define quantiles to compute
qs = [0.95, 0.675, 0.02]

# Generate dataframe with mean fitness samples
df_meanfit = BayesFitness.utils.var_jld2_to_df(
    param[:outputdir], param[:outputname], :sₜ
)

# Define colors
colors = reverse(get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs))))

# Initialize figure
fig = Figure(resolution=(400, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="log(fₜ₊₁ / fₜ)",
)

# Plot posterior predictive checks
BayesFitness.viz.logfreqratio_neutral_ppc!(ax, qs, df_meanfit; colors=colors)

# Group data by barcod
df_group = DF.groupby(data[data.neutral, :], :barcode)

# Loop through groups
for d in df_group
    # Sort data by time
    DF.sort!(d, :time)

    # Compute log ratios

    # Plot data
    scatterlines!(
        ax, diff(log.(d.count ./ d.count_sum)), color=(:black, 0.2)
    )
end # for

fig

## 