# Import plotting-related libraries
using Measures, CairoMakie
import Makie
import ColorSchemes

# Import library to handle dataframes
import DataFrames as DF

# Import library to handle MCMCChains
import MCMCChains

# Import function from stats module
import BayesFitness.stats: matrix_quantile_range, freq_mutant_ppc_quantile, logfreqratio_neutral_ppc_quantile

@doc raw"""
    mcmc_trace_density!(fig::Figure, chain::MCMCChains.Chains; colors, labels)

Function to plot the traces and density estimates side-to-side for each of the
parametres in the `MCMCChains.Chains` object.

# Arguments
- `fig::Makie.Figure`: Figure object to be populated with plot. This allows the
  user to decide the size of the figure outside of this function.
- `chain::MCMCChains.Chains`: Samples from the MCMC run generated with
  Turing.jl.

## Optional arguments
- `colors=ColorSchemes.seaborn_colorblind`: List of colors to be used in plot.
- `labels`: List of labels for each of the parameters. If not given, the default
  will be to use the names stored in the MCMCChains.Chains object.
- `alpha::AbstractFloat=1`: Level of transparency for plots.
- `title::Union{String,Nothing}=nothing`: Plot title.
- `title_valign::Symbol=:bottom`: Vertical alignment for title label,
- `title_font::Symbol=:bold`: Type of font to be used in plot.
- `title_fontsize::Real=20`: Font size for title.
- `title_padding::Vector{<:Real}=(0, 0, 5, 0)`: Padding for plot text.
"""
function mcmc_trace_density!(
    fig::Figure,
    chain::MCMCChains.Chains;
    colors=ColorSchemes.seaborn_colorblind,
    labels=[],
    alpha::AbstractFloat=1.0,
    title::Union{String,Nothing}=nothing,
    title_valign::Symbol=:bottom,
    title_font::Symbol=:bold,
    title_fontsize::Real=20,
    title_padding::NTuple{4,<:Real}=(0, 0, 5, 0)
)
    # Extract parameters
    params = names(chain, :parameters)
    # Extract number of chains
    n_chains = length(MCMCChains.chains(chain))
    # Extract number of parameters
    n_samples = length(chain)

    # Check that the number of given labels is correct
    if (length(labels) > 0) & (length(labels) != length(params))
        error("The number of lables must match number of parameters")
    end # if

    # Check that the number of given colors is correct
    if length(colors) < n_chains
        error("Please give at least as many colors as chains in the MCMC")
    end # if

    # Add GridLayout to give more control over figure
    gl = fig[1, 1] = GridLayout()

    # Loop through parameters
    for (i, param) in enumerate(params)
        # Check if labels were given
        if length(labels) > 0
            lab = labels[i]
        else
            lab = string(param)
        end # if
        # Add axis for chain iteration
        ax_trace = Axis(gl[i, 1]; ylabel=lab)
        # Inititalize axis for density plot
        ax_density = Axis(gl[i, 2]; ylabel=lab)
        # Loop through chains
        for chn in 1:n_chains
            # Extract values
            values = chain[:, param, chn]
            # Plot traces of walker
            lines!(ax_trace, 1:n_samples, values, color=(colors[chn], alpha))
            # Plot density
            density!(ax_density, values, color=(colors[chn], alpha))
        end # for

        # Hide y-axis decorations
        hideydecorations!(ax_trace; label=false)
        hideydecorations!(ax_density; label=false)

        # Check if it is bottom plot
        if i < length(params)
            # hide x-axis decoratiosn
            hidexdecorations!(ax_trace; grid=false)
        else
            # add x-label
            ax_trace.xlabel = "iteration"
            ax_density.xlabel = "parameter estimate"
        end # if
    end # for

    # Check if title should be added
    if typeof(title) == String
        Label(
            gl[1, 1:2, Top()],
            title,
            valign=title_valign,
            font=title_font,
            fontsize=title_fontsize,
            padding=title_padding,
        )
    end # if
end # function

@doc raw"""
    freq_mutant_ppc!(fig, quantile, chain; colors, alpha, varname_mut, varname_mean, freq_mut)

Function to plot the **posterior predictive checks** quantiles for the barcode
frequency for adaptive mutants.

# Arguments
- `fig::Makie.Axis`: Axis object to be populated with plot. 
- `quantile::Vector{<:AbstractFloat}`: List of quantiles to extract from the
    posterior predictive checks.
- `chain::MCMCChains.Chains`: `Turing.jl` MCMC chain for the fitness of a single
    mutant.

## Optional arguments
- `colors=ColorSchemes.Blues_9`: List of colors to use for each quantile.
- `alpha::AbstractFloat=0.75`: Level of transparency for band representing each
  quantile.
- `varname_mut::Symbol=Symbol("s⁽ᵐ⁾")`: Variable name for the mutant relative
    fitness in the `chain` object.
- `varname_mean::Symbol=Symbol("s̲ₜ")`: Variable name for *all* population mean
    fitness.
- `freq_mut::Symbol=Symbol("f̲⁽ᵐ⁾")`: Variable name for *all* mutant barcode
    frequencies.
"""
function freq_mutant_ppc!(
    ax::Makie.Axis,
    quantile::Vector{<:AbstractFloat},
    chain::MCMCChains.Chains;
    colors=ColorSchemes.Blues_9,
    alpha::AbstractFloat=0.75,
    varname_mut::Symbol=Symbol("s⁽ᵐ⁾"),
    varname_mean::Symbol=Symbol("s̲ₜ"),
    freq_mut::Symbol=Symbol("f̲⁽ᵐ⁾")
)
    # Make sure there are enough colors for each quant
    if length(colors) < length(quantile)
        error("There are not enough colors listed for all quantiles")
    end # if

    # Tell user that quantiles will be sorted
    if quantile != sort(quantile, rev=true)
        println("Notice that we sort the quantiles to properly display the intervals")
    end # if

    # Sort quantiles
    sort!(quantile, rev=true)

    # Compute posterior predictive checks
    f_quant = freq_mutant_ppc_quantile(
        quantile,
        chain;
        varname_mut=varname_mut,
        varname_mean=varname_mean,
        freq_mut=freq_mut
    )

    # Loop through quantiles
    for i in eachindex(quantile)
        # Add confidence interval for observation
        band!(
            ax,
            1:size(f_quant, 1),
            f_quant[:, i, 1],
            f_quant[:, i, 2],
            color=(colors[i], alpha)
        )
    end # for
end # function

@doc raw"""
    freq_mutant_ppc!(ax, quantile, df, varname_mut, varname_mean, varname_freq; colors, alpha)

Function to plot the **posterior predictive checks** quantiles for the barcode
frequency for adaptive mutants.

# Arguments
- `ax::Makie.Axis`: Axis object to be populated with plot. 
- `quantile::Vector{<:AbstractFloat}`: List of quantiles to extract from the
    posterior predictive checks.
- `df::DataFrames.DataFrame`: Dataframe containing the MCMC samples for the
variables needed to compute the posterior predictive checks. The dataframe
should have MCMC samples for
    - mutant relative fitness values.
    - population mean fitness values. NOTE: The number of columns containing
      population mean fitness values determines the number of datapoints where the
      ppc are evaluated.
    - mutant initial frequency.
  - `varname_mut::Union{Symbol, AbstractString}`: Variable name for the mutant
      relative fitness in the data frame.
  - `varname_mean::Union{Symbol, AbstractString}`: Variable name pattern for *all*
    population mean fitness. All columns in the dataframe should contain this
    pattern and the sorting of these names must correspond to the sorting of the
    time points.
  - `varname_freq::Union{Symbol, AbstractString}`: Variable name for initial mutant
    barcode frequencies.

## Optional arguments
- `colors=ColorSchemes.Blues_9`: List of colors to use for each quantile.
- `alpha::AbstractFloat=0.75`: Level of transparency for band representing each
  quantile.
"""
function freq_mutant_ppc!(
    ax::Makie.Axis,
    quantile::Vector{<:AbstractFloat},
    df::DF.AbstractDataFrame,
    varname_mut::Union{Symbol,AbstractString},
    varname_mean::Union{Symbol,AbstractString},
    varname_freq::Union{Symbol,AbstractString};
    colors=ColorSchemes.Blues_9,
    alpha::AbstractFloat=0.75
)
    # Make sure there are enough colors for each quant
    if length(colors) < length(quantile)
        error("There are not enough colors listed for all quantiles")
    end # if

    # Tell user that quantiles will be sorted
    if quantile != sort(quantile, rev=true)
        println("Notice that we sort the quantiles to properly display the intervals")
    end # if

    # Sort quantiles
    sort!(quantile, rev=true)

    # Compute posterior predictive checks
    f_quant = freq_mutant_ppc_quantile(
        quantile, df, varname_mut, varname_mean, varname_freq
    )

    # Loop through quantiles
    for i in eachindex(quantile)
        # Add confidence interval for observation
        band!(
            ax,
            1:size(f_quant, 1),
            f_quant[:, i, 1],
            f_quant[:, i, 2],
            color=(colors[i], alpha)
        )
    end # for
end # function

@doc raw"""
    logfreqratio_neutral_ppc!(fig, quantile, df; colors, alpha,)

Function to plot the **posterior predictive checks** quantiles for the log
frequency ratio for neutral lineages

# Arguments
- `fig::Makie.Axis`: Axis object to be populated with plot. 
- `quantile::Vector{<:AbstractFloat}`: List of quantiles to extract from the
    posterior predictive checks.
-`df::DataFrames.DataFrame`: DataFrame containing all population mean fitness
samples⸺multiple chains must be collapsed into a single column⸺one time point
per column. Note: we recommend using the `var_jld2_to_df` from the `utils`
module to build this dataframe.

## Optional arguments
- `colors=ColorSchemes.Blues_9`: List of colors to use for each quantile.
- `alpha::AbstractFloat=0.75`: Level of transparency for band representing each
  quantile.
"""
function logfreqratio_neutral_ppc!(
    ax::Makie.Axis,
    quantile::Vector{<:AbstractFloat},
    df::DF.AbstractDataFrame;
    colors=reverse(ColorSchemes.Blues_9),
    alpha::AbstractFloat=0.75
)
    # Make sure there are enough colors for each quant
    if length(colors) < length(quantile)
        error("There are not enough colors listed for all quantiles")
    end # if

    # Tell user that quantiles will be sorted
    if any(quantile .!= sort(quantile, rev=true))
        println("Notice that we sort the quantiles to properly display the intervals")
    end # if

    # Sort quantiles
    sort!(quantile, rev=true)

    # Compute the posterior predictive checks
    logf_quant = -1 .* logfreqratio_neutral_ppc_quantile(quantile, df)

    # Loop through quantiles
    for i in eachindex(quantile)
        # Add confidence interval for observation
        band!(
            ax,
            1:size(logf_quant, 1),
            logf_quant[:, i, 1],
            logf_quant[:, i, 2],
            color=(colors[i], alpha)
        )
    end # for
end # function

@doc raw"""
    time_vs_freq_ppc!(ax, quantile, ppc_mat; colors, alpha)

Function to plot the posterior predictive checks quantiles for the barcode
frequency time trajectories.

# Arguments
- `fig::Makie.Axis`: Axis object to be populated with plot. 
- `quantile::Vector{<:AbstractFloat}`: List of quantiles to extract from the
    posterior predictive checks.
- `ppc_mat::Matrix{<:AbstractFloat}`: Matrix containing the posterior predictive
  samples. Rows are assumed to contain the samples, columns the time points.

## Optional arguments
- `colors=ColorSchemes.Blues_9`: List of colors to use for each quantile.
- `alpha::AbstractFloat=0.75`: Level of transparency for band representing each
quantile.
"""
function time_vs_freq_ppc!(
    ax::Makie.Axis,
    quantile::Vector{<:AbstractFloat},
    ppc_mat::Matrix{<:AbstractFloat};
    colors=ColorSchemes.Blues_9,
    alpha::AbstractFloat=0.75
)
    # Tell user that quantiles will be sorted
    if quantile != sort(quantile, rev=true)
        println("Notice that we sort the quantiles to properly display the intervals")
    end # if

    # Sort quantiles
    sort!(quantile, rev=true)

    # Compute quantiles
    ppc_quant = matrix_quantile_range(quantile, ppc_mat)

    # Loop through quantiles
    for i in eachindex(quantile)
        # Add confidence interval for observation
        band!(
            ax,
            1:size(ppc_quant, 1),
            ppc_quant[:, i, 1],
            ppc_quant[:, i, 2],
            color=(colors[i], alpha)
        )
    end # for
end # function
