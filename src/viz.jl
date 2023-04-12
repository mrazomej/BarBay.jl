# Import plotting-related libraries
using Measures, CairoMakie
import Makie
import ColorSchemes

# Import library to handle MCMCChains
import MCMCChains

# Import function from stats module
import BayesFitness.stats: freq_mutant_ppc_quantile

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
"""
function mcmc_trace_density!(
    fig::Figure,
    chain::MCMCChains.Chains;
    colors=ColorSchemes.seaborn_colorblind,
    labels=[],
    alpha::AbstractFloat=1.0
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

    # Loop through parameters
    for (i, param) in enumerate(params)
        # Check if labels were given
        if length(labels) > 0
            lab = labels[i]
        else
            lab = string(param)
        end # if
        # Add axis for chain iteration
        ax_trace = Axis(fig[i, 1]; ylabel=lab)
        # Inititalize axis for density plot
        ax_density = Axis(fig[i, 2]; ylabel=lab)
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
end # function

@doc raw"""
    freq_mutant_ppc!(fig, quantile, chain, s_mut, s_pop, freq_mut)

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
- `s_mut::Symbol=Symbol("s⁽ᵐ⁾")`: Variable name for the mutant relative fitness
    in the `chain` object.
- `s_pop::Symbol=Symbol("s̲ₜ")`: Variable name for *all* population mean
    fitness.
- `freq_mut::Symbol=Symbol("f̲⁽ᵐ⁾")`: Variable name for *all* mutant barcode
    frequencies.
"""
function freq_mutant_ppc!(
    ax::Makie.Axis,
    quantile::Vector{<:AbstractFloat},
    chain::MCMCChains.Chains;
    colors=reverse(ColorSchemes.Blues_9),
    alpha::AbstractFloat=0.75,
    s_mut::Symbol=Symbol("s⁽ᵐ⁾"),
    s_pop::Symbol=Symbol("s̲ₜ"),
    freq_mut::Symbol=Symbol("f̲⁽ᵐ⁾")
)
    # Make sure there are enough colors for each quant
    if length(colors) < length(quantile)
        error("There are not enough colors listed for all quantiles")
    end # if

    # Sort quantiles
    sort!(quantile)

    # Compute posterior predictive checks
    f_quant = freq_mutant_ppc_quantile(
        quantile, chain; s_mut=s_mut, s_pop=s_pop, freq_mut=freq_mut
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