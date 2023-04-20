# Import plotting-related libraries
using Measures, CairoMakie
import Makie
import ColorSchemes
import ColorTypes

# Import library to handle dataframes
import DataFrames as DF

# Import statistical libraries
import StatsBase
import Distributions

# Import library to handle MCMCChains
import MCMCChains

# Import function from stats module
import BayesFitness.stats: matrix_quantile_range, freq_mutant_ppc_quantile, logfreqratio_neutral_ppc_quantile, gaussian_prior_mean_fitness

@doc raw"""
    bc_time_series!(ax, data; color, alpha, id_col, time_col, quant_col)

Function to plot the time series of a quantity (frequency or raw counts, for
example) for a set of barcodes. This function expects the data in a **tidy**
format. This means that every row represents **a single observation**. For
example, if we measure barcode `i` in 4 different time points, each of these
four measurements gets an individual row. Furthermore, measurements of barcode
`j` over time also get their own individual rows.
    
The `DataFrame` must contain at least the following columns:
- `id_col`: Column identifying the ID of the barcode. This can the barcode
    sequence, for example.
- `time_col`: Column defining the measurement time point.
- `quant_col`: Column with the quantity to be plot over time.

# Arguments
- `ax::Makie.Axis`: Axis object to be populated with plot.
- `data::DataFrames.AbstractDataFrame`: **Tidy dataframe** with the data to be
  used to sample from the population mean fitness posterior distribution.

## Optional Arguments
- `id_col::Symbol=:barcode`: Name of the column in `data` containing the barcode
    identifier. The column may contain any type of entry.
- `time_col::Symbol=:time`: Name of the column in `data` defining the time point
at which measurements were done. The column may contain any type of entry as
long as `sort` will resulted in time-ordered names.
- `quant_col::Symbol=:count`: Name of the column in `data` containing the raw
  barcode count.
- `zero_lim::Real=1E-8`: Number defining under which value `quant_col` should be
  considered as zero. These plots are mostly displayed in log scale, thus having
  a minimum threshold helps with undetermined values.
- `zero_label::Union{String, Nothing}`: Label to be added to the detection
  limit. If `nothing`, nothing is added to the plot.
- `n_ticks::Int`: Ideal number of ticks to add to plot. See
  `Makie.WilkinsonTicks`.
- `color::Union{ColorSchemes.ColorScheme,Symbol,ColorTypes.Colorant{Float64,
  3}}=ColorSchemes.glasbey_hv_n256`: Single color or list of colors from
  `ColorSchemes.jl` to be used in plot. Note: when a color list is provided,
  colors are randomnly assigned to each barcode by sampling from the list of
  colors.
- `alpha::AbstractFloat=1.0`: Level of transparency for plots.
- `linewidth::Real=5`: Trajectory linewidth.
"""
function bc_time_series!(
    ax::Makie.Axis,
    data::DF.AbstractDataFrame;
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    quant_col::Symbol=:count,
    zero_lim::Real=1E-8,
    zero_label::Union{String,Nothing}=nothing,
    n_ticks::Int=4,
    color::Union{ColorSchemes.ColorScheme,Symbol,ColorTypes.Colorant{Float64,3}}=ColorSchemes.glasbey_hv_n256,
    alpha::AbstractFloat=1.0,
    linewidth::Real=2
)
    # Group data by id_col
    data_group = DF.groupby(data, id_col)

    # Loop through trajectories
    for bc in data_group
        # Modify barcode values below zero_lim
        bc[bc[:, quant_col].<zero_lim, quant_col] .= zero_lim

        # Sort data by time
        DF.sort!(bc, time_col)

        # Check if unique color was assigned to each barcode
        if typeof(color) <: ColorSchemes.ColorScheme
            # Plot trajectory
            lines!(
                ax,
                bc[:, time_col],
                bc[:, quant_col],
                color=(color[StatsBase.sample(1:length(color))], alpha),
                linewidth=linewidth,
            )
        else
            # Plot trajectory
            lines!(
                ax,
                bc[:, time_col],
                bc[:, quant_col],
                color=(color, alpha),
                linewidth=linewidth,
            )
        end # if
    end # for

    # Check if extra label should be added
    if typeof(zero_label) <: String
        # Generate Automatic yticks
        yticks_auto = Makie.get_ticks(
            Makie.LogTicks(Makie.WilkinsonTicks(n_ticks)),
            log10,
            Makie.automatic,
            zero_lim,
            maximum(data[:, quant_col]),
        )
        # Check that the automatically-generated ticks don't match with the
        # zero_lim value Set tick values. This is done by appending the zero_lim
        if yticks_auto[1][1] ≈ zero_lim
            # value with the automatically-generated log ticks Makie creates.
            tickval = [zero_lim; yticks_auto[1][2:end]...]

            # Set tick labels. This is done by taking the "RichText" Makie generates
            # and appending the value of zero_label
            ticklabel = [zero_label; yticks_auto[2][2:end]...]
        else
            # value with the automatically-generated log ticks Makie creates.
            tickval = [zero_lim; yticks_auto[1]...]

            # Set tick labels. This is done by taking the "RichText" Makie generates
            # and appending the value of zero_label
            ticklabel = [zero_label; yticks_auto[2]...]

        end # if
        # Modify y-axis ticks.
        ax.yticks = (tickval, ticklabel)
    end # if
end # function

@doc raw"""
    logfreq_ratio_time_series!((ax, data; color, alpha, id_col, time_col, freq_col)

Function to plot the time series of the log frequency ratiofor a set of
barcodes. This function expects the data in a **tidy** format. This means that
every row represents **a single observation**. For example, if we measure
barcode `i` in 4 different time points, each of these four measurements gets an
individual row. Furthermore, measurements of barcode `j` over time also get
their own individual rows.
    
The `DataFrame` must contain at least the following columns:
- `id_col`: Column identifying the ID of the barcode. This can the barcode
    sequence, for example.
- `time_col`: Column defining the measurement time point.
- `freq_col`: Column with the frequency from which to compute the log ratio.

# Arguments
- `ax::Makie.Axis`: Axis object to be populated with plot.
- `data::DataFrames.AbstractDataFrame`: **Tidy dataframe** with the data to be
    used to sample from the population mean fitness posterior distribution.

## Optional Arguments
- `id_col::Symbol=:barcode`: Name of the column in `data` containing the barcode
    identifier. The column may contain any type of entry.
- `time_col::Symbol=:time`: Name of the column in `data` defining the time point
at which measurements were done. The column may contain any type of entry as
long as `sort` will resulted in time-ordered names.
- `freq_col::Symbol=:count`: Name of the column in `data` containing the barcode
  frequency.
- `color::Union{ColorSchemes.ColorScheme,Symbol,ColorTypes.Colorant{Float64,
3}}=ColorSchemes.glasbey_hv_n256`: Single color or list of colors from
`ColorSchemes.jl` to be used in plot. Note: when a color list is provided,
colors are randomnly assigned to each barcode by sampling from the list of
colors.
- `alpha::AbstractFloat=1.0`: Level of transparency for plots.
- `linewidth::Real=5`: Trajectory linewidth.
- `log_fn::Union{typeof(log), typeof(log10), typeof(log2)}=log`: Log function
  to be used in plot.
"""
function logfreq_ratio_time_series!(
    ax::Makie.Axis,
    data::DF.AbstractDataFrame;
    id_col::Symbol=:barcode,
    time_col::Symbol=:time,
    freq_col::Symbol=:count,
    color::Union{ColorSchemes.ColorScheme,Symbol,ColorTypes.Colorant{Float64,3}}=ColorSchemes.glasbey_hv_n256,
    alpha::AbstractFloat=1.0,
    linewidth::Real=2,
    log_fn::Union{typeof(log),typeof(log10),typeof(log2)}=log
)
    # Group data by id_col
    data_group = DF.groupby(data, id_col)

    # Loop through trajectories
    for bc in data_group
        # Sort data by time
        DF.sort!(bc, time_col)

        # Check if unique color was assigned to each barcode
        if typeof(color) <: ColorSchemes.ColorScheme
            # Plot trajectory
            lines!(
                ax,
                bc[2:end, time_col],
                diff(log_fn.(bc[:, freq_col])),
                color=(color[StatsBase.sample(1:length(color))], alpha),
                linewidth=linewidth,
            )
        else
            # Plot trajectory
            lines!(
                ax,
                bc[2:end, time_col],
                diff(log_fn.(bc[:, freq_col])),
                color=(color, alpha),
                linewidth=linewidth,
            )
        end # if
    end # for
end # function

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
    mcmc_trace_density!(gl::GridLayout, chain::MCMCChains.Chains; colors, labels)

Function to plot the traces and density estimates side-to-side for each of the
parametres in the `MCMCChains.Chains` object.

# Arguments
- `gl::Makie.GridLayout`: GridLayout object to be populated with plot. This
  allows the user to have more flexibility on whether they want to embed this
  plot within other subplots.
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
    gl::GridLayout,
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
    mcmc_fitdist_cdf!(ax, chain, dist; n_points, range, ecdf_label, cdf_label, ecdf_kwargs, cdf_kwargs, legend, legend_kwargs)

Function to plot the ECDF produced from an MCMC chain along with a fitn
distribution. This plot serves to compare if the parametrized distribution
matches the density of MCMC samples.

NOTE: For this function `ecdf` refers to the empirical cumulative distribution
function build from the MCMC chain, and `cdf` refers to the parametric
cumulative distribution function from the fit distribution.

# Arguments
- `ax::Makie.Axis`: Axis object to be populated with plot.
- `chain::Vectors{<:Real}`: Vector with the MCMC samples from which to build the
  ECDF plot.
- `dist::Distributions.ContinuousUnivariateDistribution`: Parametric
  distribution to be compared with the ECDF.

## Optional Arguments
- `npoints::Int=1000`: Number of points to evaluate the parametric CDF.
- `range::Union{Nothing,NTuple{2,<:Real}}=nothing`: Range on which to evaluate
  the parametric CDF. If `nothing` is provided (default), the range is inferred
  from the range of MCMC samples.
- `ecdf_label::String="mcmc"`: Legend label for the ECDF plot.
- `cdf_label::String="fit"`: Legend label for the CDF plot.
- `ecdf_kwargs::Dict`: Extra keyword arguments for the `Makie.ecdfplot!`
  function.
- `cdf_kwargs::Dict`: Extra keyword arguments for the `Makie.lines!` function.
- `legend::Bool=true`: Boolean indicating if a legend should be added to the
  plot.
- `legend_kwargs:Dict()`: Extra keyword arguments for the `Makie.axislegend`
  function.
"""
function mcmc_fitdist_cdf!(
    ax::Makie.Axis,
    chain::Vector{<:Real},
    dist::Distributions.ContinuousUnivariateDistribution;
    npoints::Int=1000,
    range::Union{Nothing,NTuple{2,<:Real}}=nothing,
    ecdf_label::String="mcmc",
    cdf_label::String="fit",
    ecdf_kwargs::Dict=Dict(:linewidth => 2.5),
    cdf_kwargs::Dict=Dict(
        :color => :black, :linewidth => 2.5, :linestyle => :dot
    ),
    legend::Bool=true,
    legend_kwargs::Dict=Dict(:position => :rb)
)
    # Plot ECDF
    ecdfplot!(
        ax,
        chain,
        npoints=length(chain);
        label=ecdf_label,
        ecdf_kwargs...
    )

    # Define range if not given
    if typeof(range) <: Nothing
        range = (minimum(chain), maximum(chain))
    end # if

    # Plot CDF
    lines!(
        ax,
        LinRange(range..., npoints),
        Distributions.cdf(dist, LinRange(range..., npoints));
        label=cdf_label,
        cdf_kwargs...
    )

    # Check if legend should be added
    if legend
        axislegend(ax; legend_kwargs...)
    end # if
end # function

@doc raw"""
    ppc_time_series!(ax, quantile, ppc_mat; time, colors, alpha)

Function to plot the posterior predictive checks quantiles for any quantity.

# Arguments
- `ax::Makie.Axis`: Axis object to be populated with plot. 
- `quantile::Vector{<:AbstractFloat}`: List of quantiles to extract from the
    posterior predictive checks.
- `ppc_mat::Matrix{<:AbstractFloat}`: Matrix containing the posterior predictive
  samples. Rows are assumed to contain the samples, columns the time points.

## Optional arguments
- `colors=ColorSchemes.Blues_9`: List of colors to use for each quantile.
- `alpha::AbstractFloat=0.75`: Level of transparency for band representing each
quantile.
"""
function ppc_time_series!(
    ax::Makie.Axis,
    quantile::Vector{<:AbstractFloat},
    ppc_mat::Matrix{<:Real};
    time::Union{Vector{<:Real},Nothing}=nothing,
    colors::Union{ColorSchemes.ColorScheme,Vector{<:ColorTypes.Colorant{Float64,3}}}=ColorSchemes.Blues_9,
    alpha::AbstractFloat=0.75
)
    # Check that all quantiles are within bounds
    if any(.![0.0 ≤ x ≤ 1.0 for x in quantile])
        error("All quantiles must be between zero and one")
    end # if

    # Tell user that quantiles will be sorted
    if quantile != sort(quantile, rev=true)
        println("Notice that we sort the quantiles to properly display the intervals")
    end # if

    # Check that there are enough colors for each quantile
    if length(colors) < length(quantile)
        error("There are not enough colors provided for all quantiles")
    end # if

    # Sort quantiles
    sort!(quantile, rev=true)

    # Compute quantiles
    ppc_quant = matrix_quantile_range(quantile, ppc_mat)

    # Check if time is provided
    if typeof(time) <: Nothing
        time = collect(1:size(ppc_quant, 1))
    end # if

    # Loop through quantiles
    for i in eachindex(quantile)
        # Add confidence interval for observation
        band!(
            ax,
            time,
            ppc_quant[:, i, 1],
            ppc_quant[:, i, 2],
            color=(colors[i], alpha)
        )
    end # for
end # function
