# BayesFitness

Welcome to the documentation of `BayesFitness.jl`! The accompanying paper,
*Bayesian inference of relative fitness on high-throughput pooled competition
assays*, explains all of the biological and mathematical background needed to
understand this package. Here, we only focus on how to use the package, assuming
the user already understands the objective of inferring the posterior
probability distribution of the relative fitness of mutant strains in a pooled
competition assay.

The package is divided into modules. Here's a brief description of the content
of each module, but please visit their respective documentations to understand
what each module is intended for.

- `utils`: Series of miscellaneous functions that make the data wrangling and
  processing much simpler.
- `viz`: [`Makie.jl`](https://docs.makie.org/stable/)-based module with useful
  plotting functions to display the data and the MCMC results for visual
  diagnostics.
- `stats`: Statistical functions used in the inference problem.
- `model`: [`Turing.jl`](https://turing.ml)-based Bayesian models used to infer
  the population mean fitness via the neutral lineages as well as the mutants'
  relative fitness.
- `mcmc`: The main module with which to perform the Markov-Chain Monte Carlo
  sampling of the posterior distributions.
  
## Example inference

To get you going with the package, let's walk through a basic inference pipeline
for one competition assay. Our ultimate goal consists of inferring the relative
fitness for each of the mutant barcodes. To that end, we assume that the
frequency time-series obeys the following equation
```math
f_{t+1}^{(b)} = f_{t}^{(b)} \mathrm{e}^{\left(s^{(b)} - \bar{s}_t \right)\tau},
\tag{1}
```
where ``f_{t}^{(b)}`` is the frequency of barcode ``b`` at time ``t``,
``s^{(b)}`` is the relative fitness of this barcode, ``\bar{s}_t`` is the
population mean fitness at time ``t``, and ``\tau`` is the time interval between
time ``t`` and ``t+1``.

The first step consists of importing the necessary packages. 

!!! note 
    We use `import` rather than the more common `using` command. We find it
    better to keep the project organized, but feel free to use whatever is more
    convenient for you!

```julia
# Import Bayesian inference package
import BayesFitness

# Import libraries to manipulate data
import DataFrames as DF
import CSV
```

After having imported the libraries, we need to load our dataset into memory.
This dataset is already in the format needed for `BayesFitness.jl` to work, so
we don't have to modify anything.
```julia
# Import data
data = CSV.read("~/git/BayesFitness/test/data/data_example_01.csv", DF.DataFrame)
```
Here you will replace `"~/git/BayesFitness/test/data"` with the directory where
your data is stored, and `"data_example_01.csv"` with the name of the file
containing the data. The resulting `DataFrame` looks something like this:
```
| BCID_x | barcode                                               | name                    | count | time | neutral | count_sum  |
|--------|-------------------------------------------------------|-------------------------|-------|------|---------|------------|
| 0      | TGATCAATCTACAAAAATATTTAATG_GAGTGAAACATGAATGGTATTCATCA | Batch1_1Day-T0_combined | 53    | 0    | FALSE   | 543947     |
| 1      | CCGCCAATCCCGAACCCCGTTTCGCC_ACTCTAACGTGTAACTAATTTTGAGT | Batch1_1Day-T0_combined | 1213  | 0    | FALSE   | 543947     |
| 2      | GACAGAAAAGCCAAATGGATTTACCG_ATGGGAACACGGAATGATCTTTTATT | Batch1_1Day-T0_combined | 17    | 0    | FALSE   | 543947     |
| 3      | CCAACAAAACACAAATCTGTTGTGTA_TACTAAATAAGTAAGGGAATTCTGTT | Batch1_1Day-T0_combined | 19    | 0    | FALSE   | 543947     |
| 4      | TATCGAAACCCAAAGAGATTTAATCG_ATGACAAACTTTAAATAATTTAATTG | Batch1_1Day-T0_combined | 23    | 0    | FALSE   | 543947     |
| 5      | TATCGAAACCCAAAGAGATTTAATCG_CGATCAAAGACTAACTTATTTTGTGG | Batch1_1Day-T0_combined | 16    | 0    | FALSE   | 543947     |
| 6      | TATCGAAACCCAAAGAGATTTAATCG_TTGCCAAGCTGGAAAGCTTTTTATGA | Batch1_1Day-T0_combined | 12    | 0    | FALSE   | 543947     |
| 7      | ATCACAATAACTAAACTGATTCTTCA_CTCATAACATCAAAAAAAATTCAAAT | Batch1_1Day-T0_combined | 161   | 0    | FALSE   | 543947     |
| 8      | TATCGAAACCCAAAGAGATTTAATCG_GTTTAAACCATTAATTATATTAGATC | Batch1_1Day-T0_combined | 19    | 0    | FALSE   | 543947     |
```
The relevant columns in this data frame are:
- `barcode`: The unique ID that identifies the barcode.
- `count`: The number of reads for this particular barcode.
- `time`: The time point ID indicating the order in which samples were taken.
- `neutral`: Indicator of whether the barcode belongs to a neutral lineage or
  not.

Let's take a look at the data. The [`BayesFitness.viz`](./viz.md) module has
several [`Makie.jl`](https://docs.makie.org/stable/)-based functions to easily
display the data. Let's import the necessary plotting libraries

```julia
# Import plotting libraries
using CairoMakie
import ColorSchemes
```

First, we plot the barcode frequency trajectories. For this, we use the
convenient [`BayesFitness.viz.bc_time_series!`](@ref) function.

```julia
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

# Plot Mutant barcode trajectories with varying colors
BayesFitness.viz.bc_time_series!(
    ax,
    data[.!data.neutral, :];
    quant_col=:freq,
    zero_lim=1E-9,
    zero_label="extinct",
    alpha=0.25,
    linewidth=2
)

# Plot Neutral barcode trajectories with a single dark blue color
BayesFitness.viz.bc_time_series!(
    ax,
    data[data.neutral, :];
    quant_col=:freq,
    zero_lim=1E-9,
    color=ColorSchemes.Blues_9[end],
    alpha=0.9,
    linewidth=2
)
```

We highlight the neutral barcodes⸺defined to have relative fitness
``s^{(n)}=0``⸺with dark blue lines. The rest of the light-color lines correspond
to individual barcodes.

![](./figs/fig01.svg)

We can rewrite Eq. (1) as
```math
\frac{1}{\tau} \ln \frac{f_{t+1}^{(b)}}{f_{t}^{(b)}} = 
\left(s^{(b)} - \bar{s}_t \right).
\tag{2}
```
In this form, we can se that the relevant quantity we need to infer the values
of the population mean fitness ``\bar{s}_t`` and the barcode relative fitness
``s^{(b)}`` are not the frequencies themselves, but the log ratio of these
frequencies between two adjacent time points. Let's plot this log frequency
ratio using the [`BayesFitness.viz.logfreq_ratio_time_series!`](@ref) function.
```julia
# Initialize figure
fig = Figure(resolution=(450, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="ln(fₜ₊₁/fₜ)",
    title="log-frequency ratio"
)

# Plot log-frequency ratio of mutants with different colors
BayesFitness.viz.logfreq_ratio_time_series!(
    ax,
    data[.!data.neutral, :];
    freq_col=:freq,
    alpha=0.25,
    linewidth=2
)

# Plot log-frequency ratio of neutrals with a single dark blue color
BayesFitness.viz.logfreq_ratio_time_series!(
    ax,
    data[data.neutral, :];
    freq_col=:freq,
    color=ColorSchemes.Blues_9[end],
    alpha=1.0,
    linewidth=2
)
```
From this, we can see that the neutral lineages follow very similar
log-frequency ratio trajectories as expected.

![](./figs/fig02.svg)

### Inferring the population mean fitness

With the data in hand, our first task is to infer the population mean fitness
using the neutral lineages. For this, we use the
[`BayesFitness.mcmc.mcmc_mean_fitness`](@ref) function from the `mcmc` module.
The main parameters we need to define are:
- `:data`: Tidy data frame containing the raw barcode counts.
- `:n_walkers`: Number of MCMC chains to run in parallel. NOTE: Having multiple
  chains run in parallel is convenient for diagnostics. `BayesFitness.jl` will
  use the available threads, so make sure you have more than one thread in your
  `julia` session if you want to run this inference in a multi-threaded way.
- `:outputdir`: String pointing to the output directory.
- `outputname`: String defining the pattern for the output files. This can be
  something related to the dataset. For example, the growth media, or the date
  of the experiment, of whatever metadata used to distinguish different
  datasets.
- `model`: Bayesian model from the `model` module that defines the posterior
  distribution to be sampled.
- `model_kwargs`: The parameters required by the `model` function.

We compile all of these parameters into a dictionary that looks something like
this:
```julia
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
```

We are now ready to sample the posterior distribution for the population mean
fitness. [`BayesFitness.mcmc`](./mcmc.md) makes this extremely easy by using the
[`BayesFitness.mcmc.mcmc_mean_fitness`](@ref) function. All we have to do is run
```julia
# Run inference
BayesFitness.mcmc.mcmc_mean_fitness(; param...)
```
The output of this function are [`jld2`](https://github.com/JuliaIO/JLD2.jl)
files that save the native data structure. To extract the MCMC samples of the
variable we care about⸺equivalent to marginalizing out all the nuisance
variables⸺we can use the [`BayesFitness.utils.var_jld2_to_df`](@ref) from the
[`BayesFitness.utils`](./utils.md) module, indicating the name of the variable
we want to extract.
```julia
BayesFitness.utils.var_jld2_to_df("./output/", "data_01_meanfitness", :sₜ)
```

To diagnose the inference of this mean fitness value, it is useful to plot
```julia
# Concatenate population mean fitness chains into single chain
chains = BayesFitness.utils.var_jld2_concat(
    param[:outputdir], param[:outputname], :sₜ
)

# Initialize figure
fig = Figure(resolution=(600, 600))

# Generate mcmc_trace_density! plot
BayesFitness.viz.mcmc_trace_density!(fig, chains; alpha=0.5)
```

![](./figs/fig03.svg)

### Inferring mutants' relative fitness
Once we make sure that the mutants relative fitness looks okay, we can tackle
the inference of each mutant relative fitness. The process is very similar, the
main difference being that we use the results from the previous step as part of
the inputs that go into the corresponding Bayesian model defined in the `model`
module. More specifically, the inferred population mean fitness enters our
inference as a "prior" on this value. This prior has to be parametrized, for
which we chose a Gaussian distribution. So we need to fit one Gaussian
distribution for each MCMC chain sampled in the previous section. The
[`BayesFitness.stats.gaussian_prior_mean_fitness`](@ref) function in the `stats`
module can help us with this.
```julia
# Infer mean fitness distributions
mean_fitness_dist = BayesFitness.stats.gaussian_prior_mean_fitness(
    BayesFitness.utils.var_jld2_to_df("./output/", "data_01_meanfitness", :sₜ)
)
```
We can now define the dictionary containing the parameters that go into the
[`BayesFitness.mcmc.mcmc_mutants_fitness_multithread`] function from the `mcmc`
module. 

!!! note
    Notice that we can either use the
    [`BayesFitness.mcmc.mcmc_mutants_fitness_multithread`] or the
    [`BayesFitness.mcmc.mcmc_mutants_fitness`] function. The difference being
    that the first one can run multiple mutants simultaneously, but only one
    chain at the time, while the second one can run the chains in parallel, but
    only one mutant at the time. You might need to use either depending on what
    you are trying to do, but we recommend the `_multithread` function to speed
    up the inference of thousands of fitness values.

```julia
# Define function parameters
param = Dict(
    :data => data,
    :n_walkers => 3,
    :n_steps => 1_000,
    :outputdir => "./output/",
    :outputname => "data_01_mutantfitness",
    :model => BayesFitness.model.mutant_fitness_lognormal,
    :model_kwargs => Dict(
        :α => BayesFitness.stats.beta_prior_mutant(
            data[data.time.==0, :barcode],
        ),
        :μ_s̄ => mean_fitness_dist[1],
        :σ_s̄ => mean_fitness_dist[2],
    )
)
```
Finally, we run the inference.
```julia
# Run inference
BayesFitness.mcmc.mcmc_mutant_fitness_multithread(; param...)
```

## Contents

```@contents
```