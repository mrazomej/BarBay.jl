# BarBay.jl

[![codecov](https://codecov.io/gh/mrazomej/BarBay.jl/graph/badge.svg?token=W2JREEH8O8)](https://codecov.io/gh/mrazomej/BarBay.jl)

BarBay.jl is a Julia package for performing Bayesian inference of relative fitness from high-throughput barcode sequencing data. The package implements robust statistical models to analyze competitive fitness assays and estimate relative fitness values with quantified uncertainty.

## Features

- 🧬 Analyze barcode sequencing data from competitive fitness assays
- 📊 Support for multiple experimental designs:
    - Single condition experiments
    - Multiple experimental replicates
    - Multiple environmental conditions
    - Multiple genotype groupings
- 📈 Bayesian inference using:
    - Variational inference with ADVI
- 🎯 Built-in support for:
    - Population mean fitness estimation
    - Mutant fitness effects
    - Hierarchical models for replicates and genotypes
    - Environment-specific fitness effects

## Installation

```julia
using Pkg
Pkg.add("BarBay")
```

## Quick Start

Here's a simple example of analyzing fitness data from a single condition
experiment:

```julia
using BarBay
using CSV
using DataFrames

# Load your data (must be in tidy format)
data = DataFrame(
    :barcode => [...],    # Barcode identifiers
    :time => [...],       # Time points
    :count => [...],      # Raw counts
    :neutral => [...]     # Boolean indicating neutral lineages
)

# Run Bayesian inference using ADVI
results = BarBay.vi.advi(
    data = data,
    model = BarBay.model.fitness_normal,  # Basic fitness model
    outputname = "fitness_results"        # Optional: save to CSV
)

# The results DataFrame contains posterior distributions for:
# - Population mean fitness
# - Individual mutant fitness effects
# - Associated uncertainty measurements
```

For more complex experimental designs:

```julia
# Multiple replicates
results = BarBay.vi.advi(
    data = data,
    model = BarBay.model.replicate_fitness_normal,
    rep_col = :replicate  # Column indicating replicates
)

# Multiple environments
results = BarBay.vi.advi(
    data = data,
    model = BarBay.model.multienv_fitness_normal,
    env_col = :environment  # Column indicating environments
)

# Multiple genotypes
results = BarBay.vi.advi(
    data = data,
    model = BarBay.model.genotype_fitness_normal,
    genotype_col = :genotype  # Column indicating genotypes
)
```

## Documentation

For detailed documentation, tutorials, and API reference, please visit our
[documentation page](https://mrazomej.github.io/BarBay.jl/stable/).

## Contributing

We welcome contributions! Please feel free to:

- Open issues for bug reports or feature requests
- Submit pull requests
- Suggest improvements to documentation
- Share example use cases

## Citation

If you use BarBay.jl in your research, please cite:

Razo-Mejia, M., Mani, M. & Petrov, D. Bayesian inference of relative fitness on
high-throughput pooled competition assays. *PLoS Comput Biol* 20,
[e1011937](https://dx.plos.org/10.1371/journal.pcbi.1011937) (2024).

## License

BarBay.jl is released under the MIT License.

## Contact

Please open an issue on this repository if you have any questions, comments, or
would like to contribute.