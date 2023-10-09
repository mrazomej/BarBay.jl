# contributing

!!! warning "Disclaimer"
    The authors of this package have no formal training in software engineering.
    But we are eager to learn about best practices and are very open to
    suggestions via GitHub issues or PRs!


We welcome contributions to the package via pull requests. One such example
could be adding a new variation of the base model to the [model](@ref) module.
All models within `BayesFitness.jl` that can be fit with the [mcmc](@ref) or the
[vi](@ref) modules are standardized to take as the first four arguments the
following:

- `R̲̲`: Array-like object that contains the raw barcode counts. These can be
  either a matrix, a tensor, or a list of arrays. The main feature is that each
  "face" of the array-like object represents a matrix where each column contains
  the time series data for a single barcode. Here are some examples of the types
  used in different models:
  - `R̲̲::Matrix{Int64}`: `T × B` matrix--split into a vector of vectors for
    computational efficiency--where `T` is the number of time points in the data
    set and `B` is the number of barcodes. Each column represents the barcode
    count trajectory for a single lineage.
  - `R̲̲::Array{Int64, 3}`:: `T × B × R` where `T` is the number of time points
    in the data set, `B` is the number of barcodes, and `R` is the number of
    experimental replicates. For each slice in the `R` axis, each column
    represents the barcode count trajectory for a single lineage.
  - `R̲̲::Vector{Matrix{Int64}}`:: Length `R` vector wth `T × B` matrices
      where `T` is the number of time points in the data set, `B` is the number
      of barcodes, and `R` is the number of experimental replicates. For each
      matrix in the vector, each column represents the barcode count trajectory
      for a single lineage.
- `n̲ₜ`: Array-like object with the total number of barcode counts for each time
  point. As with `R̲̲`, the structure of `n̲ₜ` must be adapted for the needs of
  your data structure. Here are some examples of the types used in different
  models:
  - `n̲ₜ::Vector{Int64}`: Vector with the total number of barcode counts for
    each time point. **NOTE**: This vector **must** be equivalent to computing
    `vec(sum(R̲̲, dims=2))`.
  - `n̲ₜ::Vector{Vector{Int64}}`: Vector of vectors with the total number of
    barcode counts for each time point on each replicate. **NOTE**: This vector
    **must** be equivalent to computing `vec.(sum.(R̲̲, dims=2))`.
- `n_neutral::Int`: Number of neutral lineages in dataset. 
- `n_bc::Int`: Number of mutant lineages in dataset.

The rest of the arguments fed to the model function must be optional keyword
arguments. This means that they should be listed after the semi-colon that
separates the first four inputs from the rest and they should have a default
value. As an example, take a look at the definition of the input arguments for
one of the base models:

```julia
Turing.@model function fitness_normal(
    R̲̲::Matrix{Int64},
    n̲ₜ::Vector{Int64},
    n_neutral::Int,
    n_bc::Int;
    s_pop_prior::VecOrMat{Float64}=[0.0, 2.0],
    logσ_pop_prior::VecOrMat{Float64}=[0.0, 1.0],
    s_bc_prior::VecOrMat{Float64}=[0.0, 2.0],
    logσ_bc_prior::VecOrMat{Float64}=[0.0, 1.0],
    logλ_prior::VecOrMat{Float64}=[3.0, 3.0]
)
```

Furthermore, the four inputs to the model are automatically generated within the
[mcmc](@ref) and the [vi](@ref) modules via the `data_to_arrays` function from
the [utils](@ref) module. Take a look at the source code for this function to
familiarize yourself with it. In case this is inconvenient, please open an issue
in the [GitHub repository](https://github.com/mrazomej/BayesFitness.jl) and we
will be happy to make changes to adapt the package to your needs!

If your defined model follows these standards, you should be able to fit it with
the tools provided within this package. Here are some resources to get familiar
with the model structure within [`Turing.jl`](https://turinglang.org/stable/):

- [Getting Started with `Turing.jl`](https://turinglang.org/v0.29/docs/using-turing/get-started)
- [Bayesian Statistics using Julia and Turing](https://storopoli.io/Bayesian-Julia/)

Furthermore, we recommend looking at the source code within [this package GitHub
repository](https://github.com/mrazomej/BayesFitness.jl). Every function and
model is highly annotated and can serve as a guide to get you going!