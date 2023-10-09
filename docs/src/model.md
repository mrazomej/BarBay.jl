# model

In this section, we list the available models to be fit with either [mcmc](@ref)
or [vi](@ref). To see examples on how to implement these models, please check
the [examples](@ref) tab.

## Single-dataset single-environment

```@docs
BarBay.model.fitness_normal
```

## Single-dataset multi-environment
```@docs
BarBay.model.multienv_fitness_normal
```

## Single-dataset hierarchical model on genotypes
```@docs
BarBay.model.genotype_fitness_normal
```

## Multi-replicate single-environment hierarchical model for experimental replicates
```@docs
BarBay.model.replicate_fitness_normal
```

## Multi-replicate multi-environment hierarchical model for experimental replicates
```@docs
BarBay.model.multienv_replicate_fitness_normal
```