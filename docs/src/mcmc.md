# mcmc

The `mcmc` module contains functions necessary to fit the statistical model via
a Markov Chain Monte Carlo sampling-based method. MCMC is guaranteed to converge
to the "true" posterior distribution. Therefore, for small number of barcodes (≈
100-250) we recommend trying this approach. To scale the analysis, please check
the [vi](@ref) module for variational inference.

```@autodocs
Modules = [BarBay.mcmc]
Order   = [:function, :type]
```
