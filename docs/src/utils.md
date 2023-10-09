# utils

The `utils` module contains useful functions to handle the raw data and the
output results.

The `data_to_arrays` function takes the tidy dataframes and converts it into the
set of arrays used as input for all the models in the [model](@ref) module. This
function has different options to build the slight variations needed for each of
the models.

```@docs
BayesFitness.utils.data_to_arrays
```

The `advi_to_df` function takes the output when fitting a model performing
variational inference **using the mean-field approximation**, i.e., assuming a
diagonal covariance matrix.

```@docs
BayesFitness.utils.advi_to_df
```
