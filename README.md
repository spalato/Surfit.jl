# Surfit: fitting simple data to complicated models

Surfit provides bounded surrogate-accelerated least-squares fitting to efficiently fit costly non-differentiable models to experimental data, with helpers for least-squares objectives and persistent HDF5-backed function evaluation caches. All arguments and optimizer parameters are on data scales.

Compared to Levenberg-Marquart, surfit finds equivalent
results, and tends to require less evaluations (about ~25% less) for single evaluations. Compared to Nedler-Mead simplex, `surfit` finds better results (lower minima, closer to LM),
with 1/2 to 1/4 of the function evaluations. Benchmarking battery requires expansion. `surfit` can reuse previous results, greatly accelerating multiple
fits to the same model, which none of these existing algorithms can do.

The logic is to use an approximation of the function to accelerate convergence
for very costly functions of a few parameters.
The algorithm is initialized by sampling points randomly between the bounds.
Then, a "surrogate" estimating the function is constructed, currently using
interpolation by radial basis functions (`cubicRadial` from `Surrogates.jl`).
A deterministic algorithm, currently NM simplex, then minimizes the residuals
for the surrogate function vs the target data. This provides a new candidate,
which is then evaluated using the true function. The true result is used to
refine the surrogate for the next minimization step. This process is repeated
until convergence, and convergence is only evaluated using true results.

The results of the (costly) evaluations are stored in a data cache, which can
be reused between runs. Therefore, multiple fits using the same model are
greatly accelerated (needs quantification).



## Installation

From the repository root, activate and instantiate the project:

```julia
julia --project=.
```

```julia
] instantiate
```

For a registered package installation, use Julia's package manager:

```julia
] add Surfit
```

## Basic Usage

Define a vectorized model and its observations, then call `surfit` with an
initial parameter guess and lower and upper bounds:

```julia
using Surfit

model(x, a, b) = a .* exp.(-b .* x)
x = collect(0.0:0.1:2.0)
y = model(x, 2.0, 0.5)

guess = [1.5, 0.7]
lower = [0.5, 0.1]
upper = [3.0, 1.5]

popt, value, samples, sample_values = surfit(
    model,
    x,
    y,
    guess,
    lower,
    upper,
    1e-8, # x_tol
    1e-10, # f_tol
    100, # max f_calls
)
```

`popt` contains the fitted parameters, `value` is the sum of squared residuals,
`samples` contains evaluated parameter points, and `sample_values` contains the
corresponding model outputs.

For a scalar objective, use `surfit_scalar`:

```julia
objective(p) = sum((p .- [2.0, 0.5]).^2)

popt, value, samples, sample_values = surfit_scalar(
    objective,
    [1.5, 0.7],
    [0.5, 0.1],
    [3.0, 1.5],
    1e-8,
    1e-10,
    100,
)
```

Initial guesses must be inside the supplied bounds, and model functions used by
`surfit` should return a vector with the same shape as the observations.

## Residual Helpers

`resid_vs` creates a residual function and `ssq_of` converts it into a scalar
sum-of-squares objective:

```julia
residual = resid_vs(y, p -> model(x, p...))
objective = ssq_of(residual)
```

## Persistent Evaluation Cache

Use `stored_scalar` or `stored` with an open HDF5 store to cache expensive model
evaluations between runs:

```julia
using HDF5
using Surfit

h5open("surfit_cache.h5", "a") do store
    cached_model = stored(model, store)
    prediction = cached_model(x, 2.0, 0.5)
end
```

The cache validates the independent-variable data when retrieving a stored
vector-valued result.

## Testing

Run the package test suite from the repository root:

```powershell
julia +lts --project=. -e "using Pkg; Pkg.test()"
```

You can run the same suite with another installed Julia channel:

```powershell
julia +release --project=. -e "using Pkg; Pkg.test()"
```

## Development

The package source is in `src/`, and tests are in `test/`. Use Julia's project
activation with `--project=.` when developing so the repository environment is
used consistently.
